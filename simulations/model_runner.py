"""
main class for running and saving models
"""

import numpy as np
import os
import simulations.auxin_model as am
#import voxelizer as vx

import pyvista as pv

import scipy.spatial as spsp

from mayavi import mlab
import matplotlib.pyplot as plt

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


class Vasculature:

    model_type = None
    init_num_pts = None
    save_dir = None
    voxel_size = 256


    edges = None
    d_nbrs = None
    all_nbrs = None
    coords = None
    A = None

    init_radii = 0.01
    radii = None
    edge_radii = None

    SDF = None

    voxels = None

    fundus_image = None

    def __init__(self,mt,inp,save_dir):
        self.model_type = mt
        self.init_num_pts = inp
        self.save_dir = save_dir

        return

    def add_tumor(self):
        #???

        return

    def set_geometry(self, od=None, fovea=None):
        if od is not None:
            am.g_od_pos = od
        if fovea is not None:
            am.g_fovea_pos = fovea

    def run_simulation(self,step_size = 0.9,fovea_radius = 0.2,init_vasc = None,bounding_box=None):
        if self.model_type == "auxin":
            #run simulation
            result = am.vascular_growth_sim(fovea_radius = fovea_radius, init_num_pts = self.init_num_pts,D_step = step_size,init_vasc = init_vasc,bounding_box=bounding_box)

            #convert back to Euclidean coords
            self.coords = am.convert_from_product(result[0])/1.2
            init_sample = am.convert_from_product(result[-1])/1.2

            branches = result[1]
            branch_membership = result[2]

            #extract
            self.edges, self.A,self.d_nbrs = am.extract_graph(len(self.coords),branches)
            all_nbrs = {i:[] for i in range(len(self.A))}
            for e in self.edges:
                all_nbrs[e[0]].append(e[1])
                all_nbrs[e[1]].append(e[0])
            self.all_nbrs = all_nbrs

            self.edge_lookup = {tuple(np.sort(e)):i for i,e in enumerate(self.edges)}

            self.radii = am.get_vein_radii(self.d_nbrs,self.A,init_radii = self.init_radii,branch_power = 3)
#            import pdb; pdb.set_trace()

        return

    def generate_radii(self,init_r = 0.01):
        self.init_radii = init_r
        self.radii = am.get_vein_radii(self.d_nbrs,self.A,init_radii = init_r,branch_power = 3)
        self.edge_radii = np.array([np.mean(self.radii[list(e)]) for e in self.edges])
        return

    def voxelize(self,fast_marching = True):
        if self.edges is None:
            self.run_simulation()

        self.voxels = vx.generate_voxels(self.coords,self.voxel_size,self.edges,self.radii, fast_marching = fast_marching)

        return

    def generate_fundus_image(self,im_type="voxel",save = False,save_name = ""):

        if im_type == "exact":
            #set up circle mesh
            thetas = 2.*np.pi*np.linspace(0,1,100)
            xx = [1.1*np.cos(t) for t in thetas]
            yy = [1.1*np.sin(t) for t in thetas]
            xx.append(0); yy.append(0)
            zz = [0. for i in xx]
            tris = [[i,i+1,len(xx)-1] for i in range(len(xx)-2)]
            tris.append([len(xx)-2,0,len(xx)-1])
            tris = np.array(tris)

            #draw mesh
            mlab.figure(bgcolor=(0.,0.,0.), size=(1000,1000))
            mlab.triangular_mesh(xx,yy,zz,tris,
                    opacity = 0.5,color = (0.95,0.7,0.1))


            proj_pts, proj_e,proj_idx = am.project_points(self.coords, self.A)

            src = mlab.plot3d(-proj_pts[:,0],proj_pts[:,1],[0. for p in proj_pts],0.01*self.radii[proj_idx],color=(1.,0.,0.))
            src.parent.parent.filter.vary_radius = 'vary_radius_by_absolute_scalar'
            src.mlab_source.dataset.lines = proj_e
            lines = mlab.pipeline.stripper(src)

#            import pdb; pdb.set_trace()
            if save:
                mlab.savefig(f"{self.save_dir}plots/{save_name}_exact-vein_radius-{self.init_radii:.3f}.png", size = (300,300))
                mlab.close("all")
            else:
                mlab.show()
        elif im_type == "voxel":
            if self.voxels is None:
                self.voxelize()
            test_image = np.zeros((self.voxel_size+1,self.voxel_size+1))

            for v in self.voxels:
                if v[2] <= self.voxel_size/3.:
                    test_image[int(v[0]),int(v[1])] += 1

            plt.imshow(test_image)
            if save:
                plt.savefig(f"{self.save_dir}plots/{save_name}_voxel-vein_radius-{self.init_radii:.3f}.png", size = (300,300))
                plt.close("all")
            else:
                plt.show()
        return

    def generate_SDF(self):

        if self.radii is None:
            self.generate_radii()

        inner_KDTREE = spsp.KDTree(self.coords)

        def inner_SDF(pt):
            nns = inner_KDTREE.query(pt,k=2)[1]

            min_pair = None
            min_dist = np.inf

            for ii in nns:
                for jj in self.all_nbrs[ii]:
                    proposed_dist = dist_to_line(pt,self.coords[ii],self.coords[jj])
                    if proposed_dist < min_dist:
                        min_pair = (ii,jj)
                        min_dist = proposed_dist


            return min_dist - self.edge_radii[self.edge_lookup[tuple(np.sort([ii,jj]))]]

        self.SDF = lambda x: inner_SDF(x)
        return

    def save_simulation(self,save_name = ""):

        np.save( pj(self.save_dir, f"plots/{save_name}_edges") ,self.edges )
        np.save( pj(self.save_dir, f"plots/{save_name}_coords") ,self.coords )

        return

    def load_simulation(self,save_name = "",model_type = "auxin"):

        self.edges = np.load( pj(self.save_dir, f"plots/{save_name}_edges") )
        self.coords = np.load( pj(self.save_dir, f"plots/{save_name}_coords") )
        self.init_num_pts = len(self.coords)
        self.model_type = model_type

        return

    def save_radii(self,save_name=""):

        np.save(f"{self.save_dir}plots/{save_name}_radii-vein_radius-{self.init_radii:.3f}",self.radii)

        return

    def load_radii(self,init_radii,save_name=""):

        self.radii = np.load(f"{self.save_dir}plots/{save_name}_radii-vein_radius-{init_radii:.3f}")

        return

    def save_voxels(self,save_name=""):

        np.save(f"{self.save_dir}plots/{save_name}_voxel_centers-vein_radius-{self.init_radii:.3f}",self.voxels)
        np.save(f"{self.save_dir}plots/{save_name}_voxel_fundus_image-vein_radius-{self.init_radii:.3f}",self.fundus_image)

        return

    def load_voxels(self,init_radii,save_name=""):

        self.voxels = np.save(f"{self.save_dir}plots/{save_name}_voxel_centers-vein_radius-{init_radii:.3f}")
        self.fundus_image = np.save(f"{self.save_dir}plots/{save_name}_voxel_fundus_image-vein_radius-{init_radii:.3f}")

        return

    def save_all_csv(self,save_name = ""):
        #get and save coords

        #get and save adj

        #get and save adjD

        #get and save leaves
        return

def dist_to_line(pt, line_pt1, line_pt2):
    """
    returns the distance of a point pt to the line spanned by line_pt1 and line_pt2

    :param pt: np.array; the point in question
    :param line_pt1: np.array; one endpoint of the line segment in question
    :param line_pt2: np.array; another endpoint of the line segment in question

    :return dist: the distance from the point to the line
    """
    try:
        #print(f"pt: {pt}, line_pt1: {line_pt1}, line_pt2: {line_pt2}")
        s1 = line_pt2 - line_pt1
        s1 /= np.linalg.norm(s1)

        dist = np.linalg.norm((pt - s1) - np.dot(pt-s1,s1)*s1)

        return dist
    except:
        return np.linalg.norm(pt - line_pt1)
