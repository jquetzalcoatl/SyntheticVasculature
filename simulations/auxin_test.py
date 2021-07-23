import model_runner as mr

import pyvista as pv

import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

#run the simulation
test_sim = mr.Vasculature("auxin", 200, "/Users/wham/Documents/vessel_sims/")

time1 = time.time()
test_sim.run_simulation()
time2 = time.time()
print(f"Time to run sim was: {time2-time1}")

test_sim.generate_radii(0.001)

test_sim.generate_SDF()

test_sim.SDF([0.,0.,0.])

xx = np.linspace(np.min(test_sim.coords), np.max(test_sim.coords),100)

"""
time1 = time.time()
test_grid_values = [test_sim.SDF([ii,jj,kk]) for ii in xx for jj in xx for kk in xx ]
time2 = time.time()

print(time2-time1)
"""


#draw cone for field of view
#view_cone = pv.Cone(center = (0.,0.,1.) , direction = (0.,0.,-1.), height = 2. , angle = 30.)
#view_cone.plot()

"""
pyvista looks best for images, mayavi for mesh processing?
"""

if len(test_sim.radii) == len(test_sim.coords):
    #adapt vertex-based radii to edge-based radii
    edge_radii = np.array([np.mean(test_sim.radii[list(e)]) for e in test_sim.edges])
else:
    edge_radii = test_sim.radii

#test surface 2
time1 = time.time()

poly = pv.PolyData()
poly.points = test_sim.coords

cells = np.full((len(test_sim.edges), 3), 2, dtype=np.int_)

cells[:, 1] =  test_sim.edges[:,0]
cells[:, 2] = test_sim.edges[:,1]
poly.lines = cells

#poly["scalars"] = np.arange(poly.n_points)
poly["radii"] = test_sim.radii

tube = poly.tube(radius = np.min(test_sim.radii),scalars = "radii",capping = False)
#tube = tube.smooth(n_iter = 1000)

time2 = time.time()
print(f"Time to revoxelize was: {time2-time1}")

tube.plot(color = "red")


cone_center = (0.,0.,np.mean(test_sim.coords[:,2]))
cone_direction = (0.,0.,1.)
cone_height = 2.


view_cone = pv.Cone(center = cone_center , direction = cone_direction , height = cone_height , angle = 30.)


cpos = [tuple(2.*cone_height*np.array(cone_direction) + cone_center),
        (0.,0.,0.),
        (0.,0.,0.)
        ]

pl = pv.Plotter()

time1 = time.time()
#set up plotting mesh
pl.add_mesh(tube.clip_surface(view_cone),color = "red")
time2 = time.time()
print(f"time to clip was: {time2-time1}")

#set up camera
pl.savefig()
pl.show(cpos=cpos)

#tube.plot(color = "red")


"""
voxel tester
"""
from pyvista import examples
import pyvista as pv
import numpy as np

# Load a surface to voxelize
surface = examples.download_foot_bones()
surface

cpos = [(7.656346967151718, -9.802071079151158, -11.021236183314311),
 (0.2224512272564101, -0.4594554282112895, 0.5549738359311297),
 (-0.6279216753504941, -0.7513057097368635, 0.20311105371647392)]

surface.plot(cpos=cpos, opacity=0.75)

time1 = time.time()
voxels = pv.voxelize(surface, density=surface.length/200)
time2 = time.time()

p = pv.Plotter()
p.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)
p.add_mesh(surface, color="lightblue", opacity=0.5)
p.show(cpos=cpos)




#testing resurfacing
cloud = pv.PolyData(poly.points)
cloud.plot(point_size=test_sim.radii)

surf = cloud.delaunay_3d(alpha = np.max(test_sim.radii))
surf.plot(show_edges=True)

#test surface 2
time1 = time.time()
voxels = pv.voxelize(view_cone, check_surface = True, density=0.001)
time2 = time.time()
print(f"Time to revoxelize was: {time2-time1}")

p = pv.Plotter()
p.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)
p.add_mesh(view_cone, color="lightblue", opacity=0.5)
p.show()


#test naive voxelization
time1 = time.time()

#get point cloud of cylinders
cyl_center = np.mean( test_sim.coords[list(test_sim.edges[0])] , axis = 0)
cyl_direction = test_sim.coords[test_sim.edges[0][1]] - test_sim.coords[test_sim.edges[0][0]]

cyl_radius = test_sim.radii[0]
cyl_height = np.linalg.norm(cyl_direction)

all_pts = pv.Cylinder(center = cyl_center , direction = cyl_direction , radius = cyl_radius , height = cyl_height , resolution = 20 , capping = False).points

for i,e in enumerate(test_sim.edges[1:]):
    cyl_center = np.mean(test_sim.coords[list(e)] , axis = 0 )
    cyl_direction = test_sim.coords[e[1]] - test_sim.coords[e[0]]

    cyl_radius = test_sim.radii[i]
    cyl_height = np.linalg.norm(cyl_direction)

    all_pts = np.vstack([all_pts,pv.Cylinder(center = cyl_center , direction = cyl_direction , radius = cyl_radius , height = cyl_height , resolution = 20 , capping = False).points])

cloud = pv.PolyData(all_pts)
surf = cloud.delaunay_2d()


time2 = time.time()

print(f"Time to revoxelize was: {time2-time1}")

surf.plot(show_edges=True)




#test the plotting
test_sim.generate_radii(init_r = 0.1)
test_sim.generate_fundus_image(im_type="exact",save=True)


#need to fix
name1 = "fast_marching"
print("With Fast marching:")
for xx in np.linspace(0.001,0.1,6):
    test_sim.generate_radii(init_r = xx)

    time1 = time.time()
    test_sim.voxelize()
    time2 = time.time()

    print(f"Time to revoxelize was: {time2-time1}")

    test_sim.generate_fundus_image(save=True,save_name = name1)


name2 = "generate and redistribute"
print("with Generate and redistribute:")
for xx in np.linspace(0.001,0.1,6):
    test_sim.generate_radii(init_r = xx)

    time1 = time.time()
    test_sim.voxelize()
    time2 = time.time()

    print(f"Time to revoxelize was: {time2-time1}")

    test_sim.generate_fundus_image(save=True,save_name = name2)
