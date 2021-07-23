"""
package to voxelize an embedded graph
"""
import numpy as np

def Bresenham3D(p1, p2):
    #from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints


def generate_voxels(coords,voxel_size,edges,radii,fast_marching = True):
    """
    generates the voxelization for an embedded tree, with radii associated to vertices/edges

    :param coords: np.array; the coordinates of vertices in the tree
    :param voxel_size: int; the sidelength of the voxel cube the tree is embedded in
    :param edges: np.array; the list of edges in the tree
    :param radii: np.array; the list of radii associated to the tree, either with vertices or edges

    :return voxel_centers: np.array; the list of voxel centers in the voxelization
    """

    if len(radii) == len(coords):
        #adapt vertex-based radii to edge-based radii
        radii = np.array([np.mean(radii[list(e)]) for e in edges])

    #get bounding box
    bbox_bounds = np.array([(np.min(coords[:,i]),np.max(coords[:,i])) for i in range(coords.shape[1]) ])
    #print("bbox bounds: "); print(bbox_bounds)

    #convert 3d coords to voxel centers
    data_scaling = voxel_size/np.max([bb[1]-bb[0] for bb in bbox_bounds])
    #print("data scaling:"); print(data_scaling)

    new_pts = ((coords - bbox_bounds[:,0])*data_scaling).astype(int)



    if fast_marching:

        voxel_centers = []

        front = []
        #front_dict keeps track of whether a voxel has been checked, what its nearest edge neighbor is, and the distance to that edge neighbor
        front_dict = {}

        #start by getting the bresenham points
        for e_idx,e in enumerate(edges):
            front_to_add = Bresenham3D(new_pts[e[0]],new_pts[e[1]])

            #print(f"front to add: {front_to_add}")
            for pt in front_to_add:
                try:
                    if front_dict[pt]["front"] == 0:
                        pass
                except:

                    front_dict[pt] = {"front" : 0, "nearest_edge" : e_idx, "dist_to_edge" : 0.}
                    front.append(pt)

        #now propogate the front
        while(len(front) > 0):
            #pop a member of the front. If it's close to it's nearest edge, add it to voxelization and consider neighbors
            temp_pt = front.pop(0)

            #check whether the point has been checked yet
            if front_dict[temp_pt]["front"] == 0:
                #if it hasn't, get the edge info
                nearest_edge_idx = front_dict[temp_pt]["nearest_edge"]
                nearest_edge = edges[nearest_edge_idx]

                #check whether the proposed voxel is close enough to the edges
                if front_dict[temp_pt]["dist_to_edge"] <= radii[nearest_edge_idx]*data_scaling:
                    #point is close enough to an edge, so add it to the voxels
                    voxel_centers.append(temp_pt)


                    for nn in voxel_nn(temp_pt,voxel_size):
                        #check each nn, whether they've been seen and/or the current edge is closer
                        try:
                            #try checking whether the next point is closer to this edge or another
                            new_dist = dist_to_line(nn,new_pts[nearest_edge[0]],new_pts[nearest_edge[1]])
                            if front_dict[nn]["dist_to_edge"] > new_dist:
                                #if the last voxels edge is closer than what was written, rewrite
                                front_dict[nn]["dist_to_edge"] = new_dist
                                front_dict[nn]["nearest_edge"] = nearest_edge_idx

                        except:
                            #nn hasn't been seen yet, so add it to the front and initialize an entry in the dict
                            dist_to_edge = dist_to_line(nn,new_pts[nearest_edge[0]],new_pts[nearest_edge[1]])
                            front_dict[nn] = {"front" : 0, "nearest_edge" : nearest_edge_idx, "dist_to_edge" : dist_to_edge}
                            front.append(nn)

                #regardless, the point is no longer in the front
                front_dict[pt]["front"] = 1

        #once the front has propogated through, collect the remaining voxels

    else:
        #for each edge in the network, draw the line with width along edge
        voxel_centers = set()

        #for each edge
        for ii,e in enumerate(edges):
            ##compute the correct Rotation + translation
            pt1 = coords[e[0]]
            pt2 = coords[e[1]]

            R = edge_affine_transform(pt1,pt2)
            p = new_pts[e[0]]

            ##draw the corresponding cylinder
            cyl_voxels = generate_voxel_cylinder((np.linalg.norm(pt2-pt1)*data_scaling).astype(int),(radii[ii]*data_scaling).astype(int),affine_transformation = [R,p],return_set = True)

            voxel_centers = voxel_centers.union(cyl_voxels)

    return np.array(list(voxel_centers))

def dist_to_line(pt, line_pt1, line_pt2):
    """
    returns the distance of a point pt to the line spanned by line_pt1 and line_pt2

    :param pt: np.array; the point in question
    :param line_pt1: np.array; one endpoint of the line segment in question
    :param line_pt2: np.array; another endpoint of the line segment in question

    :return dist: the distance from the point to the line
    """

    if tuple(line_pt1) == tuple(line_pt2):
        return np.linalg.norm(np.array(pt) - line_pt1)
    else:
        #print(f"pt: {pt}, line_pt1: {line_pt1}, line_pt2: {line_pt2}")
        s1 = np.array(line_pt2).astype(float) - line_pt1
        s1 /= np.linalg.norm(s1)

        dist = np.linalg.norm((pt - s1) - np.dot(pt-s1,s1)*s1)

        return dist

def voxel_nn(pt,voxel_size):
    """
    compute the adjacent voxels to a given voxel

    :param pt: tuple; the center voxel
    :param voxel_size: int; the size of the voxel cube (side length)

    :return nn: list; list of nearest neighbor voxel neighbors in the voxel grid
    """

    nn = []

    perturb_vals = [-1.,0.,1.]
    perturb_pts = np.array([(ii,jj,kk) for ii in perturb_vals for jj in perturb_vals for kk in perturb_vals])


    for pp in perturb_pts:
        proposed = pp + pt
        if (np.min(proposed) >=0.) and (np.max(proposed) <= voxel_size):
            nn.append(tuple(proposed))

    return nn

def generate_voxel_cylinder(cyl_length,cyl_radius,affine_transformation = None,return_set = True):
    """
    generates a standard voxelized cylinder as an alternative to Bresenham's algorithm with thickness

    :param cyl_length: int; the length of the cylinder, in voxels along the x-axis
    :param cyl_radius: int; the radius of the cylinder, in voxels in the y-z plane
    :param affine_trnasformation: [R,p]; list of rotation array R and translation vector p
    :return: a list of voxel centers
    """
    if affine_transformation is None:
        cyl_voxels = [(float(ii),0.,0.) for ii in range(cyl_length)]
        temp_idx = cyl_radius

    #start at the top of the circle and work down

        while temp_idx > 0:
            #extend the voxels in the x direction
            for jj in range(int(np.sqrt(cyl_radius**2 - temp_idx**2))+1):
                #print(f"jj is {jj}")
                for ii in range(cyl_length):
                    cyl_voxels.append((float(ii), float(jj), float(temp_idx)))
                    cyl_voxels.append((float(ii), float(temp_idx),-float(jj)))
                    cyl_voxels.append((float(ii), -float(jj), -float(temp_idx)))
                    cyl_voxels.append((float(ii), -float(temp_idx),float(jj)))
            temp_idx -= 1
    else:
        R, p = affine_transformation

        cyl_voxels = [Rp(R,p,(float(ii),0.,0.)) for ii in range(cyl_length)]
        temp_idx = cyl_radius

    #start at the top of the circle and work down

        while temp_idx > 0:
            #extend the voxels in the x direction
            for jj in range(int(np.sqrt(cyl_radius**2 - temp_idx**2))+1):
                #print(f"jj is {jj}")
                for ii in range(cyl_length):
                    cyl_voxels.append(Rp(R,p, (float(ii), float(jj), float(temp_idx)) ))
                    cyl_voxels.append(Rp(R,p, (float(ii), float(temp_idx),-float(jj)) ))
                    cyl_voxels.append(Rp(R,p, (float(ii), -float(jj), -float(temp_idx)) ))
                    cyl_voxels.append(Rp(R,p, (float(ii), -float(temp_idx),float(jj)) ))
            temp_idx -= 1

    if return_set:
        return set(cyl_voxels)
    else:
        return list(set(cyl_voxels))

def edge_affine_transform(pt1, pt2):
    """
    given initial point pt1 and terminal point pt2, compute the affine transformation from (1.,0.,0.) to pt2 - pt1

    :param pt1: np.array; initial point
    :param pt2: np.array; terminal point

    :return R, p: [np.array, tuple]; linear transformation and translation to move and orient edge from origin, (1.,0.,0.)
    """

    s1 = np.array(pt2) - pt1
    s1 /= np.linalg.norm(s1)

    #compute orthogonal plane
    if np.abs(s1[0]) > 1e-6:
        s2 = np.array([(-s1[1]-s1[2])/s1[0],1.,1.])
    elif np.abs(s1[1]) > 1e-6:
        s2 = np.array([1.,(-s1[0]-s1[2])/s1[1],1.])
    else:
        s2 = np.array([1.,1.,(-s1[0]-s1[1])/s1[2]])
    s2 /= np.linalg.norm(s2)

    s3 = np.array([s1[1]*s2[2] - s1[2]*s2[1],-s1[0]*s2[2] + s1[2]*s2[0],s1[0]*s2[1] - s1[1]*s2[0]])
    s3 /= np.linalg.norm(s3)

    return np.vstack([s1,s2,s3]).T

def Rp(R,p,pt):
    """
    explicitly computes the affine transformation R * pt + p

    :param R: np.array; linear transformation
    :param p: np.array; translation vector
    :param pt: tuple; point to be transformed by

    :return new_pt: tuple; transformed point
    """

    print(R)
    print(p)

    new_pt = tuple([ int(sum([R[i][j]*pt[j] for j in range(len(pt))]) + p[i]) for i in range(len(R))])

    return new_pt

def sample_SOn(n):
    """
    sample a matrix in SO(n) by taking three vectors on the sphere and orthogonalizing

    :param n: int; dimension of ambient space
    :return A: np.array; resulting matrix in SO(n)
    """
    samples = [np.random.normal(size = n) for i in range(n)]
    samples = [s/np.linalg.norm(s) for s in samples]

    s1 = samples[0]
    s2 = samples[1] - np.dot(samples[1],s1)*s1
    s2 /= np.linalg.norm(s2)

    s3 = samples[2] - np.dot(samples[2],s1)*s1 - np.dot(samples[2],s2)*s2
    s3 /= np.linalg.norm(s3)

    return np.vstack([s1,s2,s3]).T

def run_voxelization_test(cyl_length = 3, cyl_radius = 3,affine_transformation = True):

    from mayavi import mlab
    import time

    if affine_transformation == True:
        affine_transformation = [sample_SOn(3),np.random.randint(low = 20, high = 50)*np.random.rand(3)]
    else:
        affine_transformation = None
    tic = time.time()
    voxel_centers = np.array(generate_voxel_cylinder(cyl_length,cyl_radius, affine_transformation = affine_transformation,return_set = False) )
    toc = time.time()
    print(f"time to compute was: {toc-tic:.3f}")
    #for p in test:
    #    print(p)

    mlab.points3d(voxel_centers[:,0],voxel_centers[:,1],voxel_centers[:,2], mode = "cube", scale_factor = 1.,opacity = 0.2,color= (0.9,0.,0.))

    if affine_transformation is not None:
        voxel_centers = np.array(generate_voxel_cylinder(cyl_length,cyl_radius, affine_transformation = None,return_set = False) )
        mlab.points3d(voxel_centers[:,0],voxel_centers[:,1],voxel_centers[:,2], mode = "cube", scale_factor = 1.,opacity = 0.2,color= (0.,0.,0.9))
    mlab.show()

    return

#run_voxelization_test(cyl_length = 100)

def run_large_voxelization_test(edges = [[0,1],[0,2],[1,3],[1,4]], coords = np.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,1.,1.],[1.,1.,0.]]),radii = 0.5*np.array([0.2,0.1,0.05,0.025])):

    from mayavi import mlab
    import time

    tic = time.time()
    voxel_centers = generate_voxels(coords, 100,edges,radii)
    toc = time.time()
    print(f"time to compute was: {toc-tic:.3f}")
    #for p in test:
    #    print(p)

    mlab.points3d(voxel_centers[:,0],voxel_centers[:,1],voxel_centers[:,2], mode = "cube", scale_factor = 1.,opacity = 0.2,color= (0.9,0.,0.))

    mlab.show()

    return

#run_large_voxelization_test()
