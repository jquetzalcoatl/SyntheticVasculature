"""
auxin-based model code
"""
import numpy as np

g_fovea_pos = [0.0, 0.0, -1.0]
g_od_pos = [0.5, 0.0, -0.5*np.sqrt(3)]

def sphere_init_config(fovea_radius = 0.3,lens_depth = 0.3,num_pts = 100,inner_rad = 0.8,outer_rad = 1.2,prune_into_eye = True,bounding_box = None):

    sample = []
    while(len(sample) < num_pts):
        pt = np.random.normal(size = 3)
        pt /= np.linalg.norm(pt)
        pt_rad = np.random.rand()*(outer_rad-inner_rad)+inner_rad
        sample_pt = [pt,pt_rad]

        if bounding_box is None:
            if prune_into_eye:
                if ((pt*pt_rad)[-1] <= 1-lens_depth) \
                        and (np.linalg.norm(pt*pt_rad - np.array(g_fovea_pos)) \
                        >= fovea_radius):
                    sample.append(sample_pt)
        else:
            if prune_into_eye:
                if ((pt*pt_rad)[-1] <= 1-lens_depth) \
                        and (np.linalg.norm(pt*pt_rad - np.array(g_fovea_pos)) \
                        >= fovea_radius) and (isInBox(pt*pt_rad,bounding_box)):
                    sample.append(sample_pt)

    return np.array(sample,dtype=object)

def geodesic_dist(p1,p2):
    p1norm = np.linalg.norm(p1[0])
    p2norm = np.linalg.norm(p2[0])
    p1dotp2 = np.dot(p1[0],p2[0])
    if np.abs(p1dotp2)>1.:
        p1dotp2 = np.sign(p1dotp2)

    return np.arccos(p1dotp2) + np.abs(p1[1] - p2[1])

def tangent_vector(p1,p2,normalized = True):

    p1dotp2 = np.dot(p1[0],p2[0])
    if np.abs(p1dotp2)>1.:
        p1dotp2 = np.sign(p1dotp2)
    p2bar = p2[0] - (p1dotp2)*np.array(p1[0])
    p2bar /= np.linalg.norm(p2bar)

    #print(p1dotp2)
    if normalized:
        return np.array([p2bar,(p2[1]-p1[1])/np.abs(p2[1]-p1[1])],dtype=object)
    else:
        return np.array([(np.arccos(p1dotp2))*p2bar, p2[1]-p1[1]],dtype=object)

def exp_map(pt, direction):
    dirnorm = np.linalg.norm(direction[0])
    #pt_dot_dir = np.dot(pt,dir)
    #dir_bar = dir - pt_dot_dir*np.array(pt)
    #dir_bar /= np.linalg.norm(dir_bar)
    #theta_star = np.arccos(pt_dot_dir)

    return np.array([np.cos(dirnorm)*np.array(pt[0]) + np.sin(dirnorm)*np.array(direction[0])/dirnorm,pt[1]+direction[1] ],dtype=object)

def isInInterval(pt, interval):
    if (pt >= interval[0]) and (pt <= interval[1]):
        return True
    else:
        return False


def isInBox(pt, bbox):
    """
    pt should be of theform [x,y,z],
    bbox should be [[xlow,xhigh],[ylow,yhigh],[zlow,zhigh]]
    """
    if sum([isInInterval(pt[i],bbox[i]) for i in range(len(pt))]) == 3:
        return True
    else:
        return False

def vascular_growth_sim(fovea_radius = 0.2,lens_depth = 0.5,max_iter = 1000,init_num_pts = 200,inner_rad = 0.7,outer_rad = 1.2,D_step = 0.9,death_dist = None,init_vasc = None,bounding_box=None):
    """
    if init_vasc is None, then initialize pt_list and vascular structure
    otherwise, init_vasc = (pt_list )
    """
    if death_dist is None:
        shell_vol = 4.*np.pi*0.5
        approx_cover_rad = 0.1*np.sqrt((shell_vol/init_num_pts)*(3./4.)/np.pi)
        death_dist = approx_cover_rad

    #set up data structure
    if init_vasc is None:
        pt_list = [[g_od_pos, outer_rad]]
        to_grow_indicator = np.array([1])
        branches = [[0]]
        branch_membership = [[0]]
    else:
        pt_list = list(init_vasc[0])
        branches = init_vasc[1]
        branch_membership = init_vasc[2]

        #construct the indicator for whether a point is at the end of a branch
        # by looping through branches
        to_grow_indicator = np.zeros(len(pt_list))
        for b in branches:
            to_grow_indicator[b[-1]] = 1.

    #sample auxin
    sample_auxin = sphere_init_config(fovea_radius = fovea_radius,lens_depth = lens_depth,num_pts = init_num_pts,inner_rad = inner_rad,outer_rad = outer_rad,bounding_box = bounding_box)

    init_sample = np.array(sample_auxin)

    #print("sampled points are: \n");print(sample_auxin)

    #set up auxin-vein node distance chart
    auxin_vein_dists = [geodesic_dist(pt_list[0],s) for s in sample_auxin]
    auxin_min_dists = [[0,d] for d in auxin_vein_dists ]

    active_auxin = np.arange(len(init_sample))

    #print("sampled point dists are: \n");print(auxin_vein_dists)
    #print("sampled point dists are: \n");print(auxin_min_dists)

    count = 0
    #"while there are auxin nodes"
    #while((count < max_iter) and (len(sample_auxin)>0)):
    while((count < max_iter) and (len(active_auxin)>0)):
        count += 1

        #manually find the nearest neighbor
        nns = [[] for pt in pt_list]
        #print("getting nearest neighbors for {} auxin".format(len(sample_auxin)))
        #for i in range(len(sample_auxin)):
        for i in active_auxin:
            #if i in list_deleted_red:
            #   continue
            #match the nearest neighbor of an auxin node to the index of said auxin node
            nns[int(auxin_min_dists[i][0])].append(i)
            #

        #now compute the step vectors
        #print("the to grow indicators are {}".format(to_grow_indicator))
        for i in range(len(pt_list))[::-1]:
            #print("the nearest neighbors for {} are {}".format(i,nns[i]))
            #print("pt {} s nearest neighbors are: {}".format(i,nns[i]))
            if len(nns[i])>0:
                #check if the given point is a head or not
                #if not, generate a new branch
                if to_grow_indicator[i] == 0:
                    branches.append([i])
                    branch_membership[i].append(len(branches)-1)

                #compute the new step size
                step_vec = sum([(1./len(nns[i]))*tangent_vector(pt_list[i],sample_auxin[k],normalized = False) for k in nns[i]])
                vprime = exp_map(pt_list[i], [D_step*step_vec[0],D_step*step_vec[1]])

                #check whether the proposed point is in the bounding box
                #have a boolean defaulted to true, and then possibly turn to false otherwise
                in_box_indicator = True
                if bounding_box is not None:
                    if not isInBox(vprime[1]*vprime[0],bounding_box):
                        in_box_indicator = False

                #if the new point is far enough away from the fovea:
                if (np.linalg.norm(vprime[1]*vprime[0] - np.array(g_fovea_pos))\
                        > fovea_radius) and in_box_indicator:
                    #print("growing from {} to {}".format(pt_list[i],vprime))
                    #add the new point to the list of points
                    pt_list = np.vstack([pt_list,vprime])

                    #change the old grow indicator to 0
                    to_grow_indicator[i] = 0
                    #change the new grow indicator to 1
                    to_grow_indicator = np.append(to_grow_indicator,1)
                    #add branch information for this new branch
                    branch_membership.append([branch_membership[i][-1]])
                    branches[branch_membership[i][-1]].append(len(to_grow_indicator)-1)

                    #update distance array
                    #dists = np.array([geodesic_dist(vprime,s) for s in sample_auxin])
                    dists = np.array([geodesic_dist(vprime,sample_auxin[j]) for j in active_auxin])

                    #print("distances to auxin for vprime are: {}".format(dists))
                    #set up auxin-vein node distance chart
                    #auxin_vein_dists = np.vstack([auxin_vein_dists,dists])

                    #update min distances
                    #for j in range(len(sample_auxin))[::-1]:
                    temp_active_len = len(active_auxin)
                    for idx, j in enumerate(active_auxin):
                        if dists[idx] <= auxin_min_dists[j][1]:
                            #update the min distance array
                            #sample_auxin = np.delete(sample_auxin,j,0)
                            #print(f"idx: {idx}"); print(f"j: {j}")
                            #active_auxin = np.delete(active_auxin,temp_active_len-idx-1,0)

                            auxin_min_dists[j][1] = dists[idx]
                            auxin_min_dists[j][0] = len(to_grow_indicator)-1

        #prune auxin nodes
        #alternative: updated list_deleted_red
        #for j in range(len(sample_auxin))[::-1]:
        #for j in active_auxin[::-1]:

            #first check whether or not the new point got close enough to an auxin node
            #print(dists)
            #if auxin_min_dists[j][1] < death_dist:

        temp_active_len = len(active_auxin)
        for j in np.arange(temp_active_len)[::-1]:

            #first check whether or not the new point got close enough to an auxin node
            if auxin_min_dists[active_auxin[j]][1] < death_dist:
                #delete auxin
                #sample_auxin = np.delete(sample_auxin,j,0)
                #active_auxin = np.delete(active_auxin,j,0)
                active_auxin = np.delete(active_auxin,j)

                #auxin_vein_dists = np.delete(auxin_vein_dists,j,1)
                #auxin_min_dists = np.delete(auxin_min_dists,j,0)

                #print("to grow indicator is: \n"); print(to_grow_indicator)
                #print("new point dists are: \n");print(auxin_vein_dists)
                #print("new point dists are: \n");print(auxin_min_dists)

    #while there are auxin nodes left or max_counts has been exceeded
    #print(f"active_auxin: {len(active_auxin)}"); print(f"count: {count}")
    return np.array(pt_list), branches, branch_membership, init_sample

def convert_from_product(pt_list):
    new_pts = []
    for pt in pt_list:
        new_pts.append(pt[1]*np.array(pt[0]))
    return np.array(new_pts)

def restrict_branches(pts,branches,branch_membership,max_height = -0.1):

    pt_birth_times = np.zeros(len(pts))
    pt_birth_times[0] = 1.

    for br in branches:
        for i in range(1,len(br)):
            if pts[br[i]][-1] > max_height:
                pt_birth_times[br[i]] = np.inf
            else:
                pt_birth_times[br[i]] = pt_birth_times[br[i-1]] + 1

    #prune for points with birth times < infinity
    new_branches = [[] for br in branches]
    new_branch_membership = [[] for pt in pts]

    for i in range(len(new_branches)):
        for br_pt in branches[i]:
            if pt_birth_times[br_pt] < np.inf:
                new_branches[i].append(br_pt)
                new_branch_membership[br_pt].append(i)
            else:
                break


    return new_branches, new_branch_membership

#new_branches, new_branch_membership = restrict_branches(pts,pt_idx,branches,branch_membership)

def extract_graph(num_pts,branches):
    #construct network
    all_edges = []

    for br in branches:
        for i in range(len(br)-1):
            all_edges.append((br[i],br[i+1]))
    all_edges = list(set(all_edges))

    A = np.zeros((num_pts,num_pts))

    for e in all_edges:
        A[e[0],e[1]] = 1
        A[e[1],e[0]] = 1

    #directed neighbors point from leafs to root
    directed_neighbors = {i:[] for i in range(num_pts)}

    for e in all_edges:
        directed_neighbors[e[1]].append(e[0])

    return np.array(all_edges), A,directed_neighbors

def get_vein_radii(directed_nbrs, A,init_radii = 0.05,branch_power = 3.):

    num_pts = len(directed_nbrs)
    vein_radii = np.zeros(num_pts)

    #initialize leaves with init_radii
    degrees = np.array([sum(r) for r in A])
    vein_radii[degrees == 1] = init_radii
    #make sure root does not have init_radii
    vein_radii[0] = 0.

    for i in range(num_pts)[::-1]:
        for j in directed_nbrs[i]:
            vein_radii[j] = (vein_radii[j]**branch_power + vein_radii[i]**branch_power)**(1./branch_power)

    return vein_radii

def project_points(pts, A,projection_lim = 0.2):

    projected_idx = np.arange(len(pts))[pts[:,2]<= projection_lim]
    projected_pts = pts[projected_idx]

    projected_A = A[np.ix_(projected_idx,projected_idx)]

    projected_edges = []

    for i in range(len(projected_A)-1):
        for j in range(i+1, len(projected_A)):
            if projected_A[i,j] >0:
                projected_edges.append((i,j))

    projected_edges = np.array(projected_edges)

    return projected_pts, projected_edges,projected_idx
