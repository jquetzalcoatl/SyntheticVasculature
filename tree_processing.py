"""
convert to/from tree structure
"""

import numpy as np
import matplotlib.pyplot as plt

def sim_to_tree(pts, branches):

    tree_obj = [[None, pt] for pt in pts]

    for br in branches:
        if len(br) > 0:
            for local_idx in np.arange(len(br))[:0:-1]:
                tree_obj[br[local_idx]][0] = br[local_idx-1]

    return tree_obj


def tree_to_inout_nbrs(tree_obj):

    inout_nbrs = [[None, None, []] for obj in tree_obj]

    for idx in range(len(tree_obj)):
        obj = tree_obj[idx]

        #write the point
        inout_nbrs[idx][1] = obj[1]

        #write the pointer
        inout_nbrs[idx][0] = obj[0]

        #add the in nbr pointer to another node
        if obj[0] != None:
            inout_nbrs[obj[0]][2].append(idx)

    return inout_nbrs

def simplify_tree(inout_nbrs):

    for idx in range(len(inout_nbrs))[::-1]:
        obj = inout_nbrs[idx]
        if (obj[0] != None) & (len(obj[1]) == 1):
            inout_nbrs[obj[2][0]][0] = obj[0]
            inout_nbrs.pop(idx)

    return inout_nbrs

def get_leaves(inout_nbrs):
    leaves = []

    for idx in range(len(inout_nbrs)):
        if len(inout_nbrs[idx][2]) == 0:
            leaves.append(idx)

    return leaves

def get_root(inout_nbrs):
    for idx in range(len(inout_nbrs)):
        if inout_nbrs[idx][0] == None:
            return idx

    return None

def TMD(inout_nbrs,f_vals):

    TMD_bc = []

    active = [0 for pt in inout_nbrs]
    v_vals = np.zeros(len(f_vals))
    leaves = get_leaves(inout_nbrs)
    root_idx = get_root(inout_nbrs)

    for l_idx in leaves:
        v_vals[l_idx] = f_vals[l_idx]
        active[l_idx] = 1

    while active[root_idx] != 1:

        for l in np.nonzero(active)[0]:
            p = inout_nbrs[l][0]
            C = inout_nbrs[p][2]

            if np.min([active[c_idx] for c_idx in C]) == 1:
                c_max = C[np.argmax([v_vals[c_idx] for c_idx in C])]
                active[p] = 1

                for c_idx in C:
                    active[c_idx] = 0

                    if c_idx != c_max:
                        TMD_bc.append([v_vals[c_idx],f_vals[p]])
                v_vals[p] = v_vals[c_max]
    TMD_bc.append([v_vals[root_idx],f_vals[root_idx]])
    return TMD_bc

def bc_density_profile(TMD_bc):

    sorted_TMD_bc = np.array([np.sort(bc) for bc in TMD_bc])

    def func(t, full_bc):
        return sum([(bc[0] <= t) & (t <= bc[1]) for bc in full_bc])

    return lambda t: func(t,sorted_TMD_bc)

def quick_gauss(center = [0.,0.], sigma = 1.):

    return lambda x: (1./(2.*np.pi*sigma**2))*np.exp(-((x[0]-center[0])**2 + (x[1] - center[1])**2)/(2.*sigma**2))

def unweighted_persistent_surface(TMD_bc, sigma = 1.):

    return lambda x: sum([ quick_gauss(center = bc,sigma = sigma)(x) for bc in TMD_bc])

def unweighted_persistent_image(TMD_bc, sigma = 1., bounds = [[0.,2.],[0.,2.]],num_grid_pts_edge = 30):
    pers_surf = unweighted_persistent_surface(TMD_bc, sigma = sigma)
    xx = np.linspace(bounds[0][0],bounds[0][1],num_grid_pts_edge)
    yy = np.linspace(bounds[1][0],bounds[1][1],num_grid_pts_edge)

    im_grid = np.zeros((num_grid_pts_edge,num_grid_pts_edge))

    for i in range(num_grid_pts_edge):
        for j in range(num_grid_pts_edge):
            im_grid[num_grid_pts_edge-j-1,i] = pers_surf([xx[i],yy[j]])

    return im_grid

def init_branch_map():
    return

"""
tree = sim_to_tree(pts,branches)
tree_inout = tree_to_inout_nbrs(tree)
leaves = get_leaves(tree_inout)

f_vals_test = [np.linalg.norm(tree_inout[0][1] - obj[1]) for obj in tree_inout]
tmd_test = TMD(tree_inout, f_vals_test)

bc = np.array(tmd_test)

plt.scatter(bc[:,0],bc[:,1]);plt.show()

test_dens = bc_density_profile(tmd_test)

x = np.linspace(0,3,1000)
#plt.plot(x,[test_dens(x_val) for x_val in x]); plt.show()

test_im = unweighted_persistent_image(tmd_test, sigma = 0.05, bounds = [[0.,np.max(bc)],[0.,np.max(bc)]],num_grid_pts_edge = 40)
plt.imshow(test_im); plt.show()
"""
