"""
3d vascular growth sim
just the commands
"""

import io

import numpy as np
from scipy import spatial as spspat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import integrate as spint

import time

def sphere_init_config(fovea_radius = 0.3,lens_depth = 0.3,num_pts = 100,inner_rad = 0.8,outer_rad = 1.2,prune_into_eye = True):
    """
    sample = np.random.normal(size = (num_pts,3))
    random_radii = np.random.rand(num_pts)*(outer_rad-inner_rad)+inner_rad
    sample = [[sample[i]/np.linalg.norm(sample[i]),random_radii[i]] for i in range(len(sample))]

    if prune_into_eye:
        #remove portions near iris
        for i in range(len(sample)-1,-1,-1):
          #print(i)
          if (sample[i][0][-1] > 1-lens_depth) or (np.linalg.norm(sample[i][0] - np.array([0.,0.,-1.])) < fovea_radius):
            sample.pop(i)
    """
    sample = []
    while(len(sample) < num_pts):
        pt = np.random.normal(size = 3)
        pt /= np.linalg.norm(pt)
        pt_rad = np.random.rand()*(outer_rad-inner_rad)+inner_rad
        sample_pt = [pt,pt_rad]

        if prune_into_eye:
            if ((pt*pt_rad)[-1] <= 1-lens_depth) and (np.linalg.norm(pt*pt_rad - np.array([0.,0.,-1.])) >= fovea_radius):
                sample.append(sample_pt)

    return np.array(sample)

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
        return np.array([p2bar,(p2[1]-p1[1])/np.abs(p2[1]-p1[1])])
    else:
        return np.array([(np.arccos(p1dotp2))*p2bar, p2[1]-p1[1]])

def exp_map(pt, direction):
    dirnorm = np.linalg.norm(direction[0])
    #pt_dot_dir = np.dot(pt,dir)
    #dir_bar = dir - pt_dot_dir*np.array(pt)
    #dir_bar /= np.linalg.norm(dir_bar)
    #theta_star = np.arccos(pt_dot_dir)

    return np.array([np.cos(dirnorm)*np.array(pt[0]) + np.sin(dirnorm)*np.array(direction[0])/dirnorm,pt[1]+direction[1] ])

#exp_map([0.,0.,1.2],tangent_vector([0.,0.,1.2],[0.,1,0.]))

"""
p1 = [[0.,0.,1.],1.1]
p2 = [[0.0,1.1,0.],0.9]
print(geodesic_dist(p1,p2))
print(tangent_vector(p1,p2))
"""


"""
X = sphere_init_config(num_pts = 1000)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(X[:,0],X[:,1],X[:,2])
plt.show()
"""

def prune_dist_chart(dist_chart,min_dist_pointers,death_dist = 0.1):

    return

def vascular_growth_sim(num_iterations = 3,fovea_radius = 0.3,lens_depth = 0.5,noisy = True,max_iter = 10,init_num_pts = 1000,inner_rad = 0.7,outer_rad = 1.2, growth_type = "average",weighted_stepsizes = True,D_step = 0.05,death_dist = 0.05,save_time_data = False):

    #set up data structure
    pt_list = [[[0.5,0.,-0.5*np.sqrt(3)],outer_rad]]
    to_grow_indicator = np.array([1])
    branches = [[0]]
    branch_membership = [[0]]

    if save_time_data:
        time_data = [[pt_list,list(branches),list(branch_membership)]]

    #start the iteration
    for iter_count in range(num_iterations):
        #sample auxin
        if iter_count == 0:
            sample_auxin = sphere_init_config(fovea_radius = fovea_radius,lens_depth = lens_depth,num_pts = init_num_pts,inner_rad = inner_rad,outer_rad = outer_rad)
            init_sample = np.array(sample_auxin)
        else:
            sample_auxin = sphere_init_config(fovea_radius = fovea_radius,lens_depth = lens_depth,num_pts = 2**iter_count*init_num_pts,inner_rad = inner_rad,outer_rad = outer_rad)
            D_step = D_step/(2**iter_count);death_dist = death_dist/(2**iter_count)
            init_sample = np.vstack([init_sample,sample_auxin])

        #print("sampled points are: \n");print(sample_auxin)

        #set up auxin-vein node distance chart
        if iter_count == 0:
            auxin_vein_dists = [geodesic_dist(pt_list[0],s) for s in sample_auxin]
            auxin_min_dists = [[0,d] for d in auxin_vein_dists ]
        else:
            auxin_vein_dists = np.array([[geodesic_dist(pt,s) for s in sample_auxin] for pt in pt_list])
            auxin_min_dists = []
            for s_idx in range(len(sample_auxin)):
                argmin_idx = np.argmin(auxin_vein_dists[:,s_idx])
                auxin_min_dists.append([argmin_idx,auxin_vein_dists[argmin_idx,s_idx]])
            auxin_min_dists = np.array(auxin_min_dists)

        #print("sampled point dists are: \n");print(auxin_vein_dists)
        #print("sampled point dists are: \n");print(auxin_min_dists)

        count = 0
        #"while there are auxin nodes"
        while((count < max_iter) and (len(sample_auxin)>0)):
            if noisy:
                print("at step {}".format(count))
            count += 1

            #manually find the nearest neighbor
            nns = [[] for pt in pt_list]
            #print("getting nearest neighbors for {} auxin".format(len(sample_auxin)))
            for i in range(len(sample_auxin)):
                #match the nearest neighbor of an auxin node to the index of said auxin node
                nns[int(auxin_min_dists[i][0])].append(i)

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

                    #get the step vector for the grown point
                    #geometry_type = "average" means
                    if growth_type == "average":

                        if weighted_stepsizes:
                            step_vec = sum([(1./len(nns[i]))*tangent_vector(pt_list[i],sample_auxin[k],normalized = True) for k in nns[i]])
                            vprime = exp_map(pt_list[i], [D_step*step_vec[0],D_step*step_vec[1]])
                        else:
                            step_vec = sum([(1./len(nns[i]))*tangent_vector(pt_list[i],sample_auxin[k],normalized = False) for k in nns[i]])
                            vprime = exp_map(pt_list[i], [D_step*step_vec[0],D_step*step_vec[1]])

                    elif growth_type == "nearest":
                        #print(auxin_vein_dists)
                        #print(auxin_vein_dists[i])
                        if len(pt_list) == 1:
                            nearest_auxin = 0
                        else:
                            #print(auxin_vein_dists.shape)
                            #print(np.array(auxin_min_dists).shape)
                            #print(auxin_min_dists)
                            #print(nns[i])
                            #print(len(sample_auxin))
                            nearest_auxin = np.argmin([auxin_vein_dists[i][k] for k in nns[i]])

                        #now construct the step vector
                        if weighted_stepsizes:
                            step_vec = tangent_vector(pt_list[i],sample_auxin[nns[i][nearest_auxin]],normalized = True)
                            vprime = exp_map(pt_list[i],[D_step*step_vec[0],D_step*step_vec[1]])
                        else:
                            step_vec = tangent_vector(pt_list[i],sample_auxin[nns[i][nearest_auxin]],normalized = False)
                            vprime = exp_map(pt_list[i], [D_step*step_vec[0],D_step*step_vec[1]])

                    #if the new point is far enough away from the fovea:
                    if np.linalg.norm(vprime[1]*vprime[0] - np.array([0.,0.,-1.])) > fovea_radius:
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
                        dists = np.array([geodesic_dist(vprime,s) for s in sample_auxin])

                        #print("distances to auxin for vprime are: {}".format(dists))
                        #set up auxin-vein node distance chart
                        auxin_vein_dists = np.vstack([auxin_vein_dists,dists])

                        #update min distances
                        for j in range(len(sample_auxin))[::-1]:
                            if dists[j] < auxin_min_dists[j][1]:
                                #update the min distance array
                                #sample_auxin = np.delete(sample_auxin,j,0)
                                auxin_min_dists[j][1] = dists[j]
                                auxin_min_dists[j][0] = len(to_grow_indicator)-1

            #prune auxin nodes
            for j in range(len(sample_auxin))[::-1]:

                #first check whether or not the new point got close enough to an auxin node
                #print(dists)
                if auxin_min_dists[j][1] < death_dist:
                    #delete auxin
                    sample_auxin = np.delete(sample_auxin,j,0)
                    auxin_vein_dists = np.delete(auxin_vein_dists,j,1)
                    auxin_min_dists = np.delete(auxin_min_dists,j,0)
                    #print("to grow indicator is: \n"); print(to_grow_indicator)
                    #print("new point dists are: \n");print(auxin_vein_dists)
                    #print("new point dists are: \n");print(auxin_min_dists)

            if save_time_data:
                time_data.append([pt_list,list(branches),list(branch_membership)])

    #while there are auxin nodes left or max_counts has been exceeded
    if save_time_data:
        return np.array(pt_list), branches, branch_membership, init_sample,time_data
    else:
        return np.array(pt_list), branches, branch_membership, init_sample

def convert_from_product(pt_list):
    new_pts = []
    for pt in pt_list:
        new_pts.append(pt[1]*np.array(pt[0]))
    return np.array(new_pts)

def get_vein_radii(num_pts, branches,init_radii = 0.05,branch_power = 3.):
    vein_radii = np.zeros(num_pts)

    for br in branches[::-1]:
        if len(br) > 0:
            vein_radii[br[-1]] = init_radii
            for br_idx in range(2,len(br)+1):
                vein_radii[br[-br_idx]] = np.power(vein_radii[br[-br_idx]]**(branch_power) + vein_radii[br[-br_idx+1]]**(branch_power),1./branch_power)

    return vein_radii

def sim_to_image(pts, branches,vein_radii,dpi = 500,figsize = (6,6),draw_circle = False,c_circle = [0.0,0.0],r_circle = 1.):

    fig, ax = plt.subplots(1,1,figsize = figsize,dpi = dpi)

    for br in branches:
        #isolate the branch pieces below the xy axes
        if len(br)>0:
            local_br = np.array(br)[pts[br,2]<0.05]
            ax.plot(pts[local_br,0],pts[local_br,1],c="k",linewidth = np.mean(vein_radii[local_br]))

    #rescale everything
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    #take away boundary buffers?
    ax.axis('off')

    if draw_circle:
        plot_pts = np.array([[r_circle*np.cos(t)+c_circle[0],r_circle*np.sin(t)+c_circle[1]] for t in np.linspace(-np.pi,np.pi,100)])
        ax.plot(plot_pts[:,0],plot_pts[:,1])

    return fig, ax

#from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def fig_to_img(fig, ax):
    fig.add_axes(ax)
    fig.canvas.draw()

    # this rasterized the figure
    X = np.array(fig.canvas.renderer._renderer)
    X = 0.2989*X[:,:,1] + 0.5870*X[:,:,2] + 0.1140*X[:,:,3]

    plt.close("all")

    return X

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


def write_sim_data(pts,branches,branch_membership,file_path,file_name):
    f = open("{}{}_points.dat".format(file_path,file_name),"w")
    for pt in pts:
        for coord in pt:
            f.write("{:.5f},".format(coord))
        f.write("\n")
    f.close()

    f = open("{}{}_branches.dat".format(file_path,file_name),"w")
    for br in branches:
        for b in br:
            f.write("{},".format(b))
        f.write("\n")
    f.close()

    f = open("{}{}_branch_membership.dat".format(file_path,file_name),"w")
    for br in branch_membership:
        for b in br:
            f.write("{},".format(coord))
        f.write("\n")
    f.close()
    return

def heat_signature(pts, branches,t=1.,num_eigs = 3,save_plot = True):
    #initial condition/constants come from integrating bessel functions along branches

    #get plot points
    r_vals = 0.5*(np.cos(np.linspace(0.,np.pi,20))+1.)
    theta_vals = np.linspace(0.,2.*np.pi,100)

    #sort eig_vals, get corresponding eig_fns
    eig_vals = np.array([ spspec.jn_zeros(eigfn_idx,10) for eigfn_idx in range(num_eigs)])
    eig_val_pairs =  np.array([ (spspec.jn_zeros(eigfn_idx,10),eigfn_idx) for eigfn_idx in range(num_eigs)])
    eig_val_sort_order = np.argsort(eig_vals.ravel())
    eig_val_pairs_sorted = eig_val_pairs.ravel()[eig_val_sort_order]



    R,THETA = np.meshgrid(r_vals,theta_vals)
    X = R*np.cos(THETA)
    Y = R*np.sin(THETA)

    heat_kernel_consts = []

    for i in range(num_eigs):
        e_val, e_idx = eig_val_pairs_sorted[i]
        kth_eigfn1 = lambda x: spspec.jv(e_idx,e_val*np.linalg.norm(x))*np.cos(e_idx*np.angle(x[0]+1.j*x[1]))
        kth_eigfn1_polar = lambda r,theta: spspec.jv(e_idx,e_val*r)*np.cos(e_idx*theta)
        #kth_eigfn2 = lambda x: spspec.jv(e_idx,e_val*np.linalg.norm(x))*np.sin(e_idx*np.angle(x[0]+1.j*x[1]))

        total_integral = 0.
        for br in branches:
            total_integral += sum([spint.quad(lambda t: kth_eigfn1(pts[br[ii]]*(1.-t) + pts[br[ii+1]]*t),0,1) for ii in range(len(br)-1)])
        heat_kernel_consts.append(total_integral)

    heat_kernel = lambda r,theta: sum([heat_kernel_consts[eig_idx]*np.exp(-eig_val_pairs_sorted[eig_idx][0]*t)*spspec.jv(eig_val_pairs_sorted[eig_idx][1],eig_val_pairs_sorted[eig_idx][0]*r)*np.cos(eig_val_pairs_sorted[eig_idx][1]*theta) for eig_idx in range(num_eigs)])

    Z = [[heat_kernel(r,theta) for r in r_vals] for theta in theta_vals]
    Z = np.array(Z)

    if save_plot:

        level_bound = np.max([np.abs(np.min(Z)),np.max(Z)])
        levels = np.linspace(-level_bound,level_bound,50)

        norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
        cmap = cm.bwr

        CS = axes[i,j].contourf(X, Y, Z,levels,alpha = 0.9,norm=norm,cmap = cm.get_cmap(cmap,len(levels)-1))
        fig.colorbar(CS,ax=axes[i,j])
        plt.show()

    return (X,Y), Z
