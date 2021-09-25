"""

"""

import time
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir("/Users/wham/Dropbox/_code/python/vessel_segmentation")

import vessel_sim_commands as vsm
import tree_processing as tp

import sklearn.decomposition as decomp

data_objects = []
data_cols = []

counter = 0


num_samples = 5
num_pts_list = [100,200,300]
data_cols_list = ["r","g","b"]

m_fig, m_ax = plt.subplots(nrows= 3,ncols = 3)

for i in range(len(num_pts_list)):
    for k in range(num_samples):
        num_pts = num_pts_list[i]

        #run the simulation
        print("starting sim {}/{}".format(counter,300))
        tic = time.time()

        max_iter = 1000
        num_iter = 1

        growth_type_para = 1
        if growth_type_para:
            growth_type = "average"
        else:
            growth_type = "nearest"
        fovea_radius = 0.2;lens_depth = 0.3

        D_step_para = 0.3
        shell_vol = 4.*np.pi*0.5
        approx_cover_rad = 0.1*np.sqrt((shell_vol/num_pts)*(3./4.)/np.pi)

        weighted_para = False

        tic = time.time()
        result = vsm.vascular_growth_sim(num_iterations = num_iter,noisy = False,fovea_radius = fovea_radius,lens_depth = lens_depth,max_iter = max_iter,init_num_pts = num_pts,inner_rad = 0.7,outer_rad = 1.2, growth_type = growth_type,weighted_stepsizes = weighted_para, D_step = D_step_para,death_dist = approx_cover_rad)
        toc = time.time()
        print("time to complete sim with {} pts, growth type {}, and {} iters was: {:.2f}".format(num_pts,growth_type,max_iter,toc-tic))
        print("step size was {:.2f}".format(D_step_para))

        #save the data, draw the picture
        pts = vsm.convert_from_product(result[0])
        init_sample = vsm.convert_from_product(result[-1])/1.2
        branches = result[1]
        branch_membership = result[2]

        new_branches, new_branch_membership = vsm.restrict_branches(pts,branches,branch_membership)

        vein_radii = vsm.get_vein_radii(len(pts),new_branches,init_radii = 0.2,branch_power = 3)

        #save the first three sims
        if k <= 2:
            #draw the image
            for br in new_branches:
                #isolate the branch pieces below the xy axes
                if len(br)>0:
                    m_ax[i,k].plot(pts[br,0],pts[br,1],c="k",linewidth = np.mean(vein_radii[br]))

            #rescale everything
            m_ax[i,k].set_xlim([-1.0,1.0])
            m_ax[i,k].set_ylim([-1.0,1.0])
            #take away boundary buffers?
            m_ax[i,k].axis('off')

            c_circle = [0.6/1.2,0.0]; r_circle = 0.15

            plot_pts = np.array([[r_circle*np.cos(t)+c_circle[0],r_circle*np.sin(t)+c_circle[1]] for t in np.linspace(-np.pi,np.pi,100)])
            m_ax[i,k].plot(plot_pts[:,0],plot_pts[:,1])

        #run the TMD filtration
        tree = tp.sim_to_tree(pts,branches)
        tree_inout = tp.tree_to_inout_nbrs(tree)
        leaves = tp.get_leaves(tree_inout)

        #getting the radial distance to the root
        f_vals_test = [np.linalg.norm(tree_inout[0][1] - obj[1]) for obj in tree_inout]
        tmd_test = tp.TMD(tree_inout, f_vals_test)
        bc = np.array(tmd_test)

        test_im = tp.unweighted_persistent_image(tmd_test, sigma = 0.05, bounds = [[0.,np.max(bc)],[0.,np.max(bc)]],num_grid_pts_edge = 40)

        data_objects.append(test_im)
        data_cols.append(data_cols_list[i])

        #update the counter for our own keeping track purposes
        counter +=1

#show the sample images
plt.show()

flattened_data = [np.ravel(obj) for obj in data_objects]
flattened_data_mean = sum([np.ravel(obj) for obj in data_objects])
flattened_data = [obj - flattened_data_mean for obj in flattened_data]

#run PCA on the persistent surfaces
pca_embed = decomp.PCA(n_components = 2)
pca2 = pca_embed.fit_transform(np.array(flattened_data))

plt.scatter(pca2[:,0],pca2[:,1],c = data_cols); plt.show()

#visualize some of the heat maps
m_fig, m_ax = plt.subplots(nrows= 3,ncols = 3)

for i in range(len(num_pts_list)):
    for k in range(3):
        m_ax[i,k].imshow(data_objects[i*num_samples + k])
plt.show()
