#DtN_test

#import tools

#generic packages
import numpy as np
import os
import time

#special packages
import DtN_tools as DtN
from simulations import model_runner as mr

#plotting packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

import pyvista as pv


np.random.seed(10202021)

generate_network = True
simulation_run = False

#read in test network
if generate_network:

    if not simulation_run:
        synthetic_network = mr.Vasculature("auxin", 500, os.getcwd())
        time1 = time.time()
        synthetic_network.run_simulation(step_size = 0.5,fovea_radius = 0.3)
        time2 = time.time()
        print("Simulation took {:.2f}s".format(time2-time1))
        synthetic_network.generate_radii(0.001)
        simulation_run = True

    if len(synthetic_network.radii) == len(synthetic_network.coords):
        #adapt vertex-based radii to edge-based radii
        edge_radii = np.array([np.mean(synthetic_network.radii[list(e)]) for e in synthetic_network.edges])
    else:
        edge_radii = synthetic_network.radii

    adj = np.copy(synthetic_network.A)
    adjD = np.copy(synthetic_network.A)
    coords = synthetic_network.coords

    threshold = 0.0
    # and (sum([coords[cc,2] <= threshold for cc in synthetic_network.all_nbrs[idx] ])==0)
    #project_idx = [idx for idx, c in enumerate(coords) if c[2] <= threshold]

    #fast march out from root, stopping when
    checked = [0 for i in range(len(coords))]
    checked[0] = 1.
    next_idx = 0
    next_up = list(np.copy(synthetic_network.all_nbrs[0]))

    project_idx = [ii for ii in synthetic_network.all_nbrs[0]]
    project_idx.append(0)

    while len(next_up)>0:
        ii = next_up.pop()
        for potential_idx in synthetic_network.all_nbrs[ii]:
            if (coords[potential_idx,2] <= threshold) and (checked[potential_idx] == 0):
                checked[potential_idx] = 1
                next_up.append(potential_idx)
                project_idx.append(potential_idx)
            else:
                checked[potential_idx] = 2
            checked[ii] = 2


    for idx,e in enumerate(synthetic_network.edges):
        adjD[e[0],e[1]] = edge_radii[idx]
        adjD[e[1],e[0]] = edge_radii[idx]

    p_adj = adj[np.ix_(project_idx,project_idx)]
    p_adjD = adjD[np.ix_(project_idx,project_idx)]
    p_coords = coords[project_idx]

    #identify leaves
    degrees = np.array([sum(adj[i]) for i in range(len(adj))])
    leaves = np.arange(len(adj))[degrees == 1]

    p_degrees = np.array([sum(p_adj[i]) for i in range(len(p_adj))])
    p_leaves = np.arange(len(p_adj))[p_degrees == 1]

else:
    file_name = "01_manual1"
    dir = "./Data/networks/"

    adj = []
    with open(dir+file_name+"_adj.csv","r") as f:
        for l in f:
            adj.append([int(a) for a in l.split("\t")])
    adj = np.array(adj)

    adjD = []
    with open(dir+file_name+"_adjD.csv","r") as f:
        for l in f:
            adjD.append([float(a) for a in l.split("\t")])
    adjD = np.array(adjD)

    coords = []
    with open(dir+file_name+"_nodes.csv","r") as f:
        for l in f:
            coords.append([float(a) for a in l.split("\t")])
    coords = np.array(coords)

    #read in leaf nodes
    leaves = []
    with open(dir+file_name+"_endpoints.csv","r") as f:
        for l in f:
            leaves.append(int(l) )
    leaves = np.array(leaves)



#identify leaves
#degrees = np.array([sum(adj[i]) for i in range(len(adj))])
#leaves = np.arange(len(adj))[degrees == 1]



#identify source node
p_degreesD = np.array([sum(adjD[i]) for i in range(len(adjD))])
sourceNode = project_idx.index(0)
#sourceNode = np.argsort(degreesD)[-4]

#all boundary nodes
bdyNodes = np.hstack([p_leaves,sourceNode])

#set up DtN with source and all leaves
L = np.diag([sum(r) for r in p_adjD]) - p_adjD
steklov_spec = DtN.steklov_spec(L,bdyNodes)
schur = DtN.schur_comp(L,bdyNodes)

#plt.plot(steklov_spec[0]); plt.show()

#visualize network
edgeIdx = np.nonzero(p_adj)
edge_list = [(edgeIdx[0][i],edgeIdx[1][i]) for i in range(len(edgeIdx[0]))]

for idx,e in enumerate(edge_list):
    plt.plot(p_coords[e,1],p_coords[e,0],c="k")

#plt.scatter(p_coords[sourceNode,0],p_coords[sourceNode,1],c="r")

plt.show()



poly = pv.PolyData()
poly.points = synthetic_network.coords
cells = np.full((len(synthetic_network.edges), 3), 2, dtype=np.int_)
cells[:, 1] =  synthetic_network.edges[:,0]
cells[:, 2] = synthetic_network.edges[:,1]
poly.lines = cells

#poly["scalars"] = np.arange(poly.n_points)
poly["radii"] = synthetic_network.radii
tube = poly.tube(radius = np.min(synthetic_network.radii),scalars = "radii",capping = False)
#tube = tube.smooth(n_iter = 1000)

#tube.plot(color = "red")

sargs = dict(interactive=True)  # Simply make the bar interactive
p = pv.Plotter(notebook=False)
pv.set_plot_theme("document")
p.add_mesh(tube, scalar_bar_args=sargs)
p.add_bounding_box()
p.show()



#
# solve DtN (harmonic extension)
#

#normal flow

source_sink = -np.ones(len(bdyNodes))#/(len(bdyNodes) - 1.)

source_sink[-1] = 1
Hu = DtN.harmonic_extension(schur[1],source_sink)

full_soln = np.zeros(len(L))
full_soln[bdyNodes] = source_sink
full_soln[list(set(np.arange(len(L))) - set(bdyNodes))] = Hu

fig, ax = plt.subplots()

min_width = np.min(synthetic_network.edge_radii)

for e in edge_list:
    plot_col = (np.mean([full_soln[e[1]],full_soln[e[0]]]) - np.min(full_soln))/(np.max(full_soln)-np.min(full_soln))
    ax.plot(p_coords[e,0],p_coords[e,1],c = cm.jet(plot_col))

#scatter the bad point

ax.scatter(p_coords[p_leaves,0],p_coords[p_leaves,1],c="b",s=30,alpha = 0.2)
ax.set_facecolor(cm.gist_heat(0.3))

plt.axis('off')
plt.show()


#modified flow

source_sink = -np.ones(len(bdyNodes))#/(len(bdyNodes) - 1.)

affected = []
for ii in range(len(bdyNodes)):
    if np.linalg.norm(p_coords[bdyNodes[ii]] - [0,-0.8,0]) <= 0.5:
        source_sink[ii] = 0.
        affected.append(bdyNodes[ii])
        print("got one!")

source_sink[-1] = 1
Hu = DtN.harmonic_extension(schur[1],source_sink)

full_soln = np.zeros(len(L))
full_soln[bdyNodes] = source_sink
full_soln[list(set(np.arange(len(L))) - set(bdyNodes))] = Hu

fig, ax = plt.subplots()

min_width = np.min(synthetic_network.edge_radii)

for e in edge_list:
    plot_col = (np.mean([full_soln[e[1]],full_soln[e[0]]]) - np.min(full_soln))/(np.max(full_soln)-np.min(full_soln))
    ax.plot(p_coords[e,0],p_coords[e,1],c = cm.jet(plot_col)) #,linewidth=p_adjD[e]/min_width/2.

#scatter the bad point
ax.scatter(0.,-0.8,marker="x",c="k",s = 100)

ax.scatter(p_coords[p_leaves,0],p_coords[p_leaves,1],c="b",s=30,alpha = 0.2)
ax.scatter(p_coords[affected,0],p_coords[affected,1],c="b",s=30)
ax.set_facecolor(cm.gist_heat(0.3))

plt.axis('off')
plt.show()

#plot the first 9 eigvecs + harmonic extensions

fig, ax = plt.subplots(ncols = 3, nrows = 3)

offset = 0

for i in range(3):
    for j in range(3):
        #get eigenvector, compute harmonic extension
        u = steklov_spec[1][i*3 + j+offset]


        Hu = DtN.harmonic_extension(schur[1],u)

        full_soln = np.zeros(len(L))
        full_soln[bdyNodes] = u
        full_soln[list(set(np.arange(len(L))) - set(bdyNodes))] = Hu

        for e in edge_list:
            plot_col = (np.mean([full_soln[e[1]],full_soln[e[0]]]) - np.min(full_soln))/(np.max(full_soln)-np.min(full_soln))
            ax[i,j].plot(p_coords[e,0],p_coords[e,1],c = cm.jet(plot_col), ms = p_adjD[e])

        ax[i,j].scatter(p_coords[p_leaves,0],p_coords[p_leaves,1],c="b",s = 1.5*np.max(synthetic_network.radii))
        ax[i,j].set_title("l = {:.2f}".format(np.real(steklov_spec[0][i*3 + j + offset])))
plt.show()

#compute flow from source to leaves
