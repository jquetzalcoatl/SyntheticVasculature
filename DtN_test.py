#DtN_test

#import tools

import numpy as np
import DtN_tools as DtN

import matplotlib.pyplot as plt
from matplotlib import cm


#read in test network
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

#identify leaves
#degrees = np.array([sum(adj[i]) for i in range(len(adj))])
#leaves = np.arange(len(adj))[degrees == 1]

#read in leaf nodes
leaves = []
with open(dir+file_name+"_endpoints.csv","r") as f:
    for l in f:
        leaves.append(int(l) )
leaves = np.array(leaves)

#identify source node
degreesD = np.array([sum(adjD[i]) for i in range(len(adjD))])
sourceNode = np.argsort(degreesD)[-4]

#all boundary nodes
bdyNodes = np.hstack([leaves,sourceNode])

#set up DtN with source and all leaves
L = np.diag([sum(r) for r in adjD]) - adjD
steklov_spec = DtN.steklov_spec(L,bdyNodes)
schur = DtN.schur_comp(L,bdyNodes)

#plt.plot(steklov_spec[0]); plt.show()

#visualize network
edgeIdx = np.nonzero(adj)
edge_list = [(edgeIdx[0][i],edgeIdx[1][i]) for i in range(len(edgeIdx[0]))]

for e in edge_list:
    plt.plot(coords[e,1],coords[e,0],c="k")

for l in leaves:
    plt.scatter(coords[l,1],coords[l,0],c="b")

plt.scatter(coords[sourceNode,1],coords[sourceNode,0],c="r")

plt.show()



#plot the first 9 eigvecs + harmonic extensions

fig, ax = plt.subplots(ncols = 3, nrows = 3)

offset = 8

for i in range(3):
    for j in range(3):
        #get eigenvector, compute harmonic extension
        u = steklov_spec[1][i*3 + j+offset]


        Hu = DtN.harmonic_extension(schur[1],u)

        full_soln = np.zeros(len(L))
        full_soln[bdyNodes] = u
        full_soln[list(set(np.arange(len(L))) - set(bdyNodes))] = Hu

        for e in edge_list:
            ax[i,j].plot(coords[e,1],coords[e,0],c="k")

        ax[i,j].scatter(coords[:,1],coords[:,0],c=full_soln,cmap = cm.jet)
        ax[i,j].set_title("l = {:.2f}".format(np.real(steklov_spec[0][i*3 + j + offset])))
plt.show()

#compute flow from source to leaves
