#DtN_test

#import tools

import numpy as np
import DtN_tools as DtN

import matplotlib.pyplot as plt


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
degrees = np.array([sum(adj[i]) for i in range(len(adj))])
leaves = np.arange(len(adj))[degrees == 1]

#identify source node
degreesD = np.array([sum(adjD[i]) for i in range(len(adjD))])
sourceNode = np.argsort(degreesD)[-4]

#all boundary nodes
bdyNodes = np.hstack([leaves,sourceNode])

#set up DtN with source and all leaves
L = np.diag([sum(r) for r in adjD]) - adjD
steklov_spec = DtN.steklov_spec(L,bdyNodes)

#plt.plot(steklov_spec[0]); plt.show()

#visualize network
edgeIdx = np.nonzero(adj)
edge_list = [(edgeIdx[0][i],edgeIdx[1][i]) for i in range(len(edgeIdx[0]))]

for e in edge_list:
    plt.plot(coords[e,0],coords[e,1],c="k")

for l in leaves:
    plt.scatter(coords[l,0],coords[l,1],c="b")

plt.scatter(coords[sourceNode,0],coords[sourceNode,1],c="r")

plt.show()

#get eigenvector, compute harmonic extension
schurComp = DtN.schurComp()

#compute flow from source to leaves
