"""
generates an example with some sort of "branched" solution
"""

import os

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
from numpy import linalg as lin

from math import *

# Import the useful routines
import read_off
import surface_pre_computations
import geodesic_surface_congested
import cut_off

# To plot graph
import matplotlib.pyplot as plt

# To plot triangulated surfaces
from mayavi import mlab



#mlab.points3d(Vertices[:,0],Vertices[:,1],Vertices[:,2],color = (1.,1.,1.),mode = "point",opacity = 0.5)
#mlab.show()

##########################
# running the OT stuff
#########################



# Name of the file in which is stored the triangulated surface D
nameFileD = os.path.join("meshes", "square_regular_100.off")

# Extract Vertices, Triangles, Edges
Vertices, Triangles, Edges = read_off.readOff(nameFileD)

# Discretization of the starting [0,1] (for the centered grid)
nTime = 31
# Parameter epsilon to regularize the Laplace problem
eps = 0.0*10**(-8)
# Number of iterations
Nit = 30
# Detailed Study: True if we compute the objective functional at every time step (slow), False in the case we compute every 10 iterations.
detailStudy = False
# Value for the congestion parameter (alpha in the article)
cCongestion = 1.0

# -----------------------------------------------------------------------------------------------
# Read the .off file
# -----------------------------------------------------------------------------------------------

# Compute areas of Triangles
areaTriangles, angleTriangles, baseFunction = surface_pre_computations.geometricQuantities(Vertices, Triangles, Edges)

# Compute the areas of the Vertices
originTriangles, areaVertices, vertexTriangles = surface_pre_computations.trianglesToVertices( Vertices,Triangles, areaTriangles)

# -----------------------------------------------------------------------------------------------
# Define the boundary conditions
# -----------------------------------------------------------------------------------------------

nVertices = Vertices.shape[0]

mub0 = np.zeros(nVertices)
mub1 = np.zeros(nVertices)

mub0[-1] = areaVertices[-1]

np.random.seed(100)
for i in np.random.choice(np.arange(nVertices//2),size = 2,replace = False) :
	#mub1[i] = areaVertices[i] * cut_off.f( Vertices[i,2] + 0.1 , 0.3 )
	mub1[i] = areaVertices[i]

# Normalization
mub0 /= np.dot(mub0,areaVertices)
mub1 /= np.dot(mub1,areaVertices)

mlab.triangular_mesh(Vertices[:,0], Vertices[:,1], Vertices[:,2], Triangles, scalars=mub1, colormap = "bone", transparent = True)
#mlab.triangular_mesh(Vertices[:,0], Vertices[:,1], mub1, Triangles, scalars=np.arange(nVertices), colormap = "bone", transparent = True)
mlab.show()


# -----------------------------------------------------------------------------------------------
# Call the algorithm
# -----------------------------------------------------------------------------------------------

mus = []
phis = []
toPlots = []

for cCongestion in [1e-8]:
	phi,mu,A,E,B,objectiveValue,primalResidual,dualResidual = geodesic_surface_congested.geodesic( nTime, nameFileD, mub0,mub1, cCongestion,eps, Nit, detailStudy )

	mus.append(mu)
	phis.append(phi)
	#branch_vals = np.linspace(0,1,num = mu.shape[0])[::-1]
	total_mu = np.zeros(mu.shape[1])

	for i in range(mu.shape[0]):
		total_mu += mu[i]/np.dot(mu[i],areaVertices)

	#muNormalized = total_mu/np.dot(total_mu,areaVertices)

	# The normalization of the color map is done independently for each instant
	#toPlot = (np.max(muNormalized) - muNormalized) / np.max(muNormalized)
	toPlots.append(total_mu)


mlab.triangular_mesh(Vertices[:,0], Vertices[:,1], Vertices[:,2], Triangles, scalars=np.sum(mus[0],axis=0), colormap = "jet", transparent = True)
mlab.show()

"""
for t in np.arange(0,31,5):
	mlab.triangular_mesh(Vertices[:,0], Vertices[:,1], Vertices[:,2], Triangles, scalars=mu[t]/(mu[t]@areaVertices), colormap = "jet", transparent = True)
	mlab.show()
"""

muTotal = np.zeros(len(mu[0]))

for i in range(nTime):
	muTotal[mu[i]>1e-8] += (mu[i]/(mu[i]@areaVertices))[mu[i]>1e-8]
muTotal /= nTime

muTrace = (muTotal>1e-3).astype("float")

mlab.triangular_mesh(Vertices[:,0], Vertices[:,1], Vertices[:,2], Triangles, scalars=muTrace, colormap = "jet", transparent = True)
mlab.show()
