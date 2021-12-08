#set up meshes for 2 and 3d

import numpy as np
from scipy import spatial as spsp

import write_off

import itertools as iter

def generate_sample_2D_square():
	"""
	Sample points in a spherical shell, generate a triangulation, and
	        then remove points (and simplices) that are above the cut height,
	        within the center part, or near the fovea
	"""
	points = []

	grid1 = np.linspace(0,27,num=28)
	grid2 = np.linspace(0.5,26.5,num=27)

	#add the image lattice points
	for ii in grid1:
		for jj in grid1:
			points.append([ii,jj])

	#add the half lattice points
	for ii in grid2:
		for jj in grid2:
			points.append([ii,jj])

	#triangulate
	#triangulate
	triangulation = spsp.Delaunay(points)
	triSimps = list(triangulation.simplices)

	return np.array([[p[0],p[1],0.] for p in points]),triSimps


def generate_sample_3D_shell():
	"""
	Sample points in a spherical shell, generate a triangulation, and
	        then remove points (and simplices) that are above the cut height,
	        within the center part, or near the fovea
	"""
	points = []

	grid1 = np.linspace(0,27,num=28)
	grid2 = np.linspace(0.5,26.5,num=27)

	#add the image lattice points
	for ii in grid1:
		for jj in grid1:
			points.append([ii,jj])

	#add the half lattice points
	for ii in grid2:
		for jj in grid2:
			points.append([ii,jj])

	#triangulate
	#triangulate
	triangulation = spsp.Delaunay(points)
	triSimps = list(triangulation.simplices)

	return np.array([[p[0],p[1],0.] for p in points]),triSimps

def sampled_triangulation_to_OFF(triSimps):
	"""
	given the top faces from a Delaunay triangulation, return the faces
	"""

	Triangles = []
	Edges = []

	for s in Simplices:
		for ss in iter.combinations(s,3):
			Triangles.append(ss)
		for ss in iter.combinations(s,2):
			Edges.append(ss)

	#remove the duplicates
	Triangles = np.array(list(set(Triangles)))
	Edges = np.array(list(set(Edges)))

	return Triangles, Edges

Vertices, Simplices = generate_sample()

Triangles, Edges = sampled_triangulation_to_OFF(Simplices)

write_off.write_off("/Users/wham/Documents/MNIST_OT/meshes/", "MNIST_triangulation", Vertices, Triangles)
