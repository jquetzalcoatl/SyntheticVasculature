"""
navigate to the directory, run directly in the command line:

python quick_test.py

to run the sim and viz output.
"""

import model_runner as mr

import pyvista as pv

import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

#run the simulation

for step in [0.9]:

    test_sim = mr.Vasculature("auxin", 400, "/Users/wham/Documents/vessel_sims/")

    time1 = time.time()
    test_sim.run_simulation(step_size = step)
    time2 = time.time()
    print(f"Time to run sim was: {time2-time1}")

    test_sim.generate_radii(0.001)

    if len(test_sim.radii) == len(test_sim.coords):
        #adapt vertex-based radii to edge-based radii
        edge_radii = np.array([np.mean(test_sim.radii[list(e)]) for e in test_sim.edges])
    else:
        edge_radii = test_sim.radii

    #test surface 2
    time1 = time.time()

    poly = pv.PolyData()
    poly.points = test_sim.coords

    cells = np.full((len(test_sim.edges), 3), 2, dtype=np.int_)

    cells[:, 1] =  test_sim.edges[:,0]
    cells[:, 2] = test_sim.edges[:,1]
    poly.lines = cells

    #poly["scalars"] = np.arange(poly.n_points)
    poly["radii"] = test_sim.radii

    tube = poly.tube(radius = np.min(test_sim.radii),scalars = "radii",capping = False)
    #tube = tube.smooth(n_iter = 1000)

    time2 = time.time()
    print(f"Time to revoxelize was: {time2-time1}")

    tube.plot(color = "red")


"""
plotting
"""
"""
test_sim.generate_radii(0.001)

if len(test_sim.radii) == len(test_sim.coords):
    #adapt vertex-based radii to edge-based radii
    edge_radii = np.array([np.mean(test_sim.radii[list(e)]) for e in test_sim.edges])
else:
    edge_radii = test_sim.radii

#test surface 2
time1 = time.time()

poly = pv.PolyData()
poly.points = test_sim.coords

cells = np.full((len(test_sim.edges), 3), 2, dtype=np.int_)

cells[:, 1] =  test_sim.edges[:,0]
cells[:, 2] = test_sim.edges[:,1]
poly.lines = cells

#poly["scalars"] = np.arange(poly.n_points)
poly["radii"] = test_sim.radii

tube = poly.tube(radius = np.min(test_sim.radii),scalars = "radii",capping = False)
#tube = tube.smooth(n_iter = 1000)

time2 = time.time()
print(f"Time to revoxelize was: {time2-time1}")

tube.plot(color = "red")


cone_center = (0.,0.,np.mean(test_sim.coords[:,2]))
cone_direction = (0.,0.,1.)
cone_height = 2.


view_cone = pv.Cone(center = cone_center , direction = cone_direction , height = cone_height , angle = 30.)


cpos = [tuple(2.*cone_height*np.array(cone_direction) + cone_center),
        (0.,0.,0.),
        (0.,0.,0.)
        ]

pl = pv.Plotter()

time1 = time.time()
#set up plotting mesh
pl.add_mesh(tube.clip_surface(view_cone),color = "red")
time2 = time.time()
print(f"time to clip was: {time2-time1}")

#set up camera
#pl.savefig()
pl.show(cpos=cpos)
"""
