"""
generate plenty of synthetic vasculature
saves coords of vertices, edges, and radii as np files
adapted from simulations/generate_retinas.py
"""

import argparse
import numpy as np
import os
import pandas as pd
import sys
import time

import simulations.model_runner as MR

init_sample_range = [50,100]
num_samples = 5

output_dir_name = ""

verbose = 0

for ii in range(num_samples):

        num_samples = np.random.randint(*init_sample_range)

        synet = MR.Vasculature("auxin", num_samples, output_dir_name)

        fovea_pos = [-0.5, 0.0, -0.5*np.sqrt(3)]
        od_pos = [0.0, 0.5, -0.5*np.sqrt(3)]
        synet.set_geometry(od=od_pos, fovea=fovea_pos)

        time1 = time.time()
        synet.run_simulation(step_size = 0.5, fovea_radius = 0.3)
        time2 = time.time()

        if verbose > 0:
                print("Simulation took {:.2f}s".format(time2-time1))
        synet.generate_radii(0.1)

        index_str = str(ii)
        save_name = "0"*(6-len(index_str)) + index_str
        #synet.generate_fundus_image(im_type="exact", save=True, save_name=save_name)
        synet.save_radii(save_name=save_name)
        synet.save_simulation(save_name=save_name)
