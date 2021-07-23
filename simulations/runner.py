"""
main runner for all sims

Args:
    - type of sim
    - num points in sim
    - save name
    - other parameters?
"""

#main packages
import argparse
import numpy as np
import os

#special packages
import model_runner as mr

#Read in supplied arguments.
parser = argparse.ArgumentParser(description="Run a single vasculature simulation.")
parser.add_argument("-t", "--sim_type", required=True, default = "auxin", type = str, help="Type of simulation: auxin, branched_OT")
parser.add_argument("-i", "--inf_num_pts", default=100, required=False, type = int, help="Min number of initial points to generate simulation.")
parser.add_argument("-s", "--sup_num_pts", default=200, required=False, type = int, help="Max number of initial points to generate simulation.")
parser.add_argument("-n", "--num_simulations", required=True, default = 100, type = int, help="Number of simulations to run.")
parser.add_argument("-d", "--directory", default=None, required=False, type=str, help="Directory of where to save simulation.")
parser.add_argument("-o","--other", default = None,required=False,type=str,help="Path to file with optional parameters.")

args = parser.parse_args()
t = args.sim_type
num_min = args.inf_num_pts
num_max = args.sup_num_pts
num_runs = args.num_simulations
dir = args.directory

#run the requested number of simulations
for ii in range(num_runs):
    print()

    model = mr.Vasculature()

    model.save_simulation(dir)
