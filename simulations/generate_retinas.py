"""
Generate retinas using simulated vasculature.  Parts copied fairly directly 
from DtN_test.py
"""

import argparse
import numpy as np
import os
import pandas as pd
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulations.model_runner as MR

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def create_pos3d(pos2d):
    """
    :pos2d: (wd, ht) in (0,1) image coordinates, with origin in upper left
    """
    x,y = pos2d
    x,y = 2.0*(x-0.5), 2.0*(y-0.5) 
    z = np.sqrt( 1 - (x*x + y*y) )
    pos3d = (x, y, -z) # Project to lower hemisphere
    return pos3d

def main(cfg):
    output_dir = os.path.abspath( cfg["output_dir"] )
    synet = MR.Vasculature("auxin", 500, output_dir+os.sep)

    fovea_pos2d = (0.0, 0.5)
    od_pos2d = (0.75, 0.4)

    fovea_pos = create_pos3d(fovea_pos2d)
    od_pos = create_pos3d(od_pos2d)
    synet.set_geometry(od=od_pos, fovea=fovea_pos)
    time1 = time.time()
    synet.run_simulation(step_size = 0.5, fovea_radius = 0.3)
    time2 = time.time()
    print("Simulation took {:.2f}s".format(time2-time1))
    synet.generate_radii(0.1)

    save_name = "sim_retina"
    synet.generate_fundus_image(im_type="exact", save=True,
            save_name=save_name)
    synet.save_radii(save_name=save_name)
    synet.save_simulation(save_name=save_name)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Output/retina"))
    cfg = vars( parser.parse_args() )
    main(cfg)


