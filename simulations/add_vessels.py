"""
This script adds pre-generated vessels to a pre-generated retina blank.
Basically a demo.
"""

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def main(cfg):
    blank = cv2.imread( cfg["blank"] )
    blank2 = blank.copy()
    ht,wd,ch = blank.shape

    coords = np.load( open(pj(cfg["vessels_dir"], "sim_retina_coords.npy"),
        "rb") )
    edges = np.load( open(pj(cfg["vessels_dir"], "sim_retina_edges.npy"),
        "rb") )
    radii = np.load( open(pj(cfg["vessels_dir"],
        "sim_retina_radii-vein_radius-0.100.npy"), "rb") )
    for e_i,edge in enumerate(edges):
        p0 = coords[ edge[0] ]
        p1 = coords[ edge[1] ]
#        import pdb; pdb.set_trace()
        p0 = (1+p0) * ht / 2
        p1 = (1+p1) * ht / 2
        p0 = [int(x) for x in p0]
        p1 = [int(x) for x in p1]
        e_i = min( e_i, len(radii)-1 )
#        thickness = int( max(1, 10 * radii[e_i]) )
        thickness = int( max(1, 10 * (radii[edge[0]] + radii[edge[1]]) / 2) )
#        thickness = int( max(1, 10 * radii[edge[0]] ) )
        cv2.line(blank2,  p0[:2], p1[:2], color=(127,32,96),
                thickness=thickness)
#        if e_i == 100:
#            break

    alpha = 0.75
    blank = cv2.addWeighted(blank2, alpha, blank, 1-alpha, gamma=0)
    cv2.imwrite( pj(cfg["vessels_dir"], "syn_ret.png"), blank )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blank", type=str,
            default=pj(HOME,
                "Datasets/Retina/DRIVE/training/blanks/31_blank.png"))
    parser.add_argument("-v", "--vessels-dir", type=str,
            default=pj(HOME, "Output/retina/plots"))
    cfg = vars( parser.parse_args() )
    main(cfg)

