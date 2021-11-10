"""
This script takes retina images and masks and digitally removes the vasculature
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


def remove_vasc(img, vessels, mask, output_path):
    img,vessels,mask = cv2.imread(img), cv2.imread(vessels, 0), \
            cv2.imread(mask, 0)
    vessels_inv = 255-vessels
    blank_original = cv2.bitwise_and(img, img, mask=vessels_inv)
    ksz=3
    kernel = np.ones((ksz,ksz), dtype="uint8")
    blank = cv2.dilate(blank_original, kernel, iterations=6)
#    import pdb; pdb.set_trace()
    vessels = cv2.cvtColor(vessels, cv2.COLOR_GRAY2RGB)
    vessels_inv = cv2.cvtColor(vessels_inv, cv2.COLOR_GRAY2RGB)
    vessels_inv = cv2.bitwise_and(vessels_inv, vessels_inv, mask=mask)
    mean_bgnd = np.mean( blank[vessels==255] ) 
    mean_fgnd = np.mean( blank_original[vessels_inv==255] )
    fgnd_ratio_scale = 0.3
        # TODO Instead of calculating mean_bgnd from entire retina, calculate
        # it just from pixels adjacent to vessels
    ratio = 1.0 - ( fgnd_ratio_scale*(1.0 - mean_fgnd/mean_bgnd) )
#    import pdb; pdb.set_trace()
    blank = (vessels/255)*ratio*blank + (vessels_inv/255)*blank_original
    cv2.imwrite(output_path, blank)

def main(cfg):
    input_supdir = os.path.abspath( cfg["input_supdir"] )
    images_dir = pj(input_supdir, "images")
    vessel_dir = pj(input_supdir, "1st_manual_png")
    mask_dir = pj(input_supdir, "mask_png")

    # Note, if the program fails it may be necessary to convert the images in
    # these directories to .png files first (as above)
    if not pe(vessel_dir):
        vessel_dir = pj(input_supdir, "1st_manual")
    if not pe(mask_dir):
        mask_dir = pj(input_supdir, "mask")

    output_dir = pj(input_supdir, "blanks")
    if not pe(output_dir):
        os.makedirs(output_dir)
    images = sorted( os.listdir(images_dir) )
    masks = sorted( os.listdir(mask_dir) )
    vessels = sorted( os.listdir(vessel_dir) )
    for img_fn,mask_fn,vessel_fn in zip(images, masks, vessels):
        img = pj(images_dir, img_fn)
        mask = pj(mask_dir, mask_fn)
        vessel = pj(vessel_dir, vessel_fn)
        output_path = pj(output_dir, img_fn[:2] + "_blank.png")
        remove_vasc(img, vessel, mask, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-supdir", type=str,
            default=pj(HOME, "Datasets/Retina/DRIVE/training"))
    cfg = vars( parser.parse_args() )
    main(cfg)
