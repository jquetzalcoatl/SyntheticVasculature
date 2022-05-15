"""
Extract useful segments from actual retinas
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
    img_name = cfg["input_image"]
    stub = os.path.splitext(img_name)[0]
    cat = stub[ stub.index("_")+1 : ]
    if cat == "h":
        img_subdir = "healthy"
    elif cat == "g":
        img_subdir = "glaucoma"
    elif cat == "dr":
        img_subdir = "diabetic_retinopathy"
    else:
        raise NotImplementedError(cat)
    img = cv2.imread( pj( cfg["data_supdir"], img_subdir, img_name ) )
    label_stub = stub+"_AVmanual"
    label_path_stub = pj( cfg["analysis_supdir"], "sket", label_stub )
    if not pe(label_path_stub+".png"):
        from PIL import Image
        label_img = Image.open(label_path_stub+".gif")
        label_img.write(label_path_stub+".png")
    label = cv2.imread( label_path_stub+".png", cv2.IMREAD_GRAYSCALE )
    import pdb; pdb.set_trace()

    nodes_csv = stub + "_AVmanual_nodes.csv"
    nodes = pd.read_csv( pj(cfg["analysis_supdir"], "networks", nodes_csv),
            sep="\t" )
    for i,row in nodes.iterrows():
        cv2.circle(img, (int(row[1]), int(row[0])), 10, (0,255,0))
        cv2.circle(label, (int(row[1]), int(row[0])), 10, (0,255,0))
    nodes = nodes.to_numpy().astype(int)

    adj_csv = stub + "_AVmanual_adj.csv"
    adj_mat = pd.read_csv( pj(cfg["analysis_supdir"], "networks", adj_csv),
            sep="\t" )
    adj_mat = adj_mat.to_numpy()
        # As read/written, it's quasi-symmetric; (i,j) in X implies (j-1,i+1)
        # is in X
    edges = []
    for j in range( adj_mat.shape[1] ):
        ends = [i for i in np.where( adj_mat[:,j] )[0] if i>=j ]
        for i in ends:
            p0 = (nodes[i, 1], nodes[i, 0])
            p1 = (nodes[j-1, 1], nodes[j-1, 0])
        edges.append( (p0, p1) ) 

    for edge in edges:
        cv2.line(img, edge[0], edge[1], (0,255,0), 1)
        cv2.line(label, edge[0], edge[1], (0,255,0), 1)

    output_path = pj(HOME, "Output/retina/plots", img_name)
    cv2.imwrite(output_path, img)
    print( f"Wrote image to {output_path}" )
    label_output_path = pj(HOME, "Output/retina/plots", label_stub+".png")
    cv2.imwrite(label_output_path, label)
    print( f"Wrote label to {label_output_path}" )
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/Retina/HRF_DB"))
    parser.add_argument("-i", "--input-image", type=str, default="01_h.jpg")
    parser.add_argument("-a", "--analysis-supdir", type=str,
            default=pj(HOME, "Repos/jquetzalcoatl/SyntheticVasculature/Data"))
    cfg = vars( parser.parse_args() )
    main(cfg)

