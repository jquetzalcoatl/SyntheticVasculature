"""
read samples with indeces in specified range
"""

import argparse
import numpy as np
import os
import pandas as pd
import sys
import time

read_range = [0,3]

dir_name = "plots/"

coords = []
edges = []
radii = []

verbose = 0

for ii in range(*read_range):

        index_str = str(ii)
        file_name = "0"*(6-len(index_str)) + index_str

        coords.append(np.load("plots/"+file_name + "_coords.npy"))
        edges.append(np.load("plots/"+file_name + "_edges.npy"))
        radii.append(np.load("plots/"+file_name + "_radii.npy"))

print(len(coords))
print(len(edges))
print(len(radii))
