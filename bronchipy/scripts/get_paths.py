#!/usr/bin/env python3

import csv
import logging
import numpy as np
from scipy.spatial import distance
from bronchipy.airwaytree import AirwayTree

logging.basicConfig(level=logging.ERROR)

branch_file = ""
inner_file = ""
inner_rad_file = ""
outer_file = ""
outer_rad_file = ""
volume_nii = "./bronchipy/assets/fixed_lumen_segmentation.nii.gz"

airway_tree = AirwayTree(branch_file=branch_file,
                         inner_file=inner_file,
                         inner_radius_file=inner_rad_file,
                         outer_file=outer_file,
                         outer_radius_file=outer_rad_file,
                         volume=volume_nii)

path_length = 0
path_tree = None

for path_id in airway_tree.paths:
    length = len(airway_tree.paths[path_id])
    if length > path_length:
        path_length = length
        print(f"Longest path {path_id} with length {length}")
        try:
            path_tree = airway_tree.tree.loc[airway_tree.paths[path_id]]
        except KeyError as e:
            f"No branch for this path {path_id}\n{e}"

centreline = path_tree.centreline.explode().to_list()
points = path_tree.points.explode().to_list()
lumen_radii = path_tree.inner_radii.explode().to_list()

carr = np.array(centreline)
xx = np.zeros(carr.shape[0])
for i in range(1, len(xx)):
    xx[i] = (distance.euclidean(carr[i - 1], carr[i]) + xx[i - 1]
             )  # Convert from relative distance to absolute distance.
with open("branch_centreline.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['r', 'a', 's'])
    writer.writerows(points)

with open("lumen_radii.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(xx, lumen_radii))
