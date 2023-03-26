#!/usr/bin/env python3

from pathlib import Path
import multiprocessing as mp
import argparse
import csv
import re

import numpy as np
from tqdm import tqdm
from ants import image_read

from bronchipy.util.imageoperations import affine_register
from bronchipy.calc.summary_stats import fractal_dimension


def calc_afd(seg_path):
    fixed = image_read("./bronchipy/assets/fixed_lumen_segmentation.nii.gz")
    try:
        id = re.search(r'\d{6}', str(seg_path)).group()
        moving = image_read(str(seg_path))
        air_tree = affine_register(fixed, moving)
        n, r = fractal_dimension(air_tree)
        afd_arr = -np.diff(np.log(n)) / np.diff(np.log(r))
        afd = np.mean(afd_arr[2:-2])
        result = f"{id},{afd}"
        with open(str(seg_path.parent / f"afd/{id}_afd.txt"), "w") as f:
            f.write(result)
        return result
    except:
        print(f"Error processing {str(seg_path.stem)}")


def main(args):
    in_path = Path(args.in_dir)
    in_list = [f for f in in_path.iterdir() if f.is_file()]
    in_list.sort()
    print("Loaded list. Starting AFD calculation")

    with mp.Pool(10) as pool:
        results = list(tqdm(pool.imap_unordered(calc_afd, in_list), total=len(in_list)))

    header = ["patientID", "bp_afd"]
    with open(args.out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir",
                        type=str,
                        help="Input directory containing lumen segmentations.")
    parser.add_argument("out_csv", type=str, help="Output csv report")
    args = parser.parse_args()
    main(args)
