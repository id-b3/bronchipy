#!/usr/bin/env python3

import os
import argparse
import multiprocessing as mp
import logging

from bronchipy.airwaytree import AirwayTree

logging.basicConfig(level=logging.ERROR)


def calc_tapering(id, dir_path, output_file):
    """
    This function takes in an id and a directory path.
    It reads the csv files for that id in the directory and calculates tapering.
    It returns the tapering result for that id.
    """
    volume_nii = "./bronchipy/assets/fixed_lumen_segmentation.nii.gz"

    try:
        # create file paths using id_pattern.csv
        branch_file = os.path.join(dir_path, f"{id}_centrelines.csv")
        inner_file = os.path.join(dir_path, f"{id}_inner.csv")
        inner_rad_file = os.path.join(dir_path, f"{id}_inner_radii.csv")
        outer_file = os.path.join(dir_path, f"{id}_outer.csv")
        outer_rad_file = os.path.join(dir_path, f"{id}_outer_radii.csv")

        airway_tree = AirwayTree(branch_file=branch_file,
                                 inner_file=inner_file,
                                 inner_radius_file=inner_rad_file,
                                 outer_file=outer_file,
                                 outer_radius_file=outer_rad_file,
                                 volume=volume_nii)
        tapering_l = airway_tree.taper_lumen
        tapering_t = airway_tree.taper_total
        with open(f"{output_file}/{id}_tapering.txt", "w") as f:
            f.write(f"{id},{airway_tree.tapers_lumen},{airway_tree.tapers_total}")

        return f"{id},{tapering_l[0]},{tapering_l[1]},{tapering_l[2]},{tapering_t[0]},{tapering_t[1]},{tapering_t[2]}"
    except FileNotFoundError as e:
        print(f"File not found for {id}\n{e}")
        return f"{id},,,"


def main(input_dir, output_file):
    """
    This function takes in an input directory path and output csv path.
    It gets a list of unique ids from the csv files in the directory.
    Then, it calculates tapering for each participant using multiprocessing.
    It writes the result to the output csv file.
    """
    # get list of unique ids
    ids = list(
        set([
            os.path.splitext(file)[0].split('_')[0]
            for file in os.listdir(input_dir)
        ]))
    # create pool of processes
    pool = mp.Pool(mp.cpu_count())
    # calculate tapering for each id using multiprocessing
    results = [
        pool.apply_async(calc_tapering, args=(id, input_dir, output_file)) for id in ids
    ]
    tapering_results = [result.get() for result in results]
    # write output to csv file
    with open(f"{output_file}/total.csv", 'w') as f:
        f.write('id,tapering\n')
        for i in range(len(ids)):
            f.write(f'{tapering_results[i]}\n')


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(
        description='Calculate tapering for each participant.')
    parser.add_argument('input_dir', type=str, help='Path to input directory')
    parser.add_argument('output_file',
                        type=str,
                        help='Path to output csv file')
    args = parser.parse_args()
    # run main function
    main(args.input_dir, args.output_file)
