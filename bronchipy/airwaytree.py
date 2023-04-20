from functools import reduce
from math import pi, pow
import pkg_resources

from ants import image_read
import nibabel as nib
import pandas as pd
import numpy as np
import logging

from .calc.measure_airways import calc_branch_length, calc_tapering
from .calc.summary_stats import calc_pi10, fractal_dimension
from .io import branchio as brio
from .util.imageoperations import affine_register


class AirwayTree:

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        branch_file: str
        inner_file: str
        inner_radius_file: str
        outer_file: str
        outer_radius_file: str
        volume: str
            The volume file in NIFTI format. REQUIRED
        tree_csv: str
            If loading from airway_tree csv, use this.

        Returns
        ----------
        AirwayTree Object containing volume information and the airway tree data.

        """
        logging.basicConfig(level=logging.INFO)
        if "volume" not in kwargs:
            logging.error("Missing volume file!")
            raise

        vol_header = nib.load(kwargs["volume"]).header
        self.vol_dims = vol_header.get_data_shape()
        self.vol_vox_dims = vol_header.get_zooms()
        self.paths = {}

        if "config" in kwargs:
            self.config = kwargs.get("config", None)
        else:
            # TODO: IMPLEMENT CONFIG FILE!!
            self.config = {"min_length": 5.0}
        logging.debug(
            f"Ignoring airways smaller than {self.config['min_length']}")

        if "tree_csv" in kwargs:
            self.tree = brio.load_tree_csv(kwargs["tree_csv"])
        else:
            self.files = {
                "branch": kwargs.get("branch_file", None),
                "inner": kwargs.get("inner_file", None),
                "inner_rad": kwargs.get("inner_radius_file", None),
                "outer": kwargs.get("outer_file", None),
                "outer_rad": kwargs.get("outer_radius_file", None),
                "vol": kwargs.get("volume", None),
            }
            self.tree = self.organise_tree()
            self.get_tapering()
            # Drop branches smaller than minimum length
            self.tree = self.tree[self.tree.length > self.config['min_length']]

    def organise_tree(self, full: bool = False) -> pd.DataFrame:
        """
        Takes the input files and combines them into a single merged dataframe.
        Calculates and inserts columns containing branch area data too.
        Uses the config to determine minimum length, etc.

        Returns
        -------
        A dataframe that is the merged combination of all csvs.
        """
        # Load branches csv into dataframe
        branch_df = brio.load_branch_csv(self.files["branch"])
        # Apply the voxel dimensions to the points and create a data entry containing the centreline points in mm.
        branch_df["centreline"] = branch_df.apply(
            lambda row: [self.vox_to_mm(point) for point in row.points],
            axis=1)
        logging.debug(branch_df.columns)
        # Add entry describing the number of points in the airway measurement.
        branch_df["num_points"] = branch_df.apply(lambda row: len(row.points),
                                                  axis=1)
        # Calculate and add the branch lengths in mm.
        branch_df["length"] = branch_df.apply(
            lambda row: calc_branch_length(row.centreline), axis=1)

        # Load inner measurements csvs into dataframes
        inner_df = brio.load_csv(self.files["inner"], True)
        inner_df.drop(
            "generation", axis=1,
            inplace=True)  # Redundant as branch_df already has generations
        # Calculate the area from the radius and insert as new column. Using pi*r^1
        inner_df["inner_global_area"] = inner_df.apply(
            lambda row: pow(row.inner_radius, 2) * pi, axis=1)

        # Load inner smoothed measurements csvs into dataframes
        inner_radius_df = brio.load_local_radius_csv(self.files["inner_rad"],
                                                     True)

        # Load outer measurements csvr into dataframes
        outer_df = brio.load_csv(self.files["outer"], False)
        outer_df.drop("generation", axis=1, inplace=True)
        # Calculate the area from the radius and insert as new column.
        outer_df["outer_global_area"] = outer_df.apply(
            lambda row: pow(row.outer_radius, 2) * pi, axis=1)

        # Load outer smoothed measurements csvs into dataframes
        outer_radius_df = brio.load_local_radius_csv(self.files["outer_rad"],
                                                     False)
        logging.debug(outer_radius_df.dtypes)

        # Combine all the loaded data frames based on branches ID.
        all_dfs = [
            branch_df, inner_df, inner_radius_df, outer_df, outer_radius_df
        ]
        for df in all_dfs:
            logging.debug(df.dtypes)

        organised_tree = reduce(
            lambda left, right: pd.merge(
                left, right, on=["branch"], how="outer"),
            all_dfs,
        )
        organised_tree.set_index("branch", inplace=True)

        self.dropped_branches = organised_tree[organised_tree.outer_global_area
                                               == 0.0]
        organised_tree = organised_tree[
            organised_tree.outer_global_area != 0.0]

        # Calculate Global Wall Area and WA%
        organised_tree["wall_global_area"] = organised_tree.apply(
            lambda row: row.outer_global_area - row.inner_global_area, axis=1)
        organised_tree["wall_global_area_perc"] = organised_tree.apply(
            lambda row: (row.wall_global_area / row.outer_global_area) * 100,
            axis=1)

        # Calculate Global Wall Thickness and WT%
        organised_tree["wall_global_thickness"] = organised_tree.apply(
            lambda row: row.outer_radius - row.inner_radius, axis=1)
        organised_tree["wall_global_thickness_perc"] = organised_tree.apply(
            lambda row: (row.wall_global_thickness - row.outer_radius) * 100,
            axis=1)

        # Calculate Area Tapering
        organised_tree[["lumen_tapering", "lumen_tapering_perc", "interpolated_lumen"]] = organised_tree.apply(
            lambda row: calc_tapering(
                row.inner_radii, row.centreline, use_robust=True),
            axis=1, result_type='expand'
        )

        organised_tree[["total_tapering", "total_tapering_perc", "interpolated_outer"]] = organised_tree.apply(
            lambda row: calc_tapering(
                row.outer_radii, row.centreline, use_robust=True),
            axis=1, result_type='expand'
        )

        # Get midpoint co-ordinates
        organised_tree["x"] = organised_tree.apply(
            lambda row: row.points[int(len(row.points) / 2)][0], axis=1)
        organised_tree["y"] = organised_tree.apply(
            lambda row: row.points[int(len(row.points) / 2)][1], axis=1)
        organised_tree["z"] = organised_tree.apply(
            lambda row: row.points[int(len(row.points) / 2)][2], axis=1)

        return organised_tree.round(3)

    def set_minimum_length(self, minlen: float = 5.0):
        """
        Set the minimum length of a branch in the airway tree.

        Parameters
        ----------
        minlen: float = 5
            Minimum length of a branch in millimeters.
        """
        self.config["min_length"] = minlen

    def get_airway_count(self) -> int:
        """
        Method to return the number of branches in the airway tree.
        Returns
        -------
        Number of branches in airway tree.
        """
        return self.tree.shape[0]

    def get_tapering(self):

        # Calculating tapering
        self.trace_paths()

        lumen_tapers = []
        total_tapers = []

        # Loop through each path in the list of paths
        for path in self.paths:
            # Select the rows corresponding to the current path
            path_tree = self.tree.loc[path]
            # Extract the inner radii, outer radii, and centreline data

            # Calculate the lumen taper and append it to lumen_tapers
            lum_taper = calc_tapering(path_tree.inner_radii,
                                      path_tree.centreline,
                                      use_robust=True)[0]
            lumen_tapers.append(lum_taper)
            # Calculate the total taper and append it to total_tapers
            tot_taper = calc_tapering(path_tree.outer_radii,
                                      path_tree.centreline,
                                      use_robust=True)[0]
            total_tapers.append(tot_taper)

        self.tapers_lumen = lumen_tapers
        self.tapers_total = total_tapers

        self.taper_lumen = [
            np.mean(lumen_tapers),
            np.std(lumen_tapers),
            np.median(lumen_tapers)
        ]
        self.taper_total = [
            np.mean(total_tapers),
            np.std(total_tapers),
            np.median(total_tapers)
        ]

    def get_pi10(self,
                 plot_name: str = None,
                 plot_path: str = None,
                 max_gen: int = 5) -> float:
        plot = False

        pi10_tree = self.tree[(self.tree.generation <= max_gen)]
        logging.info(
            f"Calculating Pi10 for generations {pi10_tree.generation.unique()}"
        )
        if plot_name is not None and plot_path is not None:
            plot = True

        self.pi10 = calc_pi10(
            pi10_tree["wall_global_area"],
            pi10_tree["inner_radius"],
            name=plot_name,
            save_dir=plot_path,
            plot=plot,
        )
        return self.pi10

    def get_airway_fractal_dimension(self, seg_path: str) -> float:
        fixed_path = pkg_resources.resource_filename(
            'bronchipy', 'assets/fixed_lumen_segmentation.nii.gz')
        fixed = image_read(fixed_path)
        moving = image_read(str(seg_path))
        air_tree = affine_register(fixed, moving)
        n, r = fractal_dimension(air_tree)
        afd_arr = -np.diff(np.log(n)) / np.diff(np.log(r))
        self.afd = np.mean(afd_arr[2:-2])
        return self.afd

    def vox_to_mm(self, point: tuple) -> tuple:
        """
        Takes a tuple x, y, z coordinate and applies the voxel dimensions to it.
        Parameters
        ----------
        point: tuple
            x, y, z co-ordinates

        Returns
        -------
        tuple
            x, y, z coordinates in millimeters
        """
        return (
            point[0] * self.vol_vox_dims[0],
            point[1] * self.vol_vox_dims[1],
            point[2] * self.vol_vox_dims[2],
        )

    def trace_paths(self):
        """
        For each terminal branch in the airway tree self.tree, looks back at the parent recursively to create a list of IDs from terminal to initial branch (trachea, 0). 
        Reverses the list to get the path trachea->terminal branch. Performs it for all terminal branches and saves it to self.paths.
        Args:
            None
        Returns:
            None
        """

        def _get_terminal_branches():
            """
            Gets terminal branches by finding any branch without a parent
            """
            empty_rows = self.tree[self.tree['children'].apply(
                lambda x: len(x) == 0)]
            return empty_rows.index.to_list()

        # Loop through all terminal branches
        for branch_id in _get_terminal_branches():

            # Initialize an empty list to store the path from terminal to initial branch
            path = []

            # Start from the current terminal branch and trace back to the root of the tree
            current_branch_id = branch_id
            while current_branch_id != 0:
                path.append(current_branch_id)
                current_branch = self.get_branch(current_branch_id)
                if current_branch is not None:
                    current_branch_id = current_branch.parent
                else:
                    current_branch = self.dropped_branches.loc[
                        current_branch_id]
                    if current_branch is not None:
                        current_branch_id = current_branch.parent
                    else:
                        logging.debug(
                            f"Broken path {'->'.join([str(i) for i in path])}")
                        break

            # Add the initial branch (trachea) to the end of the path
            # path.append(0)

            # Reverse the path to get the correct order (trachea -> terminal branch)
            path.reverse()

            # Add the path to the dictionary of paths
            self.paths[branch_id] = path

    def get_branch(self, branch_id: int) -> pd.Series:
        """

        Parameters
        ----------
        branch_id: int
            id value of the branch.

        Returns
        -------
        Series containing branch information

        See Also
        -------
        branch: The branch ID
        generation: The branch generation
        parent: The branch ID of the Parent branch
        children: list[int] A list of branch IDs of the Children branches
        points: list[(Tuple)] A list of (x, y, z) tuples of branch points along centreline in voxels (not in mm)
        centreline: list [(Tuple)] A list of (x, y, z) tuples of branch points along centreline in millimeters
        inner_radius: The branch lumen global (summary) radius in mm
        inner_intensity: The branch lumen global (summary) intensity in HU
        inner_samples: Number of samples used for global measurement
        inner_global_area: The global luminal area of the branch in mm^1
        inner_radii: list[float] A list of non-smoothed measurements of luminal local radii
        outer_radius: The branch total branch thickness global (summary) radius in mm
        outer_intensity: The branch total branch thickness global (summary) intensity in HU
        outer_samples: Number of samples used for global measurement
        outer_global_area: The branch global total branch area measurement in mm^1
        outer_radii: list[float] A list of non-smoothed measurements of total branch thickness local radii
        wall_global_area: The global wall area of the branch
        wall_global_area_perc: Global WA% of the branch
        wall_global_thickness: Global Wall Thickness of the branch
        wall_global_thickness_perc: Global WT% of the branch
        lumen_tapering: Luminal tapering from start to end of branch
        lumen_tapering_perc: Luminal tapering as a percentage of starting lumen diameter.
        total_tapering: Total airway (wall+lumen) tapering
        total_tapering_perc: Total airway tapering as percentage of starting total diameter.
        x: X voxel at midpoint of branch
        y: Y voxel at midpoint of branch
        z: Z voxel at midpoint of branch
        """
        try:
            return self.tree.loc[branch_id]
        except KeyError as e:
            logging.error(f"No branch with id {e}.")
            return None
