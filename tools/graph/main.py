"""
Code used by 'switch compare' and 'switch graph' to run the graphing functions.

See docs/Graphs.md to learn how to add graphs.
"""
# Standard packages
import functools
import importlib
import os
import warnings
from typing import List, Dict, Optional
import datetime

# Third-party packages
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
import plotnine

# Local imports
import logging
from .maps import GraphMapTools
from switch_model.tools.graph.maps import GraphMapTools
from switch_model.utilities import StepTimer#, get_module_list, query_yes_no, catch_exceptions
# from switch_model.solve import get_module_list
# When True exceptions that are thrown while graphing will be caught
# and outputted to console as a warning instead of an error
CATCH_EXCEPTIONS = True


# List of graphing functions. Every time a function uses the @graph() decorator,
# the function gets registered here.
registered_graphs = {}


def graph(
        name,
        title=None,
        supports_multi_scenario=False,
        requires_multi_scenario=False,
        is_long=False,
        note=None
):
    """
    This function should be used as a decorator to register a graphing function.
    Graphing functions are functions that are run by 'switch graph' or 'switch compare'.
    Graphing functions take one argument, an instance of GraphTools.

    @param name: name of the graph created by the decorated function.
                    This is used as the filename for the output.png file.
    @param title: The title to be put on the graph.
    @param supports_multi_scenario: If true, the function is responsible for graphing data from multiple scenarios.
    @param requires_multi_scenario: If true, the graphing function will only be run when comparing multiple scenarios
    @param is_long: If true, the --skip-long CLI flag will skip this function
    @param note: Note to add to the bottom of the graph
    """

    def decorator(func):
        @functools.wraps(func)
        # @catch_exceptions("Failed to run a graphing function.", should_catch=CATCH_EXCEPTIONS)
        def wrapper(tools: GraphTools):
            if tools.skip_long and is_long:
                return

            if tools.num_scenarios < 2 and requires_multi_scenario:
                return

            func(tools)

        wrapper.name = name
        wrapper.multi_scenario = supports_multi_scenario or requires_multi_scenario
        wrapper.title = title
        wrapper.note = note

        if name in registered_graphs:
            raise Exception(f"Graph '{name}' already exists. Make sure to pick a unique name.")

        registered_graphs[name] = wrapper
        return wrapper

    return decorator


class Scenario:
    """
    Stores the information related to a scenario such as the scenario name (used while graphing)
    and the scenario path.

    Also allows doing:

    with scenario:
        # some operation

    Here, some operation will be run as if the working directory were the directory of the scenario
    """
    root_path = os.getcwd()

    # def __init__(self, rel_path=".", name=""):
    #     self.path = os.path.normpath(os.path.join(Scenario.root_path, rel_path))
    #     self.name = name

    #     if not os.path.isdir(self.path):
    #         raise Exception(f"Directory does not exist: {self.path}")
    def __init__(self, rel_path=".", name="", inputs_dir=None, outputs_dir=None):
        self.path = os.path.normpath(os.path.join(Scenario.root_path, rel_path))
        self.name = name
        # NEW: absolute (or scenario-relative) dirs for inputs/outputs
        self.inputs_dir = inputs_dir  # e.g., "/abs/path/to/inputs" or "inputs"
        self.outputs_dir = outputs_dir  # e.g., "/abs/path/to/outputs" or "outputs"

        if not os.path.isdir(self.path):
            raise Exception(f"Directory does not exist: {self.path}")
        
    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(Scenario.root_path)


class TransformTools:
    """
    Provides helper functions that transform dataframes
    to add value. Can be accessed via tools.transform in graph() functions.
    """

    def __init__(self, graph_tools, time_zone="US/Pacific"):
        self.time_zone = time_zone
        self.tools = graph_tools

    def gen_type(self, df: pd.DataFrame, map_name='default', gen_tech_col='gen_tech',
                 energy_source_col='gen_energy_source', drop_previous_col=True,
                 others=None):
        """
        Returns a dataframe that contains a column 'gen_type'.

        By default 'gen_type' is the aggregation of 'gen_tech' + 'gen_energy_source'
        however this can be overidden in graph_tech_types.csv
        """
        # If there's no mapping, we simply make the mapping the sum of both columns
        # Read the tech_colors and tech_types csv files.
        try:
            cols = ["map_name", "gen_type", "gen_tech", "energy_source", "scenario_index"]
            tech_types = self.tools.get_dataframe("graph_tech_types.csv", from_inputs=True, drop_scenario_info=False)[cols]
        except FileNotFoundError:
            df['gen_type'] = df[gen_tech_col] + "_" + df[energy_source_col]
            return df
        tech_types = tech_types[tech_types['map_name'] == map_name].drop('map_name', axis=1)
        # If we got many scenarios "scenario_name" will exist in tech_types and in that case
        # we want to merge by scenario
        left_on = [gen_tech_col, energy_source_col]
        right_on = ["gen_tech", "energy_source"]
        if "scenario_index" in df:
            left_on.append("scenario_index")
            right_on.append("scenario_index")
        else:
            tech_types = tech_types.drop(columns=["scenario_index"]).drop_duplicates()
        df = df.merge(
            tech_types,
            left_on=left_on,
            right_on=right_on,
            validate="many_to_one",
            how="left")
        df["gen_type"] = df["gen_type"].fillna("Other")  # Fill with Other so the colors still work
        if drop_previous_col:
            df = df.drop([gen_tech_col, energy_source_col], axis=1)
        if others is not None:
            df["gen_type"] = df["gen_type"].replace(others, "Other")
        return df

    def build_year(self, df, build_year_col="build_year"):
        """
        Replaces all the build years that aren't a period with the value "Pre-existing".
        """
        # Get list of valid periods
        periods = self.tools.get_dataframe("periods", from_inputs=True)["INVESTMENT_PERIOD"].astype("str")
        df = df.copy()  # Make copy to not modify source
        df[build_year_col] = df[build_year_col].apply(
            lambda b: str(b) if str(b) in periods.values else "Pre-existing"
        ).astype("category")
        return df
    def timestamp(self, df, key_col="timestamp", use_timepoint=False):
        """
        Adds to df:
        - period: from timeseries.csv
        - tp_duration: from timeseries.csv (ts_duration_of_tp)
        - time_row: default period (overridable by graph_timestamp_map.csv)
        - time_column: default timeseries (overridable by graph_timestamp_map.csv)
        - hour: 0..23, from trailing index or order-in-timeseries
        - day: 0.., from trailing index or order-in-timeseries
        Note: does NOT parse datetimes; labels like 2050_p27_0 are treated as strings.
        """
        import numpy as np
        import pandas as pd

        # Load and normalize inputs
        timepoints = self.tools.get_dataframe("timepoints.csv", from_inputs=True, drop_scenario_info=False).copy()
        timeseries = self.tools.get_dataframe("timeseries.csv", from_inputs=True, drop_scenario_info=False).copy()
        timepoints.columns = [c.lower() for c in timepoints.columns]
        timeseries.columns = [c.lower() for c in timeseries.columns]

        required_tp = {"timepoint_id", "timestamp", "timeseries"}
        required_ts = {"timeseries", "ts_period", "ts_duration_of_tp"}
        missing_tp = required_tp - set(timepoints.columns)
        missing_ts = required_ts - set(timeseries.columns)
        if missing_tp:
            raise KeyError(f"timepoints.csv missing columns: {sorted(missing_tp)}")
        if missing_ts:
            raise KeyError(f"timeseries.csv missing columns: {sorted(missing_ts)}")

        if "scenario_index" not in timepoints.columns:
            timepoints["scenario_index"] = 0
        if "scenario_index" not in timeseries.columns:
            timeseries["scenario_index"] = 0

        # Merge ts_* into timepoints to form a single map
        tp_map = timepoints.merge(
            timeseries,
            how="left",
            on=["timeseries", "scenario_index"],
            validate="many_to_one",
            copy=False,
        )

        # Build compact mapping and compute order index k within each timeseries
        tp_map = tp_map.loc[:, [
            "timepoint_id", "timestamp", "timeseries", "ts_period", "ts_duration_of_tp"
        ]].drop_duplicates().copy()

        tp_map["timestamp"] = tp_map["timestamp"].astype(str)
        tp_map = tp_map.rename(columns={
            "timepoint_id": "timepoint",
            "ts_period": "period",
            "ts_duration_of_tp": "tp_duration"
        })
        # k = position within each timeseries (0..N-1), ordered by timepoint
        tp_map = tp_map.sort_values(["timeseries", "timepoint"]).copy()
        tp_map["k"] = tp_map.groupby("timeseries").cumcount()
        tp_map["period"] = tp_map["period"].astype("category")

        # Attach mapping to df once
        if use_timepoint:
            df = df.rename(columns={key_col: "timepoint"})
            df = df.merge(tp_map, how="left", on="timepoint", copy=False)
        else:
            df = df.rename(columns={key_col: "timestamp"})
            df["timestamp"] = df["timestamp"].astype(str)
            df = df.merge(tp_map, how="left", on="timestamp", copy=False)

        # Derive k from label suffix if present, else use merged k
        suf = df["timestamp"].astype(str).str.extract(r"_(\d+)$", expand=False)
        k_from_suffix = pd.to_numeric(suf, errors="coerce")

        # prefer suffix when available, else fall back to map-derived k
        k = k_from_suffix.where(k_from_suffix.notna(), df.get("k", np.nan))
        k = pd.to_numeric(k, errors="coerce")

        # Compute hour/day only where k is known
        m = k.notna()
        df.loc[m, "hour"] = (k[m].astype(int) % 24).astype(int)
        df.loc[m, "day"] = (k[m].astype(int) // 24).astype(int)

        # Optional override from graph_timestamp_map.csv
        try:
            ts_map = self.tools.get_dataframe("graph_timestamp_map.csv", from_inputs=True, force_one_scenario=True).copy()
            ts_map.columns = [c.lower() for c in ts_map.columns]
            if "timestamp" in ts_map.columns:
                ts_map["timestamp"] = ts_map["timestamp"].astype(str)
            df = df.merge(ts_map, how="left", on="timestamp", copy=False)
        except FileNotFoundError:
            pass

        # Defaults if no override provided
        if "time_row" not in df.columns or df["time_row"].isna().all():
            df["time_row"] = df["period"]
        if "time_column" not in df.columns or df["time_column"].isna().all():
            df["time_column"] = df["timeseries"]

        # Do NOT create datetime/season from labels
        return df

    # def timestamp(self, df, key_col="timestamp", use_timepoint=False):
        """
        Adds the following columns to the dataframe:
        - time_row: by default the period but can be overridden by graph_timestamp_map.csv
        - time_column: by default the timeseries but can be overridden by graph_timestamp_map.csv
        - datetime: timestamp formatted as a US/Pacific Datetime object
        - hour: The hour of the timestamp (US/Pacific timezone)
        """
        timepoints = self.tools.get_dataframe(filename="timepoints.csv", from_inputs=True, drop_scenario_info=False)
        timeseries = self.tools.get_dataframe(filename="timeseries.csv", from_inputs=True, drop_scenario_info=False)

        # Normalize column names to lower-case for robustness
        timepoints.columns = [c.lower() for c in timepoints.columns]
        timeseries.columns = [c.lower() for c in timeseries.columns]

        # Ensure a scenario column exists on both sides (single-scenario -> 0)
        if "scenario_index" not in timepoints.columns:
            timepoints["scenario_index"] = 0
        if "scenario_index" not in timeseries.columns:
            timeseries["scenario_index"] = 0

        # Sanity check for the key
        if "timeseries" not in timepoints.columns or "timeseries" not in timeseries.columns:
            raise KeyError("Expected a 'timeseries' column in both timepoints.csv and timeseries.csv.")

        # Merge on timeseries + scenario_index (case-insensitive-safe now)
        timepoints = timepoints.merge(
            timeseries,
            how="left",
            left_on=["timeseries", "scenario_index"],
            right_on=["timeseries", "scenario_index"]
        )

        timepoints = timepoints.merge(
            timeseries,
            how='left',
            left_on=['timeseries', 'scenario_index'],
            right_on=['timeseries', 'scenario_index'],
            validate="many_to_one"
        )
        timestamp_mapping = timepoints[
            ["timepoint_id", "timestamp", "ts_period", "timeseries", "ts_duration_of_tp"]].drop_duplicates()
        
        timestamp_mapping = timestamp_mapping.rename({
            "ts_period": "period",
            "timepoint_id": "timepoint",
            "ts_duration_of_tp": "tp_duration"}, axis=1)
        timestamp_mapping = timestamp_mapping.astype({"period": "category"})

        if use_timepoint:
            df = df.rename({key_col: "timepoint"}, axis=1)
            df = df.merge(
                timestamp_mapping,
                how='left',
                on="timepoint"
            )
        else:
            df = df.rename({key_col: "timestamp"}, axis=1)
            df = df.merge(
                timestamp_mapping,
                how='left',
                on="timestamp",
            )

        try:
            # TODO support using graph_timestamp_map on multiple scenarios
            df = df.merge(
                self.tools.get_dataframe("graph_timestamp_map.csv", from_inputs=True, force_one_scenario=True),
                how='left',
                on="timestamp",
            )
        except FileNotFoundError:
            df["time_row"] = df["period"]
            df["time_column"] = df["timeseries"]

        # Add datetime and hour column
        df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H").dt.tz_localize("utc").dt.tz_convert(
            self.time_zone)
        df["hour"] = df["datetime"].dt.hour
        season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
        df["season"] = df["datetime"].dt.quarter.apply(lambda x: season_map[x])

        return df
    # def timestamp(self, df, key_col="timestamp", use_timepoint=False):
    #     """
    #     Adds:
    #     - time_row: by default the period (or from graph_timestamp_map.csv if present)
    #     - time_column: by default the timeseries (or from graph_timestamp_map.csv if present)
    #     - datetime: timestamp as timezone-aware Datetime in self.time_zone
    #     - hour: hour of day
    #     - season: quarter name (Winter/Spring/Summer/Fall)
    #     """
    #     # Load inputs
    #     timepoints = self.tools.get_dataframe("timepoints.csv", from_inputs=True, drop_scenario_info=False)
    #     timeseries = self.tools.get_dataframe("timeseries.csv", from_inputs=True, drop_scenario_info=False)

    #     # Normalize column names to lower-case for robustness
    #     timepoints.columns = [c.lower() for c in timepoints.columns]
    #     timeseries.columns = [c.lower() for c in timeseries.columns]

    #     # Ensure these columns exist per your files
    #     required_tp = {"timepoint_id", "timestamp", "timeseries"}
    #     required_ts = {"timeseries", "ts_period", "ts_duration_of_tp"}
    #     missing_tp = required_tp - set(timepoints.columns)
    #     missing_ts = required_ts - set(timeseries.columns)
    #     if missing_tp:
    #         raise KeyError(f"timepoints.csv is missing columns: {sorted(missing_tp)}")
    #     if missing_ts:
    #         raise KeyError(f"timeseries.csv is missing columns: {sorted(missing_ts)}")

    #     # Add scenario_index=0 for single-scenario inputs
    #     if "scenario_index" not in timepoints.columns:
    #         timepoints["scenario_index"] = 0
    #     if "scenario_index" not in timeseries.columns:
    #         timeseries["scenario_index"] = 0

    #     # Single merge to bring ts_* into timepoints; avoid duplicate merge that causes _x/_y suffixes
    #     timepoints = timepoints.merge(
    #         timeseries,
    #         how="left",
    #         left_on=["timeseries", "scenario_index"],
    #         right_on=["timeseries", "scenario_index"],
    #         validate="many_to_one",
    #         copy=False,
    #     )

    #     # Build compact mapping frame
    #     # Ensure timestamp is string of form YYYYMMDDHH for downstream parsing
    #     tp_map = timepoints.loc[:, [
    #         "timepoint_id", "timestamp", "timeseries", "ts_period", "ts_duration_of_tp"
    #     ]].drop_duplicates().copy()

    #     # Coerce timestamp to string in case it was int
    #     tp_map["timestamp"] = tp_map["timestamp"].astype(str)

    #     tp_map = tp_map.rename(columns={
    #         "ts_period": "period",
    #         "timepoint_id": "timepoint",
    #         "ts_duration_of_tp": "tp_duration"
    #     })
    #     tp_map = tp_map.astype({"period": "category"})

    #     # Attach mapping to df
    #     if use_timepoint:
    #         df = df.rename(columns={key_col: "timepoint"})
    #         df = df.merge(tp_map, how="left", on="timepoint", copy=False)
    #     else:
    #         df = df.rename(columns={key_col: "timestamp"})
    #         df["timestamp"] = df["timestamp"].astype(str)
    #         df = df.merge(tp_map, how="left", on="timestamp", copy=False)
    #     if np.issubdtype(df["timestamp"].dtype, np.number):
    #         df["timestamp"] = df["timestamp"].astype("int64").astype(str)
    #     else:
    #         # Try parsing as datetime; fallback leaves strings as-is
    #         parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    #         mask = parsed.notna()
    #         df.loc[mask, "timestamp"] = parsed[mask].dt.strftime("%Y%m%d%H")

    #     df = df.merge(tp_map, how="left", on="timestamp", copy=False)
    #     # Optional override from graph_timestamp_map.csv
    #     try:
    #         ts_map = self.tools.get_dataframe("graph_timestamp_map.csv", from_inputs=True, force_one_scenario=True)
    #         # Expect columns: timestamp, time_row, time_column (timestamp must match YYYYMMDDHH string)
    #         ts_map.columns = [c.lower() for c in ts_map.columns]
    #         if "timestamp" in ts_map.columns:
    #             ts_map["timestamp"] = ts_map["timestamp"].astype(str)
    #         df = df.merge(ts_map, how="left", on="timestamp", copy=False)
    #     except FileNotFoundError:
    #         pass

    #     # Defaults if no override provided
    #     if "time_row" not in df.columns or df["time_row"].isna().all():
    #         df["time_row"] = df["period"]
    #     if "time_column" not in df.columns or df["time_column"].isna().all():
    #         df["time_column"] = df["timeseries"]

    #     # Datetime fields
    #     # Input timestamps are YYYYMMDDHH in UTC; convert to target zone
    #     df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H", errors="coerce") \
    #                         .dt.tz_localize("utc").dt.tz_convert(self.time_zone)
    #     df["hour"] = df["datetime"].dt.hour

    #     # Season by quarter
    #     season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    #     df["season"] = df["datetime"].dt.quarter.map(season_map)

    #     return df


    def load_zone(self, df, load_zone_col="load_zone"):
        """
        Adds a 'region' column that is usually load_zone's state.
        'region' is what comes before the first underscore. If no underscores are present
        defaults to just using the load_zone.
        """
        df = df.copy()  # Don't modify the source
        df["region"] = df[load_zone_col].apply(
            lambda z: z.partition("_")[0]
        )
        return df


class Figure:
    """
    This class simply stores a Matplotlib figure and axes. It's only purpose
    is to make code in FigureHandler more readable.
    """

    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = axes

    def save_figure(self, path, dpi_scale=1.2):
        self.fig.savefig(path, bbox_inches="tight", dpi=int(100 * dpi_scale))
        plt.close(self.fig)  # Close figure to save on memory
    
    def add_note(self, note):
        if note is not None:
            self.fig.text(0.5, -0.1, note, wrap=True, horizontalalignment='center', fontsize=12)


class FigureHandler:
    """
    This class handles the storage of Matplotlib figures during graphing and is responsible for
    saving these figures to .png files.
    """

    def __init__(self, output_dir: Optional[str], scenarios):
        self._output_dir: Optional[str] = output_dir
        self._scenarios: List[Scenario] = scenarios

        # This dictionary stores the figures.
        # It is a map of file names to a list of figures for that file.
        # If there are multiple figures for one files, the figures will be plotted side by side.
        self._figures: Dict[str, List[Figure]] = {}

        # These properties will get set in reset()
        self._default_filename = None
        self._title = None
        self._note = None
        self._allow_multiple_figures = None  # If False there can only be one figure per file

    def set_properties(self, default_filename, title, note, allow_multiple_figures):
        """
        Called before running a graphing function to set the properties
        """
        self._default_filename = default_filename
        self._title = title
        self._note = note if note is not None else ""
        self._allow_multiple_figures = allow_multiple_figures

    def add_figure(self, fig, axes=None, filename=None, title=None):
        # Use default name if unspecified
        if filename is None:
            if self._default_filename:
                filename = self._default_filename
            else:
                # use timestamp (e.g. unnamed_plot_20251020_182045)
                filename = f"unnamed_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if not str(filename).lower().endswith(".png"):
            filename = f"{filename}.png"
        if title is None:
            title = self._title

        # Set a title for the figure
        if title is not None:
            fig.suptitle(title)

        # Create our figure
        figure = Figure(fig, axes)

        # Add the Figure to our list of figures
        if filename not in self._figures:
            self._figures[filename] = [figure]
        elif self._allow_multiple_figures:
            self._figures[filename].append(figure)
        else:
            raise Exception(f"A figure with name '{filename}' already exists and multiple figures are not allowed for"
                            f" {self._default_filename}.")

    def get_axes(self, name=None):
        if name is None:
            name = self._default_filename
        if name not in self._figures:
            return None
        figures = self._figures[name]
        if len(figures) > 1:
            raise Exception("Can't call get_axes() when multiple figures exist.")
        return figures[0].axes  # We access the 0 index since we expect there to only be 1 figure

    def save_figures(self):
        if self._output_dir is None:
            raise Exception("Cannot call save_figures() when the output directory is None.")
        
        for filename, figures in self._figures.items():
            # If we have a single figure just save it
            if len(figures) == 1:
                figures[0].add_note(self._note)
                print((self._output_dir))
                print(filename)
                figures[0].save_figure(os.path.join(self._output_dir, filename))
                continue

            # If we have multiple figures, save each one to a separate file and then concat the files
            for i, fig in enumerate(figures):
                # Get note from self._note and the scenario name and add it to the figure
                fig.add_note(("" if self._note is None else self._note) + f"\nScenario: {self._scenarios[i].name}")
                fig.save_figure(os.path.join(self._output_dir, filename + "_" + str(i)))

            # If we have multiple figures, concat them into a single one
            FigureHandler._concat_figures(os.path.join(self._output_dir, filename), len(figures))

        self._figures = {}  # Reset our list of figures

    @staticmethod
    def _concat_figures(basepath, n):
        """
        This function merges n figures together side by side.
        The figures must have the same base path and only differ in their suffix
        (_0.png, _1.png, _2.png etc).
        """
        # Get the paths of each image
        image_paths = tuple(basepath + "_" + str(i) + ".png" for i in range(n))
        # Open each image
        images = tuple(Image.open(path) for path in image_paths)

        # Get the dimension of our final figure
        height = max(map(lambda x: x.size[1], images))
        width = sum(map(lambda x: x.size[0], images))

        # Create our final figure
        concated = Image.new("RGB", (width, height), "white")

        # For each image, paste it into
        x = 0
        for image in images:
            concated.paste(image, (x, 0))
            x += image.size[0]

        # Save the concated image
        concated.save(basepath + ".png", "PNG")

        # Delete the individual images
        for image_path in image_paths:
            os.remove(image_path)


class DataHandler:
    """
    This class handles accessing and caching csv files for graphing
    """

    # When True, csv files with the same name will only be loaded once and will then get cached
    # in case they're needed again by another graphing function.
    ENABLE_DF_CACHING = True

    def __init__(self, scenarios):
        # Check that the scenario names are unique. This is required so that get_dataframe doesn't have conflicts
        all_names = list(map(lambda s: s.name, scenarios))
        if len(all_names) > len(set(all_names)):  # set() drops duplicates, so if not unique len() will be less
            raise Exception("Scenario names are not unique.")

        self._scenarios: List[Scenario] = scenarios
        # If true the current function being run should only be run
        # once with all the scenarios rather than re-run for each scenario
        self._is_multi_scenario_func = None
        self._active_scenario = 0

        # Here we store a mapping of csv file names to their dataframes.
        # Each dataframe has a column called 'scenario' that specifies which scenario
        # a given row belongs to.
        self._dfs: Dict[str, pd.DataFrame] = {}

    @property
    def scenarios(self):
        return self._scenarios

    def get_scenario_name(self, index):
        """
        Returns the scenario_name given the scenario_index.
        Can be used as follows to convert an scenario_index that's an
        index in a Dataframe to the scenario names.
        df.index = df.index.map(tools.get_scenario_name)
        """
        return self._scenarios[index].name

    def get_dataframe(self, filename, folder=None, from_inputs=False, convert_dot_to_na=False, force_one_scenario=False,
                      drop_scenario_info=True, usecols=None, **kwargs):
        """
        Returns the dataframe for the active scenario.

        @param filename: Name of the csv file to read from
        @param folder: Overrides which folder to read from.
        @param from_inputs: If true, the csv file will be read from the inputs
        @param convert_dot_to_na if True cells with "." will be replaced with na
        @param force_one_scenario if True this will only return one scenario of data even if we are running
        @param drop_scenario_info if True, we will drop the columns relating to the scenario when we are dealing with just one scenario
        a multi-scenario function.
        @param only return the following functions
        """
        if not filename.endswith(".csv") and not filename.endswith(".parquet"):
            filename += ".csv"

        path = self.get_file_path(filename, folder, from_inputs, scenario_specific=False)

        # If doesn't exist, create it
        if path not in self._dfs:
            df = self._load_dataframe(path, na_values="." if convert_dot_to_na else None, **kwargs)
            if DataHandler.ENABLE_DF_CACHING:
                self._dfs[path] = df.copy()  # We save a copy so the source isn't modified
        else:
            df = self._dfs[path].copy()  # We return a copy so the source isn't modified

        if not self._is_multi_scenario_func or force_one_scenario:
            # Filter dataframe to only the current scenario
            df = df[df['scenario_index'] == self._active_scenario]
            # Drop the columns related to the scenario
            if drop_scenario_info:
                df = df.drop(["scenario_index", "scenario_name"], axis=1)
        if usecols is not None:
            df = df[usecols]
        return df

    # def get_file_path(self, filename, folder=None, from_inputs=False, scenario_specific=True):
    #     if folder is None:
    #         folder = "inputs" if from_inputs else "outputs"

    #     path = os.path.join(folder, filename)

    #     if scenario_specific:
    #         path = os.path.join(self._scenarios[self._active_scenario].path, path)

    #     return path
    def get_file_path(self, filename, folder=None, from_inputs=False, scenario_specific=True):
        # 1) Prefer explicit folder arg if given
        if folder is None:
            # 2) Next, prefer env vars set by solve.py
            env_key = "SWITCH_INPUTS_DIR" if from_inputs else "SWITCH_OUTPUTS_DIR"
            folder = os.environ.get(env_key)
            # print('this is the folder: ',folder)
            # 3) Finally, fall back to legacy defaults
            if not folder:
                folder = "inputs" if from_inputs else "outputs"

        # If relative and scenario_specific, resolve under the current scenario path (or CWD)
        base = folder
        if scenario_specific and not os.path.isabs(base):
            base = os.path.join(self._scenarios[self._active_scenario].path, base)

        return os.path.join(base, filename)

    def _load_dataframe(self, path, dtype=None, **kwargs) -> pd.DataFrame:
        """
        Reads a csv file for every scenario and returns a single dataframe containing
        the rows from every scenario with a column for the scenario name and index.
        """
        if dtype is None:
            dtype = {"generation_project": str, "gen_dbid": str, "GENERATION_PROJECT": str}

        df_all_scenarios: List[pd.DataFrame] = []
        for i, scenario in enumerate(self._scenarios):
            df = pd.read_csv(
                os.path.join(scenario.path, path), index_col=False,
                # Fix: force the datatype to str for some columns to avoid warnings of mismatched types
                dtype=dtype,
                sep=",",
                engine="c",
                **kwargs
            )
            df['scenario_name'] = scenario.name
            df['scenario_index'] = i
            df_all_scenarios.append(df)

        return pd.concat(df_all_scenarios)
    
class GraphTools(DataHandler):
    """
    This class provides utilities to make graphing easier and standardized.
    An instance of this class gets passed as the first argument to any function that has the
    @graph() annotation.
    """

    def __init__(self, scenarios: List[Scenario], graph_dir: Optional[str] = None, skip_long=False, set_style=True):
        """
        @param scenarios list of scenarios that we should run graphing for
                graph_dir directory where graphs should be saved
        @param graph_dir folder where graphs should be outputed to
        """
        super(GraphTools, self).__init__(scenarios)
        # Default plots dir if not provided
        if graph_dir is None:
            base_out = os.environ.get("SWITCH_OUTPUTS_DIR", "outputs")
            graph_dir = os.path.join(base_out, "plots")
        os.makedirs(graph_dir, exist_ok=True)

        # Create our figure handler which handles saving figures
        self._figure_handler = FigureHandler(graph_dir, scenarios)

        # Create our figure handler which handles saving figures
        self._figure_handler = FigureHandler(graph_dir, scenarios)
        self.skip_long = skip_long

        self.num_scenarios = len(scenarios)

        # When true our graphing function is comparing across possibly many scenarios

        # Provide link to useful libraries
        self.sns = sns
        self.pd = pd
        self.np = np
        self.plt = matplotlib
        self.pn = plotnine

        if set_style:
            # Set the style to Seaborn default style
            sns.set()
            # Don't show white outline around shapes to avoid confusion
            plt.rcParams["patch.edgecolor"] = 'none'

        # Disables pandas warnings that will occur since we are constantly returning only a slice of our master dataframe
        pd.options.mode.chained_assignment = None

        self.transform = TransformTools(self)

        self.logger = logging.getLogger("switch.tools.graph")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("[%(levelname)s] %(message)s")
            h.setFormatter(fmt)
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)
        self.maps = GraphMapTools(self)

    def _create_axes(self, num_rows=1, size=(8, 5), ylabel=None, projection=None, **kwargs):
        """
        Create a set of matplotlib axes
        """
        num_columns = 1 if self._is_multi_scenario_func else self.num_scenarios
        fig = GraphTools._create_figure(
            size=(size[0] * num_columns, size[1]),
            **kwargs
        )
        ax = fig.subplots(nrows=num_rows, ncols=num_columns, sharey='row', squeeze=False, subplot_kw=dict(projection=projection))

        ax = [[ax[j][i] for j in range(num_rows)] for i in range(num_columns)]

        # Set a title to each subplot
        for col, col_plots in enumerate(ax):
            for row, a in enumerate(col_plots):
                if num_columns > 1 and row == 0:
                    a.set_title(f"Scenario: {self._scenarios[col].name}")
                if ylabel is not None:
                    if type(ylabel) == str:
                        a.set_ylabel(ylabel)
                    else:
                        a.set_ylabel(ylabel[row])

        if num_rows == 1:
            ax = [ax[i][0] for i in range(num_columns)]

        return fig, ax

    @staticmethod
    def _create_figure(size=None, xlabel=None, ylabel=None, scale=1.0,**kwargs):
        fig = plt.figure(**kwargs)

        # Set figure size based on numbers of subplots
        if size is not None:
            fig.set_size_inches(size[0], size[1])

        # Apply scale factor (e.g., 1.2 for +20%)
        scaled_size = (size[0] * scale, size[1] * scale)
        fig.set_size_inches(scaled_size[0], scaled_size[1])

        if xlabel is not None:
            fig.text(0.5, 0.01, xlabel, ha='center')
        if ylabel is not None:
            fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical')

        return fig

    def get_axes(self, filename=None, title=None, note=None, *args, **kwargs):
        """
        Returns a set of matplotlib axes that can be used to graph.

        Internally this will handle returning a different set of axes depending on the scenario
        that is active.
        """
        axes = self._figure_handler.get_axes(filename)
        if axes is None:
            fig, axes = self._create_axes(*args, **kwargs)
            self._figure_handler.add_figure(fig, axes, filename, title)

        ax = axes[self._active_scenario]

        if note is not None:
            ax.text(0.5, -0.2, note, size=12, ha='center', transform=ax.transAxes)

        return ax

    def bar_label(self, filename=None):
        """
        Adds labels to a barchart
        """
        ax = self.get_axes(filename=filename)
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", fontsize="x-small")

    def get_figure(self, *args, **kwargs):
        # Create the figure
        fig = GraphTools._create_figure(*args, **kwargs)
        # Save it to the outputs
        # Add the figure to the list of figures for that scenario
        self._figure_handler.add_figure(fig)
        # Return the figure
        return fig

    def save_figure(self, fig, filename=None):
        """
        Gets called directly from the code to save a plotnine figure or gets called from within get_figure()
        """
        # Add the figure to the list of figures for that scenario
        self._figure_handler.add_figure(fig, filename=filename)

    def pre_graphing(self, multi_scenario, name=None, title=None, note=None):
        self._is_multi_scenario_func = multi_scenario
        self._figure_handler.set_properties(
            name,
            title,
            note,
            allow_multiple_figures=not self._is_multi_scenario_func)

    def post_graphing(self):
        # Save the graphs
        self._figure_handler.save_figures()

    def get_colors(self, n=None, map_name='default'):
        """
        Returns an object that can be passed to color= when doing a bar plot.
        @param n should be specified when using a stacked bar chart as the number of bars
        @param map_name is the name of the technology mapping in use
        """
        try:
            tech_colors = self.get_dataframe(filename="graph_tech_colors.csv", from_inputs=True, force_one_scenario=True)
        except FileNotFoundError:
            return None
        filtered_tech_colors = tech_colors[tech_colors['map_name'] == map_name]
        if n is not None:
            return {r['gen_type']: [r['color']] * n for _, r in filtered_tech_colors.iterrows()}
        else:
            return {r['gen_type']: r['color'] for _, r in filtered_tech_colors.iterrows()}

    def graph_time_matrix(self, df, value_column, ylabel):
        # Add the technology type column and filter out unneeded columns
        df = self.transform.gen_type(df)
        # Keep only important columns
        df = df[["gen_type", "timestamp", value_column]]
        # Sum the values for all technology types and timepoints
        df = df.groupby(["gen_type", "timestamp"], as_index=False).sum()
        # Add the columns time_row and time_column
        df = self.transform.timestamp(df)
        # Sum across all technologies that are in the same hour and quarter
        df = df.groupby(["hour", "gen_type", "time_column", "time_row"], as_index=False,observed=False).mean(numeric_only=True)
        self.graph_matrix(df, value_column, ylabel, "time_row", "time_column")

    def graph_scenario_matrix(self, df, value_column, ylabel):
        # Add the technology type column and filter out unneeded columns
        df = self.transform.gen_type(df)
        # Keep only important columns
        df = df[["gen_type", "timestamp", value_column, "scenario_name"]]
        # Sum the values for all technology types and timepoints
        df = df.groupby(["gen_type", "timestamp", "scenario_name"], as_index=False).sum()
        # Add the columns time_row and time_column
        df = self.transform.timestamp(df)
        # Sum across all technologies that are in the same hour and scenario
        df = df.groupby(["hour", "gen_type", "scenario_name"], as_index=False).mean(numeric_only=True)
        # Plot curtailment
        self.graph_matrix(
            df,
            value_column,
            ylabel=ylabel,
            col_specifier="scenario_name",
            row_specifier=None
        )
    def graph_matrix(self, df, value_column, ylabel, row_specifier, col_specifier):
        # Normalize specifiers
        df["empty_col"] = "-"
        if row_specifier is None:
            row_specifier = "empty_col"
        if col_specifier is None:
            col_specifier = "empty_col"

        # Drop rows with missing row/col specifiers (avoids empty groups)
        df = df.dropna(subset=[row_specifier, col_specifier])

        # If no data at all, bail out gracefully
        if df.empty:
            fig = self.get_figure(size=(10, 6), ylabel=ylabel, xlabel="Time of day (PST)")
            ax = fig.subplots(1, 1)
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            return

        # Unique rows and columns
        rows = df[row_specifier].drop_duplicates().sort_values()
        # Compute nrows/ncols with safe minimums
        nrows = max(1, min(len(rows), 6))

        ncols = 0
        for row in rows:
            cols_here = df.loc[df[row_specifier] == row, col_specifier].drop_duplicates()
            ncols = max(ncols, len(cols_here))
        ncols = max(1, min(ncols, 8))

        fig = self.get_figure(
            size=(10 * ncols / nrows, 8),
            # size=(min(10 * ncols / nrows, 12), 8),
            ylabel=ylabel,
            xlabel="Time of day (PST)"
        )
        ax = fig.subplots(nrows, ncols, sharey='row', sharex=False, squeeze=False)

        # Order technologies by smoothness
        df_all = df.pivot_table(
            index="hour",
            columns="gen_type",
            values=value_column,
            aggfunc="sum",
            observed=False
        )
        ordered_columns = df_all.std().sort_values().index

        legend = {}

        # Iterate cells
        for ri in range(nrows):
            # Some row indices may not exist if rows < nrows; guard
            if ri >= len(rows):
                continue
            row_val = rows.iloc[ri]
            df_row = df.loc[df[row_specifier] == row_val]
            columns = df_row[col_specifier].drop_duplicates().sort_values()

            # Pad grid if fewer columns than ncols
            for ci in range(ncols):
                current_ax = ax[ri][ci]
                if ci >= len(columns):
                    # blank cell
                    current_ax.axis("off")
                    continue

                col_val = columns.iloc[ci]
                # BUGFIX: filter within df_row, not entire df
                sub_df = df_row.loc[df_row[col_specifier] == col_val]

                if sub_df.empty:
                    current_ax.axis("off")
                    continue

                sub_df = sub_df.pivot(index="hour", columns="gen_type", values=value_column)
                # Ensure consistent order of techs
                sub_df = sub_df.reindex(columns=ordered_columns)

                # Split positive/negative for net stacks if needed
                colors = self.get_colors()
                sub_df = sub_df.rename_axis("Type", axis="columns")

                if colors is None:
                    pos = sub_df.clip(lower=0)
                    neg = sub_df.clip(upper=0)

                    pos.plot.area(
                        ax=current_ax, stacked=True,
                        xlabel=str(col_val), ylabel=str(row_val),
                        xticks=[], legend=False
                    )
                    # Plot negatives after to render below zero
                    if (neg.values < 0).any():
                        neg.plot.area(
                            ax=current_ax, stacked=True,
                            xlabel=str(col_val), ylabel=str(row_val),
                            xticks=[], legend=False
                        )
                else:
                    sub_df.plot.area(
                        ax=current_ax, stacked=True,
                        color=colors, xlabel=str(col_val),
                        ylabel=str(row_val), xticks=[], legend=False
                    )

                # Collect legend handles (stable order by ordered_columns)
                handles, labels = current_ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                for tech in ordered_columns:
                    if tech in by_label:
                        legend[tech] = by_label[tech]

        # Tighten layout
        fig.subplots_adjust(wspace=0)

        # Legend: follow ordered_columns for consistent ordering; show only ones present
        legend_labels = [t for t in ordered_columns if t in legend]
        legend_handles = [legend[t] for t in legend_labels]
        if legend_handles:
            fig.legend(legend_handles, legend_labels)

    # def graph_matrix(self, df, value_column, ylabel, row_specifier, col_specifier):
    #     # Change None values to a column which is all the same
    #     df["empty_col"] = "-"
    #     if row_specifier is None:
    #         row_specifier = "empty_col"
    #     if col_specifier is None:
    #         col_specifier = "empty_col"
    #     # Get rows
    #     rows = df[row_specifier].drop_duplicates().sort_values()
    #     # Count number of rows and number of columns
    #     nrows = min(len(rows), 6)
    #     ncols = 0
    #     for row in rows:
    #         columns = df[df[row_specifier] == row][col_specifier].drop_duplicates()
    #         ncols = max(ncols, len(columns))
    #     ncols = min(ncols, 8)
    #     fig = self.get_figure(
    #         size=(10 * ncols / nrows, 8),
    #         ylabel=ylabel,
    #         xlabel="Time of day (PST)"
    #     )

    #     ax = fig.subplots(nrows, ncols, sharey='row', sharex=False, squeeze=False)

    #     # Sort the technologies by standard deviation to have the smoothest ones at the bottom of the stacked area plot
    #     # df_all = df.pivot_table(index='hour', columns='gen_type', values=value_column, aggfunc=np.sum)
    #     df_all = df.pivot_table(
    #         index="hour",
    #         columns="gen_type",
    #         values=value_column,
    #         aggfunc="sum",      # replaces np.sum
    #         observed=False      # optional, to preserve current categorical behavior
    #     )
    #     ordered_columns = df_all.std().sort_values().index

    #     legend = {}

    #     # for each row...
    #     for ri in range(nrows):
    #         row = rows.iloc[ri]
    #         df_row = df[df[row_specifier] == row]
    #         columns = df_row[col_specifier].drop_duplicates().sort_values()
    #         for ci in range(min(ncols, len(columns))):
    #             column = columns.iloc[ci]
    #             current_ax = ax[ri][ci]
    #             # get the dispatch for that quarter
    #             sub_df = df_row.loc[df[col_specifier] == column]
    #             # Skip if no timepoints in quarter
    #             if len(sub_df) == 0:
    #                 continue
    #             # Make it into a proper dataframe
    #             sub_df = sub_df.pivot(index='hour', columns='gen_type', values=value_column)
    #             sub_df = sub_df.reindex(columns=ordered_columns)
    #             # # Fill hours with no data with zero so x-axis doesn't skip hours
    #             # all_hours = tools.np.arange(0, 24, 1)
    #             # missing_hours = all_hours[~tools.np.isin(all_hours, sub_df.index)]
    #             # sub_df = sub_df.append(tools.pd.DataFrame(index=missing_hours)).sort_index().fillna(0)
    #             # Get axes

    #             # Rename to make legend proper
    #             sub_df = sub_df.rename_axis("Type", axis='columns')
    #             # Plot
    #             colors = self.get_colors()
    #             if colors is None:
    #                 # sub_df.plot.area(ax=current_ax, stacked=True,
    #                 #                  xlabel=column,
    #                 #                  ylabel=row,
    #                 #                  xticks=[],
    #                 #                  legend=False)
    #                 pos = sub_df.clip(lower=0)   # keep only >=0
    #                 neg = sub_df.clip(upper=0)   # keep only <=0

    #                 # Stack the positives above 0
    #                 pos.plot.area(
    #                     ax=current_ax, stacked=True,
    #                     xlabel=column, ylabel=row, xticks=[], legend=False
    #                 )

    #                 # Stack the negatives below 0
    #                 neg.plot.area(
    #                     ax=current_ax, stacked=True,
    #                     xlabel=column, ylabel=row, xticks=[], legend=False
    #                 )
                    
    #             else:
    #                 sub_df.plot.area(ax=current_ax, stacked=True,
    #                                  color=colors,
    #                                  xlabel=column,
    #                                  ylabel=row,
    #                                  xticks=[],
    #                                  legend=False)
    #             # Get all the legend labels and add them to legend dictionary.
    #             # Since it's a dictionary, duplicates are dropped
    #             handles, labels = current_ax.get_legend_handles_labels()
    #             for i in range(len(handles)):
    #                 legend[labels[i]] = handles[i]
    #     # Remove space between subplot columns
    #     fig.subplots_adjust(wspace=0)
    #     # Add the legend
    #     legend_pairs = legend.items()
    #     fig.legend([h for _, h in legend_pairs], [l for l, _ in legend_pairs])

    @staticmethod
    def create_bin_labels(bins):
        """Returns an array of labels representing te bins."""
        i = 1
        labels = []
        while i < len(bins):
            low = bins[i-1]
            high = bins[i]
            if low == float("-inf"):
                labels.append(f"<{high}")
            elif high == float("inf"):
                labels.append(f"{low}+")
            else:
                labels.append(f"{low} - {high}")
            i += 1
        return labels

    @staticmethod
    def sort_build_years(x):
        def val(v):
            r = v if v != "Pre-existing" else "000"
            return r

        xm = x.map(val)
        return xm


def graph_scenarios(scenarios: List[Scenario], graph_dir, overwrite=False, module_names=None, figures=None, **kwargs):
    # If directory already exists, verify we should overwrite its contents
    if os.path.exists(graph_dir):
        # if not overwrite and not query_yes_no(
        #         f"Folder '{graph_dir}' already exists. Some graphs may be overwritten. Continue?"):
        #     return
        f"Folder '{graph_dir}' already exists. Some graphs may be overwritten. Continue?"
    # Otherwise create the directory
    else:
        os.mkdir(graph_dir)

    # Start a timer
    timer = StepTimer()

    # If no module name specified we get them from modules.txt
    if module_names is None:
        module_names = read_modules(scenarios)

    # Import the modules
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            warnings.warn(f"Failed to load {module_name}. Graphs in this module will not be created.")

    # Initialize the graphing tool
    graph_tools = GraphTools(scenarios=scenarios, graph_dir=graph_dir, **kwargs)

    # Loop through every graphing module
    print(f">>> Graphing modules <<<")
    if figures is None:
        for graph_func in registered_graphs.values():
            run_graph_func(graph_tools, graph_func)
    else:
        for figure in figures:
            try:
                func = registered_graphs[figure]
            except KeyError:
                raise Exception(f"{figures} not found in list of registered graphs. "
                                f"Make sure your graphing function is in a module.")
            run_graph_func(graph_tools, func)

    print(f"\nTook {timer.step_time()} to generate all graphs.")


def run_graph_func(tools, func):
    """Runs the graphing function"""
    print(f"{func.name}", end=", ", flush=True)
    tools.pre_graphing(func.multi_scenario, func.name, func.title, func.note)
    if func.multi_scenario:
        func(tools)
    else:
        # For each scenario
        for i, scenario in enumerate(tools.scenarios):
            # Set the active scenario index so that other functions behave properly
            tools._active_scenario = i
            # Call the graphing function
            func(tools)
        # Reset to 0 like it was before
        tools._active_scenario = 0

    tools.post_graphing()


def read_modules(scenarios):
    """Reads all the modules found in modules.txt"""
    # late import to minimize circular dependency
    import switch_model.solve
    # import switch_model.utilities
    def read_modules_txt(scenario):
        """Returns a sorted list of all the modules in a run folder (by reading modules.txt)"""
        with scenario:
            
            
            # module_list = get_module_list(include_solve_module=False)
            module_list = switch_model.solve.get_module_list([])
        return np.sort(module_list)

    # Split compare_dirs into a base and a list of others
    scenario_base, other_scenarios = scenarios[0], scenarios[1:]
    module_names = read_modules_txt(scenario_base)

    # Check that all the compare_dirs have equivalent modules.txt
    for scenario in other_scenarios:
        scenario_module_names = read_modules_txt(scenario)
        if not np.array_equal(module_names, scenario_module_names):
            warnings.warn(f"modules.txt is not equivalent between {scenario_base.name} (len={len(module_names)}) and "
                          f"{scenario.name} (len={len(scenario_module_names)}). "
                          f"We will use the modules.txt in {scenario_base.name} however this may result "
                          f"in missing graphs and/or errors.")

    return module_names
