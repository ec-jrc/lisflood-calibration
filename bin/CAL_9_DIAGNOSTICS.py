#!/usr/bin/env python3
import pandas as pd
import sys
import os
from liscal.diagnostic_plots import construct_dfs, plot_groupings, discharge_plot, other_var_plots
from configparser import ConfigParser as Parser


def main(settings_file, catchments_to_process_file, savepath=""):
    """Generate diagnostic plots for calibrated catchments.

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file (INI format with [Path] section).
    catchments_to_process_file : str
        Path to CSV file listing catchment IDs to process.
    savepath : str, optional
        Directory to save output plots. Defaults to "".
    """
    parser = Parser()
    settings_file = os.path.normpath(settings_file)
    catchments_to_process_file = os.path.normpath(catchments_to_process_file)

    parser.read(settings_file)
    base_path = parser.get('Path', 'subcatchment_path')

    catchments_to_process_df = pd.read_csv(catchments_to_process_file, header=None)
    catchment_ids = catchments_to_process_df[0]

    for catchment_id in catchment_ids:
        obs_df, sim_df, stn_df, _ = construct_dfs(base_path, catchment_id, plot_groupings)
        dis_fig = discharge_plot(obs_df, sim_df, stn_df, catchment_id, savepath)
        other_figs = other_var_plots(sim_df, stn_df, plot_groupings, catchment_id, savepath)


if __name__ == "__main__":
    settings_file = os.path.normpath(sys.argv[1])
    catchments_to_process_file = os.path.normpath(sys.argv[2])
    savepath = sys.argv[3] if len(sys.argv) > 3 else ""

    main(settings_file, catchments_to_process_file, savepath)