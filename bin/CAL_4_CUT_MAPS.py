#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

from liscal import config, cutmaps



class ConfigCutMaps(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file, print_settings=False)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # stations
        self.stations_data = self.parser.get('Stations', 'stations_data')


def main(settings_file, path_maps, station, use_dask_config=False):
    """Cut global maps to subcatchment extent.

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file.
    path_maps : str
        Path to input global maps directory or a single map file.
    station : str
        Single station ID (as string) or path to a station list file.
    use_dask_config : bool, optional
        Whether to use manual Dask configuration (default False).
    """

    cfg = ConfigCutMaps(settings_file)

    # Read full list of stations, index is obsid
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Try to convert the input to an integer
    obsid = None
    try:
        obsid = int(station)
    except ValueError:
        # If conversion fails, assume it's a file path and check if it exists
        if os.path.isfile(station):
            station_list = station
        else:
            print("Error: Input is neither a valid integer nor an existing file path.")
            raise ValueError(f"Input '{station}' is neither a valid integer nor an existing file path.")
    if obsid is not None:
        try:
            station_data = stations_meta.loc[obsid]
        except KeyError as e:
            raise Exception('Station {} not found in stations file'.format(obsid))
    else:
        # read the list of station to process in parallel
        try:
            CatchmentsToProcess = pd.read_csv(station_list, sep=",", header=None)
            obsid = CatchmentsToProcess[0]
            obsid = np.array(obsid)
        except:
            raise Exception('Error opening station txt: {} not found'.format(station_list))

    cutmaps.cut_maps_stations(cfg, path_maps, obsid, useDaskConfig=use_dask_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('path_maps', help='Input global maps directory')
    parser.add_argument('station', help='Specify a single station as an integer or a station list file (txt format)')
    parser.add_argument('--use-dask-config', action='store_true', help='Flag to use manual Dask configuration')
    args = parser.parse_args()

    main(args.settings_file, args.path_maps, args.station, args.use_dask_config)
