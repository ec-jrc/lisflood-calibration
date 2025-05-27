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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('path_maps', help='Input global maps directory')
    parser.add_argument('station', help='Specify a single station as an integer or a station list file (txt format)')
    parser.add_argument('--use-dask-config', action='store_true', help='Flag to use manual Dask configuration')
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = ConfigCutMaps(settings_file)

    # Read full list of stations, index is obsid
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Calibrate lisflood fo specified station
    # Try to convert the input to an integer
    obsid = None
    try:
        obsid = int(args.station)
    except ValueError:
        # If conversion fails, assume it's a file path and check if it exists
        if os.path.isfile(args.station):
            station_list = args.station
        else:
            print("Error: Input is neither a valid integer nor an existing file path.")
            exit(1)
    if obsid is not None:
        try:
            station_data = stations_meta.loc[obsid]
        except KeyError as e:
            raise Exception('Station {} not found in stations file'.format(obsid))
    else:
        # read the list of station to process in parallel
        try:
            CatchmentsToProcess = pd.read_csv(station_list,sep=",",header=None)
            obsid = CatchmentsToProcess[0]
            obsid = np.array(obsid)
        except:
            raise Exception('Error opening station txt: {} not found'.format(station_list))

    cutmaps.cut_maps_stations(cfg, args.path_maps, obsid, useDaskConfig=args.use_dask_config)
