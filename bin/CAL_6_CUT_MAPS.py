#!/usr/bin/env python3
import argparse
import pandas as pd

from liscal import config, cutmaps


class ConfigCutMaps(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # stations
        self.stations_data = self.parser.get('Stations', 'stations_data')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('path_maps', help='Input global maps directory')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = ConfigCutMaps(settings_file)

    # Read full list of stations, index is obsid
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Calibrate lisflood fo specified station
    obsid = int(args.station)
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    cutmaps.cut_maps_station(cfg, args.path_maps, station_data, obsid)
