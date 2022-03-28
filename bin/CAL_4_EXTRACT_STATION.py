#!/usr/bin/env python3
import argparse
import pandas as pd
from datetime import datetime

from liscal import config, stations


class ConfigStation(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file)
        
        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # Date parameters
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")
        self.timestep = int(self.parser.get('Main', 'timestep'))  # in minutes
        
        # observations
        self.observed_discharges = self.parser.get('Stations', 'observed_discharges')
        self.stations_data = self.parser.get('Stations', 'stations_data')


if __name__ == '__main__':

    print("=================== START ===================")
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('timings', help='Whether we run in timing mode or not. Defaults to False', default=False)
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = ConfigStation(settings_file)

    # Read full list of stations, index is obsid
    print(">> Reading stations_data file...")
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Calibrate lisflood fo specified station
    obsid = int(args.station)
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    stations.extract_station_data(cfg, obsid, station_data, timings=args.timings)

    print("==================== END ====================")
