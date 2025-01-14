#!/usr/bin/env python3
import argparse
import pandas as pd

from liscal import config, stations


if __name__ == '__main__':

    print("=================== START ===================")
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('--no_check', help='Turn off check whether enough obs data is available', action='store_true')
    args = parser.parse_args()

    settings_file = args.settings_file
    check_obs = True
    if args.no_check:
        check_obs = False

    cfg = config.ConfigCalibration(settings_file)

    # Read full list of stations, index is obsid
    print(">> Reading stations_data file...")
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Calibrate lisflood fo specified station
    obsid = int(args.station)
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    stations.extract_station_data(cfg, obsid, station_data, check_obs)
    print("==================== END ====================")
