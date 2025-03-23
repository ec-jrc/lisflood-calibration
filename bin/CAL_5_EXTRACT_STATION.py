#!/usr/bin/env python3
import argparse
import pandas as pd

from liscal import config, stations, reservoirs


if __name__ == '__main__':

    print("=================== START ===================")
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('--no_check', help='Turn off check whether enough obs data is available', action='store_true')
    parser.add_argument('--with_reservoirs', help='activate reservoirs check to compute the calibration date period', action='store_true')
    args = parser.parse_args()

    settings_file = args.settings_file
    check_obs = True
    if args.no_check:
        check_obs = False
    with_reservoirs = False
    if args.with_reservoirs:
        with_reservoirs = True

    cfg = config.ConfigCalibration(settings_file)

    # Read full list of stations, index is obsid
    print(">> Reading stations_data file...")
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Calibrate lisflood fo specified station
    obsid = int(args.station)
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError:
        raise Exception('Station {} not found in stations file'.format(obsid))

    # first run of extraction_station_data, without checking the reservoir events
    stations.extract_station_data(cfg, obsid, station_data, check_obs)

    # if we have reservoirs activated, check if the station is affected by any reservoir
    if with_reservoirs:
        model = reservoirs.initialise_model(cfg, station_data, obsid)

        # second run of exctraction_station_data, checking the reservoir events to filter observations
        stations.extract_station_data(cfg, model, obsid, station_data, check_obs, model)

    print("==================== END ====================")
