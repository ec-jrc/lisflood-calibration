#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from liscal import config, subcatchment, objective, products


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = config.ConfigPostProcessing(settings_file)

    # Read full list of stations, index is obsid
    print(">> Reading stations_data file...")
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Calibrate lisflood fo specified station
    obsid = int(args.station)
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    print("=================== "+str(obsid)+" ====================")
    
    subcatch = subcatchment.SubCatchment(
        cfg, obsid, station_data, 
        start_date=cfg.validation_start,
        end_date=cfg.validation_end,
        initialise=False
        )
    if os.path.exists(os.path.join(subcatch.path, "streamflow_simulated_best.csv")):
        raise Exception('Calibration not complete! Cannot generate products...')

    obj = objective.ObjectiveKGE(cfg, subcatch)

    products.create_products(cfg, subcatch, obj)

    print("==================== END ====================")
