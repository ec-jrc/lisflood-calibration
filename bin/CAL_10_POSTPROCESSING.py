import os
import argparse
import pandas as pd
from liscal import config, subcatchment, objective, products


def plot_subcatchment(cfg, obsid, station_data):

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
        start_date=datetime.strptime(cfg.validation_start, '%d/%m/%Y %H:%M'),
        end_date=datetime.strptime(cfg.validation_end, '%d/%m/%Y %H:%M'),
        initialise=False
        )
    if os.path.exists(os.path.join(subcatch.path, "streamflow_simulated_best.csv")):
        raise Exception('Calibration not complete! Cannot generate products...')

    obj = objective.ObjectiveKGE(cfg, subcatch)

    products.create_products(cfg, subcatch, obj)

    print("==================== END ====================")
