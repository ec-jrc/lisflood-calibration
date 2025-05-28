#!/usr/bin/env python3
import os
import argparse

from liscal import config, subcatchment, objective, products
from liscal.config import ConfigPostProcessing


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = ConfigPostProcessing(settings_file)

    # Calibrate lisflood fo specified station
    obsid = int(args.station)

    print("=================== "+str(obsid)+" ====================")
    
    subcatch = subcatchment.SubCatchment(
        cfg, obsid, initialise=False
        )
    if not os.path.exists(os.path.join(subcatch.path, "out", "streamflow_simulated_best.csv")):
        print('Cannot find file {}'.format(os.path.join(subcatch.path, "out", "streamflow_simulated_best.csv")))
        raise Exception('Calibration not complete! Cannot generate products...')

    obj = objective.ObjectiveKGE(cfg, subcatch)

    products.create_products(cfg, subcatch, obj)

    print("==================== END ====================")
