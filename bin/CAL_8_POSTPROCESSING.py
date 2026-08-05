#!/usr/bin/env python3
import os
import argparse

from liscal import subcatchment, objective, products
from liscal.config import ConfigPostProcessing


def main(settings_file, station):
    """Run post-processing for a calibrated station.

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file.
    station : str or int
        Station OBSID to process.
    """

    cfg = ConfigPostProcessing(settings_file)

    obsid = int(station)

    print("=================== "+str(obsid)+" ====================")
    
    subcatch = subcatchment.SubCatchment(
        cfg, obsid, initialise=False
        )
    if not os.path.exists(os.path.join(subcatch.path, "out", "streamflow_simulated_best.csv")):
        print('Cannot find file {}'.format(os.path.join(subcatch.path, "out", "streamflow_simulated_best.csv")))
        raise Exception('Calibration not complete! Cannot generate products...')

    obj = objective.create_objective(cfg, subcatch)

    products.create_products(cfg, subcatch, obj)

    print("==================== END ====================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    main(args.settings_file, args.station)
