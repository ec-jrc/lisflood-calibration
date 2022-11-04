#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from datetime import datetime

from liscal import config, subcatchment, objective, products


class PlotParameters():

    title_size_big = 32
    title_size_small = 18
    label_size = 30
    axes_size = 24
    legend_size_small = 16
    threshold_size = 24

    file_format = 'svg'

    text = {
        'figure': {'autolayout': True},
        'font': {
            'size': 14,
            'family':'sans-serif',
            'sans-serif':['Arial'],
            'weight': 'bold'
        },
        'text': {'usetex': True},
        'axes': {'labelweight': 'bold'},
    }


class ConfigPostProcessing(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')
        self.summary_path = self.parser.get('Path','summary_path')

        # Date parameters
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")
        self.timestep = int(self.parser.get('Main', 'timestep'))  # in minutes
        if self.timestep != 360 and self.timestep != 1440:
            raise Exception('Calibration timestep {} not supported'.format(self.timestep))

        # we don't use it but required for objectives object
        self.param_ranges = None

        # stations
        self.stations_data = self.parser.get('Stations', 'stations_data')

        # plot parameters
        self.plot_params = PlotParameters()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('validation_year', help='Year used for validation')
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
    
    if args.validation_year == 0:
        observations_file = 'observations.csv'
    else:
        observations_file = f'observations_{args.validation_year}.csv'

    if os.path.isfile(os.path.join(subcatch.path_station, observations_file)):
        obj = objective.ObjectiveKGE(cfg, subcatch, observations_file=observations_file)
        products.create_products(cfg, subcatch, obj)
    else:
        raise Exception('ERROR! Observations not available! Could not compute products')

    print("==================== END ====================")
