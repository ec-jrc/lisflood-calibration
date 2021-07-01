# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import sys
import argparse
import numpy as np
import pandas
from configparser import ConfigParser # Python 3.8
import glob
import subprocess
import traceback

from liscal import templates, config, subcatchment, hydro_model


def longtermrun_subcatchment(cfg, obsid, station_data):

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data)
    if os.path.exists(os.path.join(subcatch.path, "streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
    
    if os.path.exists(os.path.join(subcatch.path,"pareto_front.csv"))==True:
        hydro_model.generate_outlet_streamflow(cfg, subcatch, lis_template)
    else:
        raise Exception('Could not find optimnal parameters for long term run')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    settings_file = args.settings_file
    stations_list = args.stations

    cfg = config.Config(settings_file)

    # Read full list of stations, index is obsid
    print(">> Reading Qmeta2.csv file...")
    stations_meta = pandas.read_csv(cfg.Qmeta_csv, sep=",", index_col=0)

    # Long term run for specified station
    obsid = args.station
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    longtermrun_subcatchment(cfg, obsid, station_data)

    print("==================== END ====================")
