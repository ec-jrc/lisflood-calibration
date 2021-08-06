#!/bin/python3
import sys
import os
from os import path
import re
import argparse
import pandas
import numpy as np
from datetime import datetime, timedelta

from liscal import hydro_model, templates, config, subcatchment, utils


def deleteOutput(subcatch_dir):
    ret, res = utils.run_cmd("rm -f {}/settings*.xml".format(subcatch_dir))
    ret, res = utils.run_cmd("rm -rf {}/out".format(subcatch_dir))
    ret, res = utils.run_cmd("rm -rf {}/*.csv".format(subcatch_dir))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings', help='Settings file')
    parser.add_argument('stations_data', help='Stations metadata CSV file')
    parser.add_argument('obsid', help='Station obsid')
    args = parser.parse_args()

    print('  - obsid: {}'.format(args.obsid))
    print('  - settings file: {}'.format(args.settings))
    obsid = int(args.obsid)
    cfg = config.ConfigTiming(args.settings)

    print(">> Reading stations.csv file...")
    stations = pandas.read_csv(args.stations_data, sep=",", index_col=0)
    try:
        station_data = stations.loc[obsid]
    except KeyError as e:
        print(stations)
        raise Exception('Station {} not found in stations file'.format(obsid))

    # hack shorter period
    start_date = cfg.forcing_start.strftime('%d/%m/%Y %H:%M')
    end_date = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data)
    out_file = os.path.join(subcatch.path, 'out', 'timing_discharge.csv')
    if os.path.exists(out_file):
        deleteOutput(subcatch.path)
        os.makedirs(subcatch.path_out, exist_ok=True)

    # create object to create lisflood settings file
    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

    # first generate synthetic observations
    param_target = 0.5*np.ones(len(cfg.param_ranges))
    hydro_model.generate_timing(cfg, subcatch, lis_template, param_target, out_file, start_date, end_date)
