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
    parser.add_argument('obsid', help='Station obsid')
    parser.add_argument('settings', help='Settings file')
    args = parser.parse_args()

    print('  - obsid: {}'.format(args.obsid))
    print('  - settings file: {}'.format(args.settings))
    obsid = int(args.obsid)
    cfg = config.Config(args.settings)

    print(">> Reading stations.csv file...")
    stations = pandas.read_csv(cfg.stations_data, sep=",", index_col=0)
    try:
        station_data = stations.loc[obsid]
    except KeyError as e:
        print(stations)
        raise Exception('Station {} not found in stations file'.format(obsid))

    # hack shorter period
    assert station_data.loc['Cal_Start'] == cfg.forcing_start.strftime('%d/%m/%Y %H:%M')
    assert station_data.loc['Cal_End'] == cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data)
    out_file = os.path.join(subcatch.path, 'out', 'convergenceTester.csv')
    if os.path.exists(out_file):
        deleteOutput(subcatch.path)
        os.makedirs(subcatch.path_out, exist_ok=True)

    # create object to create lisflood settings file
    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

    # first generate synthetic observations
    param_target = 0.5*np.ones(len(cfg.param_ranges))
    hydro_model.generate_benchmark(cfg, subcatch, lis_template, param_target, out_file)
