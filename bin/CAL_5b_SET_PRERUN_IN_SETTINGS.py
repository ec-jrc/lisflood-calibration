#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import datetime
import numpy as np
from configparser import ConfigParser


from liscal.pcr_utils import pcrasterCommand, getPCrasterPath

if __name__=="__main__":
    print("=================== START ===================")
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('stations_csv', help='Input station csv file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()
    settings_file = args.settings_file

    # Read stations data and selection stations of interest
    print('Reading input stations data {}'.format(args.stations_csv))
    stations_meta = pd.read_csv(args.stations_csv, sep=",", index_col='ObsID')
    print(stations_meta)

    obsid = int(args.station)
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    # take the prerun_start date from the station csv file
    # and update PRERUN_STARTDATE field
    # *** For the 41y prerun, uncomment these lines
    # prerun_start = 02/01/1979 00:00
    # *** For the 20y prerun, uncomment these lines
    # prerun_start = 01/01/2000 00:00
    prerun_start = station_data['prerun_start']

    parser = ConfigParser()
    if os.path.isfile(settings_file):
        parser.read(settings_file)
    else:
        raise FileNotFoundError('Incorrect path to setting file: {}'.format(settings_file))

    parser['Main']['prerun_start'] = prerun_start  
    with open(settings_file, 'w') as configfile:    # save
        parser.write(configfile)
