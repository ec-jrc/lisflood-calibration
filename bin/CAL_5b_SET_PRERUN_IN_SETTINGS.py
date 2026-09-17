#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from configparser import ConfigParser


def main(settings_file, stations_csv, station):
    """Update the prerun_start date in the settings file from station metadata.

    Reads the prerun_start value for the given station from the stations CSV
    and writes it into the [Main] section of the settings file.

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file (will be modified in-place).
    stations_csv : str
        Path to input station CSV file (must have 'prerun_start' column).
    station : str or int
        Station ObsID to look up.

    Raises
    ------
    Exception
        If the station is not found in the CSV.
    FileNotFoundError
        If the settings file does not exist.
    """

    print("=================== START ===================")

    # Read stations data
    print('Reading input stations data {}'.format(stations_csv))
    stations_meta = pd.read_csv(stations_csv, sep=",", index_col='ObsID')
    print(stations_meta)

    obsid = int(station)
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    # take the prerun_start date from the station csv file
    prerun_start = station_data['prerun_start']

    parser = ConfigParser()
    if os.path.isfile(settings_file):
        parser.read(settings_file)
    else:
        raise FileNotFoundError('Incorrect path to setting file: {}'.format(settings_file))

    parser['Main']['prerun_start'] = prerun_start
    with open(settings_file, 'w') as configfile:
        parser.write(configfile)

    print("==================== END ====================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('stations_csv', help='Input station csv file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    main(args.settings_file, args.stations_csv, args.station)
