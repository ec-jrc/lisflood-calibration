#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

from liscal import config, stations

from liscal import templates, calibration, config, subcatchment, objective, hydro_model

def main(settings_file, station, no_check=False):
    """Extract station observation data for calibration.

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file.
    station : str
        Single station ID (as string) or path to a station list file.
    no_check : bool, optional
        If True, skip checking whether enough observation data is available.
    """

    print("=================== START ===================")
    check_obs = not no_check

    cfg = config.ConfigCalibration(settings_file)

    # Read full list of stations, index is obsid
    print(">> Reading stations_data file...")
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')

    # Try to convert the input to an integer
    obsids = None
    try:
        obsid = int(station)
        obsids = [obsid]
    except ValueError:
        # If conversion fails, assume it's a file path and check if it exists
        if os.path.isfile(station):
            station_list = station
        else:
            print("Error: Input is neither a valid integer nor an existing file path.")
            raise ValueError(f"Input '{station}' is neither a valid integer nor an existing file path.")
        # read the list of station to process 
        try:
            CatchmentsToProcess = pd.read_csv(station_list, sep=",", header=None)
            obsids = CatchmentsToProcess[0]
            obsids = np.array(obsids)
        except:
            raise Exception('Error opening station txt: {} not found'.format(station_list))

    # cycle for all ids
    for obsid in obsids:
        print(f"==================== processing station {obsid} ====================")
        try:
            station_data = stations_meta.loc[obsid]
        except KeyError as e:
            raise Exception('Station {} not found in stations file'.format(obsid))

        # first run of extraction_station_data, without checking the reservoir events
        stations.extract_station_data(cfg, None, obsid, station_data, check_obs)

        # if reservoir_events is None we can just skip the second execution of extract_station_data
        if cfg.reservoir_events is not None:
            # clear the cache for any previous used catchment in case of multi-catchments preprocessing
            hydro_model.Cache.clear()
            subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=station_data, create_links=False)
            lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
            lock_mgr = calibration.LockManager(cfg.num_cpus)
            obj = objective.create_objective(cfg, subcatch, read_observations=False)
            model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
            # load forcings and input maps in cache
            # required to find reservoir
            model.init_run()

            # second run of extraction_station_data, checking the reservoir events to filter observations
            stations.extract_station_data(cfg, model, obsid, station_data, check_obs)

    print("==================== END ====================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('--no_check', help='Turn off check whether enough obs data is available', action='store_true')
    args = parser.parse_args()

    main(args.settings_file, args.station, args.no_check)
