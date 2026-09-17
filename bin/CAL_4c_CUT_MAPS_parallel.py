#!/usr/bin/env python3
"""CAL_4c: Parallel version of CAL_4b_CUT_MAPS_list.

Cuts global maps to subcatchment extents for a list of stations,
processing multiple stations in parallel using ThreadPoolExecutor.
Internally calls CAL_4_CUT_MAPS for each station.
"""
import argparse
import pandas as pd
import os
import sys
import numpy as np
import concurrent.futures

from liscal import config


class ConfigCutMaps(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file, print_settings=False)

        # paths
        self.subcatchment_path = self.parser.get('Path', 'subcatchment_path')

        # stations
        self.stations_data = self.parser.get('Stations', 'stations_data')


def process_station(obsid, stations_meta, settings_file, path_maps, cfg, cal4_script_path):
    """Process a single station by calling CAL_4_CUT_MAPS via os.system.

    Skips stations that already have all maps cut (checks file existence).

    Parameters
    ----------
    obsid : int
        Station observation ID.
    stations_meta : pd.DataFrame
        Full stations metadata DataFrame indexed by ObsID.
    settings_file : str
        Path to calibration settings file.
    path_maps : str
        Path to global maps directory or single map file.
    cfg : ConfigCutMaps
        Configuration object with subcatchment_path.
    cal4_script_path : str
        Full path to the CAL_4_CUT_MAPS.py script.
    """
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError:
        raise Exception('Station {} not found in stations file'.format(obsid))

    subcatchment_path = os.path.join(cfg.subcatchment_path, str(obsid))
    path_subcatch_maps = os.path.join(subcatchment_path, 'maps')

    cmd = f"python {cal4_script_path} {settings_file} {path_maps} {obsid} --use-dask-config"
    at_least_one_file_to_process = False

    if os.path.isfile(path_maps) and os.path.getsize(path_maps) > 0:
        afile = os.path.basename(path_maps)
        fileout = os.path.join(path_subcatch_maps, afile)
        if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
            print(f"skipping already existing {fileout}")
        else:
            at_least_one_file_to_process = True
    else:
        # Enter in maps dir and walk through subfolders
        for root, dirs, files in os.walk(path_maps, topdown=False, followlinks=True):
            if at_least_one_file_to_process:
                break
            for afile in files:
                fileout = os.path.join(path_subcatch_maps, afile)
                if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                    print(f"skipping already existing {fileout}")
                    continue
                else:
                    filenc = os.path.join(root, afile)
                    if filenc.find("bak") > -1:
                        continue
                    at_least_one_file_to_process = True
                    break

    if at_least_one_file_to_process:
        print(f">> Calling \"{cmd}\"")
        os.system(cmd)


def main(settings_file, path_maps, station_list, max_workers=None):
    """Cut global maps for a list of stations in parallel.

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file.
    path_maps : str
        Path to global maps directory or single map file.
    station_list : str
        Path to a text file listing station IDs (one per line).
    max_workers : int or None, optional
        Maximum number of parallel workers. Default None (auto).
    """
    cfg = ConfigCutMaps(settings_file)

    CatchmentsToProcess = pd.read_csv(station_list, sep=",", header=None)
    Series = CatchmentsToProcess[0]
    Series = np.array(Series)
    print(Series)

    # Read full list of stations, index is obsid
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')
    stationdata_sorted = stations_meta.sort_values(by=['DrainingArea.km2.LDD'], ascending=True)

    # Determine path to CAL_4_CUT_MAPS.py (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cal4_script_path = os.path.join(script_dir, "CAL_4_CUT_MAPS.py")

    # Use ThreadPoolExecutor to run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, row in stationdata_sorted.iterrows():
            catchment = index
            if len(Series[Series == catchment]) == 0:
                continue

            obsid = int(index)
            print(obsid)

            futures.append(executor.submit(
                process_station, obsid, stations_meta, settings_file, path_maps, cfg, cal4_script_path
            ))

        concurrent.futures.wait(futures)

    print("==================== END ====================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('path_maps', help='Input global maps directory')
    parser.add_argument('station_list', help='List of Station OBSID to process')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of workers for parallel processing')
    args = parser.parse_args()

    main(args.settings_file, args.path_maps, args.station_list, args.max_workers)
