#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import sys
import numpy as np
import concurrent.futures

from liscal import config

file_CatchmentsToProcess = os.path.normpath(sys.argv[3])
print(file_CatchmentsToProcess)

class ConfigCutMaps(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file, print_settings=False)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # stations
        self.stations_data = self.parser.get('Stations', 'stations_data')


def process_station(obsid, stations_meta, settings_file, path_maps, cfg, new_prog_name):
    try:
        station_data = stations_meta.loc[obsid]
    except KeyError:
        raise Exception('Station {} not found in stations file'.format(obsid))

    subcatchment_path = os.path.join(cfg.subcatchment_path, str(obsid))
    path_subcatch_maps = os.path.join(subcatchment_path, 'maps')

    cmd = f"python {new_prog_name} {settings_file} {path_maps} {obsid} --use-dask-config"
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('path_maps', help='Input global maps directory')
    parser.add_argument('station_list', help='List of Station OBSID to process')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of workers for parallel processing')
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = ConfigCutMaps(settings_file)

    CatchmentsToProcess = pd.read_csv(file_CatchmentsToProcess, sep=",", header=None)
    Series = CatchmentsToProcess[0]
    Series = np.array(Series)
    print(Series)

    # Read full list of stations, index is obsid
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')
    stationdata_sorted = stations_meta.sort_values(by=['DrainingArea.km2.LDD'], ascending=True)

    full_path_to_prog = sys.argv[0]
    prog_name = parser.prog
    new_prog_name = full_path_to_prog.replace(prog_name, "CAL_4_CUT_MAPS.py")

    # Use ThreadPoolExecutor to run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for index, row in stationdata_sorted.iterrows():
            catchment = index
            Series = CatchmentsToProcess[0]
            if len(Series[Series == catchment]) == 0:  # Only process catchments whose ID is in the CatchmentsToProcess.txt file
                continue

            obsid = int(index)
            print(obsid)

            # Submit the task to the executor
            futures.append(executor.submit(process_station, obsid, stations_meta, settings_file, args.path_maps, cfg, new_prog_name))

        # Optionally, wait for all futures to complete
        concurrent.futures.wait(futures)
