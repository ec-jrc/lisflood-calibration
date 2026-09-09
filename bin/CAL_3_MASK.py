#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import pandas
import time
import shutil
from liscal.pcr_utils import pcrasterCommand
from configparser import ConfigParser


def main(settings_file, file_CatchmentsToProcess):
    """Create subcatchment mask maps and directory structure.

    For each station in the CatchmentsToProcess file, creates:
    - maps/mask.map and maps/masksmall.map (subcatchment masks)
    - maps/outlet.map and maps/outletsmall.map (station outlet)
    - inflow/inflow.map (inlet locations)

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file (needs [Path] subcatchment_path
        and [Stations] stations_data).
    file_CatchmentsToProcess : str
        Path to CSV file listing station IDs to process.
    """

    print("=================== START ===================")
    print(">> Reading settings file (" + settings_file + ")...")

    parser = ConfigParser()
    parser.read(settings_file)

    subcatchment_path = parser.get('Path', 'subcatchment_path')
    stations_data_path = parser.get("Stations", "stations_data")

    path_result = os.path.dirname(stations_data_path)
    path_gauges = os.path.join(path_result, "gauges.map")
    interstation_regions = os.path.join(path_result, "interstation_regions.map")
    inlets = os.path.join(path_result, "inlets.map")

    pcrcalc = "pcrcalc"
    resample = "resample"

    ########################################################################
    #   Make stationdata array from the Qmeta csv
    ########################################################################

    print(">> Reading stations_data file...")
    stationdata = pandas.read_csv(stations_data_path, sep=",", index_col='ObsID')
    stationdata_sorted = stationdata.sort_values(by=['DrainingArea.km2.LDD'], ascending=True)

    CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess, sep=",", header=None)

    for index, row in stationdata_sorted.iterrows():
        catchment = index
        Series = CatchmentsToProcess[0]
        if not (catchment in Series.values):
            continue
        print("\n\n\n=================== " + str(catchment) + " ====================")
        print(">> Starting map subsetting for catchment " + str(catchment) + ", size " + str(row['DrainingArea.km2.LDD']) + " km2...")

        t = time.time()

        path_subcatch = os.path.join(subcatchment_path, str(catchment))
        path_temp = path_subcatch

        if not os.path.exists(path_subcatch):
            os.makedirs(path_subcatch)
        if not os.path.exists(os.path.join(path_subcatch, 'maps')):
            os.makedirs(os.path.join(path_subcatch, 'maps'))
        if not os.path.exists(os.path.join(path_subcatch, 'inflow')):
            os.makedirs(os.path.join(path_subcatch, 'inflow'))
        if not os.path.exists(os.path.join(path_subcatch, 'out')):
            os.makedirs(os.path.join(path_subcatch, 'out'))

        # Make mask map for subcatchment
        subcatchmask_map = os.path.join(path_subcatch, "maps", "mask.map")
        pcrasterCommand(pcrcalc + " 'F0 = boolean(if(scalar(F1) eq " + str(index) + ",scalar(1)))'", {"F0": subcatchmask_map, "F1": interstation_regions})
        tmp1_map = os.path.join(path_temp, "tmp1.map")
        smallsubcatchmask_map = os.path.join(path_subcatch, "maps", "masksmall.map")
        pcrasterCommand(pcrcalc + " 'F0 = if(F1==1,F2)'", {"F0": tmp1_map, "F1": subcatchmask_map, "F2": subcatchmask_map})
        pcrasterCommand(resample + " -c 0 F0 F1", {"F0": tmp1_map, "F1": smallsubcatchmask_map})

        # Ensure that there is only one outlet pixel in outlet map
        station_map = path_gauges
        subcatchstation_map = os.path.join(path_subcatch, "maps", "outlet.map")
        pcrasterCommand(pcrcalc + " 'F0 = boolean(if(scalar(F1) eq " + str(index) + ",scalar(1)))'", {"F0": subcatchstation_map, "F1": station_map})
        subcatchstation_small_map = os.path.join(path_subcatch, "maps", "outletsmall.map")
        pcrasterCommand(resample + " F0 F1 --clone F2", {"F0": subcatchstation_map, "F1": subcatchstation_small_map, "F2": smallsubcatchmask_map})

        # Make inlet map
        subcatchinlets_map = os.path.join(path_subcatch, "inflow", "inflow.map")
        shutil.copyfile(inlets, subcatchinlets_map)
        pcrasterCommand(pcrcalc + " 'F0 = F1*scalar(F2)'", {"F0": subcatchinlets_map, "F1": subcatchinlets_map, "F2": subcatchmask_map})

        elapsed = time.time() - t
        print("   Time elapsed: " + "{0:.2f}".format(elapsed) + " s")

    print("==================== END ====================")


if __name__ == '__main__':
    main(os.path.normpath(sys.argv[1]), os.path.normpath(sys.argv[2]))
