#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import sys
import numpy as np

from liscal import config, cutmaps

file_CatchmentsToProcess = os.path.normpath(sys.argv[3])
print(file_CatchmentsToProcess)

class ConfigCutMaps(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file, print_settings=False)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # stations
        self.stations_data = self.parser.get('Stations', 'stations_data')
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('path_maps', help='Input global maps directory')
    #parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('station_list', help='List of Station OBSID to process')
    args = parser.parse_args()

    settings_file = args.settings_file
    
    ###print(args.settings_file)

    cfg = ConfigCutMaps(settings_file)
    
    CatchmentsToProcess = pd.read_csv(file_CatchmentsToProcess,sep=",",header=None)
    Series = CatchmentsToProcess[0]
    Series=np.array(Series)
    print(Series)
    # Read full list of stations, index is obsid
    stations_meta = pd.read_csv(cfg.stations_data, sep=",", index_col='ObsID')
    stationdata_sorted = stations_meta.sort_values(by=['DrainingArea.km2.LDD'],ascending=True)


    for index, row in stationdata_sorted.iterrows():
      catchment = index
      Series = CatchmentsToProcess[0]
      if len(Series[Series==catchment]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
          continue

      #obsid = int(args.station)
      obsid = int(index)
      print(obsid)
      try:
         station_data = stations_meta.loc[obsid]
      except KeyError as e:
         raise Exception('Station {} not found in stations file'.format(obsid))
      
      cutmaps.cut_maps_station(cfg, args.path_maps, station_data, obsid)
