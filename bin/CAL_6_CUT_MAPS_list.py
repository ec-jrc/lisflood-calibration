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

    full_path_to_prog = sys.argv[0]
    prog_name = parser.prog
    new_prog_name = full_path_to_prog.replace(prog_name,"CAL_6_CUT_MAPS.py")

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
      
      path_maps = args.path_maps
      subcatchment_path = os.path.join(cfg.subcatchment_path, str(obsid))
      path_subcatch_maps = os.path.join(subcatchment_path,'maps')

      cmd="python " + new_prog_name + " "+settings_file+" "+path_maps+" "+str(obsid)
      atLeastOneFileToProcess=False
      if os.path.isfile(path_maps) and os.path.getsize(path_maps) > 0:
            afile = os.path.basename(path_maps)
            fileout = os.path.join(path_subcatch_maps, afile)
            if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                print("skipping already existing %s" % fileout)
            else:
                atLeastOneFileToProcess=True
      else:
            # Enter in maps dir and walk through subfolders
            for root, dirs, files in os.walk(path_maps, topdown=False, followlinks=True):
                if atLeastOneFileToProcess==True:
                    break
                for afile in files:
                    fileout = os.path.join(path_subcatch_maps, afile)
                    if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                        print("skipping already existing %s" % fileout)
                        continue
                    else:
                        filenc = os.path.join(root, afile)
                        if filenc.find("bak") > -1:
                            continue
                        atLeastOneFileToProcess=True
                        break
      if atLeastOneFileToProcess==True:          
            print(">> Calling \""+cmd+"\"")
            os.system(cmd)

      # cutmaps.cut_maps_station(cfg, args.path_maps, station_data, obsid)
