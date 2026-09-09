# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import pdb
import pandas
import string
import datetime
import time
import numpy as np
import re
import argparse
import array
import logging
import random

from liscal.pcr_utils import pcrasterCommand, getPCrasterPath
from liscal import config, stations

output_folder = os.path.normpath(sys.argv[2])

class ConfigSpread(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file)
        
        # Date parameters
        self.path_temp = self.parser.get('Path', 'Temp')
        self.path_result = self.parser.get('Path', 'Result')
        self.No_of_calibration_PCs = float(self.parser.get('DEFAULT','No_of_calibration_PCs'))
        self.Qtss_csv = self.parser.get('CSV', 'Qtss')
        self.Qgis_csv = self.parser.get('CSV', 'Qgis')


if __name__ == '__main__':

    print("=================== START ===================")
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('output_folder', help='output folder: catchments to process list')
    args = parser.parse_args()
  
    settings_file = args.settings_file
  
    cfg = ConfigSpread(settings_file)
    print(">> Reading Qgis2.csv file...")
    stationdata = pandas.read_csv(os.path.join(cfg.path_result,"Qgis3.csv"),sep=",",index_col=0)
    print(stationdata)
    stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=False) 
    print(stationdata_sorted)	
    stationdata['SPREAD_WORKLOAD_ID'] = np.nan
    tmp_map = os.path.join(cfg.path_temp,"tmp.map")
    tmp2_map = os.path.join(cfg.path_temp,"tmp2.map") 
    tmp_txt = os.path.join(cfg.path_temp,"tmp.txt")
    tmp2_txt = os.path.join(cfg.path_temp,"tmp2.txt")
    station_map = os.path.join(cfg.path_result,"gauges.map")
    SPREAD_WORKLOAD_ID_map = os.path.join(cfg.path_result,"SPREAD_WORKLOAD_ID.map")    
    pcrcalc = "pcrcalc"
    col2map = "col2map"
    map2col = "map2col"
    resample = "resample"    
    pcrasterCommand(pcrcalc + " 'F0 = scalar(cover(F1,0.0))*0.0'", {"F0": SPREAD_WORKLOAD_ID_map, "F1":station_map})
  	 	
  	# Loop through catchments, biggest catchment first, and give nested 
  	# catchments that don't have an ID yet and ID
  	# Additionally, make map of IDs, and add IDs to stationdata dataframe
    SPREAD_WORKLOAD_ID = 1
    for index, row in stationdata_sorted.iterrows():
      if ~np.isnan(stationdata.loc[index,'SPREAD_WORKLOAD_ID']):
         continue		
      catchment_map = os.path.join(cfg.path_temp,"catchmask%05d.map" % float(index))					
      pcrasterCommand(pcrcalc + " 'F0 = if(defined(F1),F2)'", {"F0": tmp_map, "F1":catchment_map, "F2":station_map})
      pcrasterCommand(map2col + " F0 F1"  , {"F0":tmp_map, "F1":tmp2_txt})
      f = open(tmp2_txt,"r")
      for line in f.readlines():
        (X,Y,value) = line.split()
        stationdata.loc[int(value),'SPREAD_WORKLOAD_ID'] = SPREAD_WORKLOAD_ID			
        print ("catchment "+str(value)+" gets a SPREAD_WORKLOAD_ID of "+str(SPREAD_WORKLOAD_ID))
      f.close()
      pcrasterCommand(pcrcalc + " 'F0 = F0+cover(scalar(F1),0)*"+str(SPREAD_WORKLOAD_ID)+"'", {"F0": SPREAD_WORKLOAD_ID_map, "F1":catchment_map})		
      SPREAD_WORKLOAD_ID = SPREAD_WORKLOAD_ID+1		
  
   	# Make dataframe with for each ID the total area in number of pixels
    df = pandas.DataFrame({'SPREAD_WORKLOAD_ID' : np.arange(1,np.max(stationdata['SPREAD_WORKLOAD_ID'])+1)})
    df = df.set_index('SPREAD_WORKLOAD_ID')
    df['Area'] = np.nan
    for index, row in df.iterrows():	
      area = np.sum(stationdata.loc[stationdata['SPREAD_WORKLOAD_ID']==index,'CatchmentArea'])
      df.loc[index,'Area'] = area
   	
   	# Assign IDs to different computers (PCs)
   	# In stationdata dataframe, list the PC for each catchment
   	# Also make map of PCs
    SPREAD_WORKLOAD_PC_map = os.path.join(cfg.path_result,"SPREAD_WORKLOAD_PC.map")
    pcrasterCommand(pcrcalc + " 'F0 = scalar(cover(F1,0.0))*0.0'", {"F0": SPREAD_WORKLOAD_PC_map, "F1":station_map})
    totalarea = np.sum(stationdata.loc[:,'CatchmentArea'])
    area_per_pc = totalarea/cfg.No_of_calibration_PCs
    df['PC'] = np.nan
    for PC in np.arange(1,float(cfg.No_of_calibration_PCs)+1):
      area = 0
      for index, row in df.iterrows():
        if np.isnan(df.loc[index,'PC']):
          if area<area_per_pc:
            df.loc[index,'PC'] = PC
            stationdata.loc[stationdata['SPREAD_WORKLOAD_ID']==index,'PC'] = PC
            area = area+df.loc[index,'Area']
            pcrasterCommand(pcrcalc + " 'F0 = F0+cover(scalar(F1=="+str(index)+"),0)*"+str(PC)+"'", {"F0": SPREAD_WORKLOAD_PC_map, "F1":SPREAD_WORKLOAD_ID_map})
   	
   	# For each PC, make file listing the catchments
    for PC in np.arange(1,float(cfg.No_of_calibration_PCs)+1):
      df_sub = df[df['PC']==PC]
      if len(df_sub)>0:			
        CatchmentsToProcess = []
        for index, row in df_sub.iterrows():			
           CatchmentsToProcess = CatchmentsToProcess+stationdata.loc[stationdata['SPREAD_WORKLOAD_ID']==index,'ObsID'].tolist()
        PC_str = "0"+str(int(PC))
        PC_str = PC_str[-2:]
        print ("Making CatchmentsToProcess_"+PC_str+".txt file")
        CatchmentsToProcess = pandas.DataFrame(CatchmentsToProcess)
        CatchmentsToProcess.to_csv(output_folder+"/CatchmentsToProcess_"+PC_str+".txt",index=False,header=False)
    CatchmentsToProcess = stationdata['ObsID'].tolist()
    CatchmentsToProcess = pandas.DataFrame(CatchmentsToProcess)
    CatchmentsToProcess.to_csv(output_folder+"/CatchmentsToProcess_All.txt",index=False,header=False)
   	
   	# Write stationdata dataframe to Qgis3.csv in results directory
    stationdata.to_csv(os.path.join(cfg.path_result,"Qgis3.csv"),',')
    print ("==================== END ====================")
