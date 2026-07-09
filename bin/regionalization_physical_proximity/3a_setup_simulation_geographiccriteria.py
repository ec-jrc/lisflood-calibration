import shutil
import os
##import lisf1 ##################### pay attention to the conda env!!!
import numpy as np
import pandas
import subprocess
import sys

startfrom=int(sys.argv[1])
endat=int(sys.argv[2])

resultsstep2 = pandas.read_csv("/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/IDs_closest_noLakes_noRes_minKGE041_corrections.csv", sep=",", index_col=0)  ###
targetcatchmentlist=np.array(resultsstep2['stationID'])[:]
clostesgeographycatchlist = np.array(resultsstep2['geography'])[:]

paramtransferdirectory = "/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/HEADCATCHMENTS_noLakes_noRes_minKGE041/"
interstationID_directories = np.load('/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/headcatchments_ALL_directories_nolakes_nores.npy')

#for tc  in np.arange(len(targetcatchmentlist)):

for tc  in np.arange(startfrom,endat):
   
   targetcatchmentID = int(targetcatchmentlist[tc])
   clostesgeographycatchID = int(clostesgeographycatchlist[tc]) 
   
   print('----------------------------------------------------------------------------------------------')
   print('----------------------------------------------------------------------------------------------')
   print('----------------------------------------------------------------------------------------------')

   print('TARGET CATCHMENT ID = ',  targetcatchmentID)
   print('CLOSEST CATCHMENT ID, GEOGRAPHY = ',  clostesgeographycatchID) 
   
   
   indicesT=[]
   spT=[]
   indicesTarray=[] 
   indicesT = np.where(np.char.find(interstationID_directories,str(targetcatchmentID))!=-1)
   spT = np.char.split(interstationID_directories[indicesT],sep='/')
   indicesTarray=np.array(indicesT)
   count = -1
   for rowT in spT:
    count=count+1
    for elementT in rowT:
        if elementT == str(targetcatchmentID):
           selectedT = rowT
           fullpathselectedT = interstationID_directories[indicesTarray[0,count]]
           print(fullpathselectedT)
   continentT = selectedT[-3]
   basinT = selectedT[-2]

   basinT='OrinocoAtlanticNorthCoast' #################################################
   fullpathselectedT='/BGFS/DISASTER/russcar/cal_workflow_2025/catchments/SouthAmerica/OrinocoAtlanticNorthCoast/'+ str(targetcatchmentID)###############
   continentT = 'SouthAmerica'  ##########################

    
   indices=[]
   sp=[]
   indicesarray=[]
   indices = np.where(np.char.find(interstationID_directories,str(clostesgeographycatchID))!=-1)
   sp = np.char.split(interstationID_directories[indices],sep='/')
   indicesarray=np.array(indices)   
   count = -1
   for row in sp:
    count=count+1
    for element in row:
        if element == str(clostesgeographycatchID):
           selected = row
           fullpathselected = interstationID_directories[indicesarray[0,count]]
           print(fullpathselected)
           
   continent = selected[-3]
   basin = selected[-2]
   
   parameters_source_file = pandas.read_csv("/BGFS/DISASTER/grimast/calibration3arcmin_workflow/catchments/v5_allresultsfromLeonardo_23dec2025/"+continent[:]+'/'+basin[:]+'/'+str(clostesgeographycatchID)+'/pHistoryWRanks.csv')
   parameters_source_folder = "/BGFS/DISASTER/grimast/calibration3arcmin_workflow/catchments/v5_allresultsfromLeonardo_23dec2025/"+continent[:]+'/'+basin[:]+'/'+str(clostesgeographycatchID)
   
   
   settings_dir = paramtransferdirectory + str(targetcatchmentID) + "/settings_geography/"
   output_dir = paramtransferdirectory + str(targetcatchmentID) + "/out_geography/"
   
   source_dir_target_calibrated = "/BGFS/DISASTER/grimast/calibration3arcmin_workflow/catchments/v5_allresultsfromLeonardo_23dec2025/"+continentT[:]+'/'+basinT[:]+'/'+str(targetcatchmentID)+'/out' 
   destination_dir_target_calibrated = paramtransferdirectory + str(targetcatchmentID) + "/out_calibrated/"  
   
   source_station_dir_target = str(fullpathselectedT)[:] + "/station/"
   destination_station_dir_target = paramtransferdirectory + str(targetcatchmentID) + "/station/" 
 
   
   
   if os.path.exists(output_dir) == False:   
      os.makedirs(output_dir, exist_ok=False)
   
   if os.path.exists(settings_dir) == False:
      #shutil.copytree(source_dir, destination_dir)
      os.makedirs(settings_dir, exist_ok=False)
   shutil.copy("/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run.xml",settings_dir)
   shutil.copy("/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run.xml",settings_dir)

   if os.path.exists(destination_dir_target_calibrated) == False:
      shutil.copytree(source_dir_target_calibrated, destination_dir_target_calibrated)   ################## only TSS of discharge!!!!!!!!!!!!!!!!
   if os.path.exists(destination_station_dir_target) == False:
      shutil.copytree(source_station_dir_target, destination_station_dir_target)        

   print('***************************************************************************************************')
   print('Set up and compute the LISFLOOD prerun')   
      
   with open(os.path.join(settings_dir , 'OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run.xml'), "r") as f:  ############## CORRECT THIS@@@
           template_xml = f.read()    
   gaugecoord = pandas.read_csv(destination_station_dir_target+'/station_data.csv',sep=",", index_col=0)
   gaugex = gaugecoord.values[15] ############## fix this!!!!!!!!!!!
   gaugey = gaugecoord.values[16] ############## fix this!!!!!!!!!!!  
   template_xml = template_xml.replace("LISFLOOD_X",str(gaugex)[2:-2])
   template_xml = template_xml.replace("LISFLOOD_Y",str(gaugey)[2:-2])   
               
   template_xml = template_xml.replace("##PathRoot",str(fullpathselectedT)[:])   
   template_xml = template_xml.replace("##PathOut",paramtransferdirectory + str(targetcatchmentID) + "/out_geography/")
   ##UpperZoneTimeConstant
   UpperZoneTimeConstantDonor = str(parameters_source_file['UpperZoneTimeConstant'][0])
   template_xml = template_xml.replace("##UpperZoneTimeConstant",UpperZoneTimeConstantDonor)
   ##LowerZoneTimeConstant
   LowerZoneTimeConstantDonor = str(parameters_source_file['LowerZoneTimeConstant'][0])
   template_xml = template_xml.replace("##LowerZoneTimeConstant",LowerZoneTimeConstantDonor)   
   ##GwPercValue
   GwPercValueDonor = str(parameters_source_file['GwPercValue'][0])
   template_xml = template_xml.replace("##GwPercValue",GwPercValueDonor) 
   ##GwLoss 
   GwLossDonor = str(parameters_source_file['GwLoss'][0])
   template_xml = template_xml.replace("##GwLoss",GwLossDonor) 
   ##LZThreshold
   LZThresholdDonor = str(parameters_source_file['LZThreshold'][0])
   template_xml = template_xml.replace("##LZThreshold",LZThresholdDonor)  
   ##b_Xinanjiang  
   b_XinanjiangDonor = str(parameters_source_file['b_Xinanjiang'][0])
   template_xml = template_xml.replace("##b_Xinanjiang",b_XinanjiangDonor)   
   ##PowerPrefFlow    
   PowerPrefFlowDonor = str(parameters_source_file['PowerPrefFlow'][0])
   template_xml = template_xml.replace("##PowerPrefFlow",PowerPrefFlowDonor) 
   ##CalChanMan
   CalChanManDonor = str(parameters_source_file['CalChanMan1'][0])
   template_xml = template_xml.replace("##CalChanMan",CalChanManDonor) 
   ##TransSub    
   TransSubDonor = str(parameters_source_file['TransSub'][0])
   template_xml = template_xml.replace("##TransSub",TransSubDonor) 
   
   ##CalChanMan3
   print(parameters_source_file.columns)
   if 'CalChanMan3' in parameters_source_file.columns:
      print('write cal3 ', parameters_source_file['CalChanMan3'][0])
      CalChanMan3Donor = str(parameters_source_file['CalChanMan3'][0])
      template_xml = template_xml.replace("##ManMCT",CalChanMan3Donor)  
   else:
      template_xml = template_xml.replace("##ManMCT",'1.0')  
   ##SnowMeltCoef
   if 'SnowMeltCoef' in parameters_source_file.columns:
      SnowMeltCoefDonor = str(parameters_source_file['SnowMeltCoef'][0])
      template_xml = template_xml.replace("##SnowMeltCoef",SnowMeltCoefDonor)  
   else:
      template_xml = template_xml.replace("##SnowMeltCoef",'4.0')        
   ##LakeMultiplier
   if 'LakeMultiplier' in parameters_source_file.columns:
       LakeMultiplierDonor = str(parameters_source_file['LakeMultiplier'][0])
       template_xml = template_xml.replace("##LakeMultiplier",LakeMultiplierDonor)  
   else:
      template_xml = template_xml.replace("##LakeMultiplier",'1.0')            
                  
   with open(os.path.join(settings_dir , 'OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run_EDITED_geography.xml'), "w") as f:
        f.write(template_xml)
        
   prerun_file = settings_dir + 'OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run_EDITED_geography.xml'
   #lisf1.main(prerun_file)
   subprocess.run([sys.executable, "/BGFS/DISASTER/grimast/LFbranchfeatureinit/GitHub_development_27dec2025/lisflood-code/src/lisf1.py",prerun_file ]) 
   
   print('***************************************************************************************************')
   print('Set up and compute the LISFLOOD run')
   with open(os.path.join(settings_dir , 'OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run.xml'), "r") as f:  ############## CORRECT THIS@@@
           template_xml = f.read()    
   gaugecoord = pandas.read_csv(destination_station_dir_target+'/station_data.csv',sep=",", index_col=0)
   gaugex = gaugecoord.values[15] ############## fix this!!!!!!!!!!!
   gaugey = gaugecoord.values[16] ############## fix this!!!!!!!!!!!  
   template_xml = template_xml.replace("LISFLOOD_X",str(gaugex)[2:-2])
   template_xml = template_xml.replace("LISFLOOD_Y",str(gaugey)[2:-2])   
               
   template_xml = template_xml.replace("##PathRoot",str(fullpathselectedT)[:])   
   template_xml = template_xml.replace("##PathOut",paramtransferdirectory + str(targetcatchmentID) + "/out_geography/")
   ##UpperZoneTimeConstant
   UpperZoneTimeConstantDonor = str(parameters_source_file['UpperZoneTimeConstant'][0])
   template_xml = template_xml.replace("##UpperZoneTimeConstant",UpperZoneTimeConstantDonor)
   ##LowerZoneTimeConstant
   LowerZoneTimeConstantDonor = str(parameters_source_file['LowerZoneTimeConstant'][0])
   template_xml = template_xml.replace("##LowerZoneTimeConstant",LowerZoneTimeConstantDonor)   
   ##GwPercValue
   GwPercValueDonor = str(parameters_source_file['GwPercValue'][0])
   template_xml = template_xml.replace("##GwPercValue",GwPercValueDonor) 
   ##GwLoss 
   GwLossDonor = str(parameters_source_file['GwLoss'][0])
   template_xml = template_xml.replace("##GwLoss",GwLossDonor) 
   ##LZThreshold
   LZThresholdDonor = str(parameters_source_file['LZThreshold'][0])
   template_xml = template_xml.replace("##LZThreshold",LZThresholdDonor)  
   ##b_Xinanjiang  
   b_XinanjiangDonor = str(parameters_source_file['b_Xinanjiang'][0])
   template_xml = template_xml.replace("##b_Xinanjiang",b_XinanjiangDonor)   
   ##PowerPrefFlow    
   PowerPrefFlowDonor = str(parameters_source_file['PowerPrefFlow'][0])
   template_xml = template_xml.replace("##PowerPrefFlow",PowerPrefFlowDonor) 
   ##CalChanMan
   CalChanManDonor = str(parameters_source_file['CalChanMan1'][0])
   template_xml = template_xml.replace("##CalChanMan",CalChanManDonor) 
   ##TransSub    
   TransSubDonor = str(parameters_source_file['TransSub'][0])
   template_xml = template_xml.replace("##TransSub",TransSubDonor) 
   
   ##CalChanMan3
   if 'CalChanMan3' in parameters_source_file.columns:
      CalChanMan3Donor = str(parameters_source_file['CalChanMan3'][0])
      template_xml = template_xml.replace("##ManMCT",CalChanMan3Donor)  
   else:
      template_xml = template_xml.replace("##ManMCT",'1.0')  
   ##SnowMeltCoef
   if 'SnowMeltCoef' in parameters_source_file.columns:
      SnowMeltCoefDonor = str(parameters_source_file['SnowMeltCoef'][0])
      template_xml = template_xml.replace("##SnowMeltCoef",SnowMeltCoefDonor)  
   else:
      template_xml = template_xml.replace("##SnowMeltCoef",'4.0')        
   ##LakeMultiplier
   if 'LakeMultiplier' in parameters_source_file.columns:
       LakeMultiplierDonor = str(parameters_source_file['LakeMultiplier'][0])
       template_xml = template_xml.replace("##LakeMultiplier",LakeMultiplierDonor)  
   else:
      template_xml = template_xml.replace("##LakeMultiplier",'1.0')            
                  
   with open(os.path.join(settings_dir , 'OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run_EDITED_geography.xml'), "w") as f:
        f.write(template_xml)
        
        
   run_file = settings_dir + 'OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run_EDITED_geography.xml'
   #lisf1.main(run_file)  
   subprocess.run([sys.executable, "/BGFS/DISASTER/grimast/LFbranchfeatureinit/GitHub_development_27dec2025/lisflood-code/src/lisf1.py",run_file ]) 
   