import shutil
import os
##import lisf1 ##################### pay attention to the conda env!!!
import numpy as np
import pandas
import subprocess
import sys

startfrom=int(sys.argv[1])
endat=int(sys.argv[2])

resultsstep2 = pandas.read_csv("/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/IDs_closest_noLakes_noRes_minKGE041.csv", sep=",", index_col=0)
targetcatchmentlist=np.array(resultsstep2['stationID'])[:]
clostesgeographycatchlist = np.array(resultsstep2['geography'])[:]

paramtransferdirectory = "/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/HEADCATCHMENTS_noLakes_noRes_minKGE041/"
interstationID_directories = np.load('/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/headcatchments_ALL_directories_nolakes_nores.npy')

#for tc  in np.arange(len(targetcatchmentlist)):

for tc  in np.arange(startfrom,endat):
   
   targetcatchmentID = int(targetcatchmentlist[tc])
   
   print('----------------------------------------------------------------------------------------------')
   print('----------------------------------------------------------------------------------------------')
   print('----------------------------------------------------------------------------------------------')

   print('TARGET CATCHMENT ID = ',  targetcatchmentID)
   print('SIMULATION WITH DEFAULT PARAMETERS') 
      
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
     
   settings_dir = paramtransferdirectory + str(targetcatchmentID) + "/settings_defaultparam/"
   output_dir = paramtransferdirectory + str(targetcatchmentID) + "/out_defaultparam/"
   
   source_station_dir_target = str(fullpathselectedT)[:] + "/station/"
   destination_station_dir_target = paramtransferdirectory + str(targetcatchmentID) + "/station/" 
  
   if os.path.exists(output_dir) == False:   
      os.makedirs(output_dir, exist_ok=False)
   
   if os.path.exists(settings_dir) == False:
      os.makedirs(settings_dir, exist_ok=False)
   shutil.copy("/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run.xml",settings_dir)
   shutil.copy("/BGFS/DISASTER/grimast/PARAM_TRANSFER_GloFASv5_headcatchments/OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run.xml",settings_dir)

   if os.path.exists(destination_station_dir_target) == False:
      shutil.copytree(source_station_dir_target, destination_station_dir_target)        

   print('***************************************************************************************************')
   print('Set up and compute the LISFLOOD prerun - DEFAULT PARAMETERS')   
      
   with open(os.path.join(settings_dir , 'OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run.xml'), "r") as f:  ############## CORRECT THIS@@@
           template_xml = f.read()    
   gaugecoord = pandas.read_csv(destination_station_dir_target+'/station_data.csv',sep=",", index_col=0)
   gaugex = gaugecoord.values[15] ############## fix this!!!!!!!!!!!
   gaugey = gaugecoord.values[16] ############## fix this!!!!!!!!!!!  
   template_xml = template_xml.replace("LISFLOOD_X",str(gaugex)[2:-2])
   template_xml = template_xml.replace("LISFLOOD_Y",str(gaugey)[2:-2])   
               
   template_xml = template_xml.replace("##PathRoot",str(fullpathselectedT)[:])   
   template_xml = template_xml.replace("##PathOut",paramtransferdirectory + str(targetcatchmentID) + "/out_defaultparam/")
   ##UpperZoneTimeConstant
   UpperZoneTimeConstantDonor = str(10.0)
   template_xml = template_xml.replace("##UpperZoneTimeConstant",UpperZoneTimeConstantDonor)
   ##LowerZoneTimeConstant
   LowerZoneTimeConstantDonor = str(100.0)
   template_xml = template_xml.replace("##LowerZoneTimeConstant",LowerZoneTimeConstantDonor)   
   ##GwPercValue
   GwPercValueDonor = str(0.8)
   template_xml = template_xml.replace("##GwPercValue",GwPercValueDonor) 
   ##GwLoss 
   GwLossDonor = str(0.0)
   template_xml = template_xml.replace("##GwLoss",GwLossDonor) 
   ##LZThreshold
   LZThresholdDonor = str(10.0)
   template_xml = template_xml.replace("##LZThreshold",LZThresholdDonor)  
   ##b_Xinanjiang  
   b_XinanjiangDonor = str(0.5)
   template_xml = template_xml.replace("##b_Xinanjiang",b_XinanjiangDonor)   
   ##PowerPrefFlow    
   PowerPrefFlowDonor = str(4.0)
   template_xml = template_xml.replace("##PowerPrefFlow",PowerPrefFlowDonor) 
   ##CalChanMan
   CalChanManDonor = str(1.0)
   template_xml = template_xml.replace("##CalChanMan",CalChanManDonor) 
   ##TransSub    
   TransSubDonor = str(0.0)
   template_xml = template_xml.replace("##TransSub",TransSubDonor)    
   ##CalChanMan3
   CalChanMan3Donor = str(1.0)
   template_xml = template_xml.replace("##ManMCT",CalChanMan3Donor)  
   ##SnowMeltCoef
   SnowMeltCoefDonor = str(4.0)
   template_xml = template_xml.replace("##SnowMeltCoef",SnowMeltCoefDonor)      
   ##LakeMultiplier
   template_xml = template_xml.replace("##LakeMultiplier",'1.0')            
                  
   with open(os.path.join(settings_dir , 'OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run_EDITED_defaultpar.xml'), "w") as f:
        f.write(template_xml)
        
   prerun_file = settings_dir + 'OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run_EDITED_defaultpar.xml'
   #lisf1.main(prerun_file)
   subprocess.run([sys.executable, "/BGFS/DISASTER/grimast/LFbranchfeatureinit/GitHub_development_27dec2025/lisflood-code/src/lisf1.py",prerun_file,'-v']) 
   
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
   template_xml = template_xml.replace("##PathOut",paramtransferdirectory + str(targetcatchmentID) + "/out_defaultparam/")
   ##UpperZoneTimeConstant
   UpperZoneTimeConstantDonor = str(10.0)
   template_xml = template_xml.replace("##UpperZoneTimeConstant",UpperZoneTimeConstantDonor)
   ##LowerZoneTimeConstant
   LowerZoneTimeConstantDonor = str(100.0)
   template_xml = template_xml.replace("##LowerZoneTimeConstant",LowerZoneTimeConstantDonor)   
   ##GwPercValue
   GwPercValueDonor = str(0.8)
   template_xml = template_xml.replace("##GwPercValue",GwPercValueDonor) 
   ##GwLoss 
   GwLossDonor = str(0.0)
   template_xml = template_xml.replace("##GwLoss",GwLossDonor) 
   ##LZThreshold
   LZThresholdDonor = str(10.0)
   template_xml = template_xml.replace("##LZThreshold",LZThresholdDonor)  
   ##b_Xinanjiang  
   b_XinanjiangDonor = str(0.5)
   template_xml = template_xml.replace("##b_Xinanjiang",b_XinanjiangDonor)   
   ##PowerPrefFlow    
   PowerPrefFlowDonor = str(4.0)
   template_xml = template_xml.replace("##PowerPrefFlow",PowerPrefFlowDonor) 
   ##CalChanMan
   CalChanManDonor = str(1.0)
   template_xml = template_xml.replace("##CalChanMan",CalChanManDonor) 
   ##TransSub    
   TransSubDonor = str(0.0)
   template_xml = template_xml.replace("##TransSub",TransSubDonor)    
   ##CalChanMan3
   CalChanMan3Donor = str(1.0)
   template_xml = template_xml.replace("##ManMCT",CalChanMan3Donor)  
   ##SnowMeltCoef
   SnowMeltCoefDonor = str(4.0)
   template_xml = template_xml.replace("##SnowMeltCoef",SnowMeltCoefDonor)      
   ##LakeMultiplier
   template_xml = template_xml.replace("##LakeMultiplier",'1.0')            
                  
   with open(os.path.join(settings_dir , 'OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run_EDITED_defaultpar.xml'), "w") as f:
        f.write(template_xml)
        
        
   run_file = settings_dir + 'OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run_EDITED_defaultpar.xml'
   #lisf1.main(run_file)  
   subprocess.run([sys.executable, "/BGFS/DISASTER/grimast/LFbranchfeatureinit/GitHub_development_27dec2025/lisflood-code/src/lisf1.py",run_file,'-v']) 
   