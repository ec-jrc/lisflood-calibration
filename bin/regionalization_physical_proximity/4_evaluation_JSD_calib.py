import numpy as np
import os
import sys
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import pandas
from pandas import ExcelFile
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd
import csv
import pandas
from pandas import ExcelFile
from datetime import datetime
from liscal import hydro_stats

    
    
ctachmentslistfile=pandas.read_csv("<PATH>/PARAM_TRANSFER_GloFASv5_headcatchments/experiment_results_HEADCATCHMENTS_noLakes_noRes_minKGE041_geography_geographyclimate_JDKGE_calib.csv", sep=",", index_col=0) 

print(ctachmentslistfile)
catchments= np.array(ctachmentslistfile['ID'])[:]


foldername = "<PATH>/PARAM_TRANSFER_GloFASv5_headcatchments/HEADCATCHMENTS_noLakes_noRes_minKGE041/"


start_ev=datetime(1975,1,2)
end_ev=datetime(2024,1,1)


dfwrite = pd.read_csv("experiment_results_HEADCATCHMENTS_noLakes_noRes_minKGE041_geography_geographyclimate_JDKGE_calib.csv")


for index in np.arange(len(catchments)):

  stationID=str(catchments[index])
  
  for simulation in ['calibrated']:
  
    if simulation == 'calibrated':
       col = 'red'
    if simulation == 'defaultparam':
       col = 'blue'
    if simulation == 'geography':
       col = 'green'   
    if simulation == 'geographyclimate':
       col = 'cyan'                   
    out_folder = '/out_' + simulation
     
    path_subcatch = os.path.join(foldername,stationID+out_folder+'/theta3totallong_term_run.tss')
    path_subcatchBEST = os.path.join(foldername,stationID+out_folder+'/streamflow_simulated_best.tss')
    
   
    if os.path.exists(path_subcatch) or os.path.exists(path_subcatchBEST):     
     if os.path.exists(path_subcatch): 
        REFdistssFile=foldername+stationID+out_folder+ '/dislong_term_run.tss'
     else:
        REFdistssFile=foldername+stationID+out_folder+ '/streamflow_simulated_best.tss'
        
     print(REFdistssFile)
     c=0
     AA=[]
     aa=-1
     streamflow_data = pandas.read_csv(REFdistssFile, index_col=0, sep=",", skiprows=3, header=None, skipinitialspace=True, engine='python')
     dis=np.zeros((len(streamflow_data.index)-1,1))
     index_gauge=[]
     index_time=[]
     #print(len(streamflow_data.index))
     for ii in np.arange(len(streamflow_data.index)):
         AA=[]
         CC=[]
         AA=streamflow_data.index[ii]
         CC=AA.split()
         if len(CC)==1:
            index_gauge.append(CC)
         else:
            index_time.append(CC[0])
            aa=aa+1
            for gg in np.arange(len(index_gauge)): 
                dis[aa,gg]=CC[gg+1]         
   
     
     M = pandas.read_csv(foldername+stationID+'/station/observations_original.csv', sep=",", skiprows=0, skipinitialspace=True, engine='python')
     meas=np.zeros((17897,1)) + np.nan
     
     if 'Day' in M.columns:
         time=M['Day'] 
     if 'day' in M.columns:
         time=M['day'] 
     obs=M[stationID]
     for t in np.arange(len(time)):       
         date_string=time[t]
         a=datetime.strptime( date_string, '%d/%m/%Y %H:%M')
         b=(a-start_ev).days
         meas[b]=obs[t]
     
     kge_components = hydro_stats.fKGE(s=dis, o=meas)
     jdkge_components = hydro_stats.fKGE_JSD(s=dis, o=meas, dt=24)  ### HARD CODED for DAILY!!! ATTENTION!!
     
   
     fig1,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,tight_layout=True, figsize=(20,12))      
     ax1.plot(meas[(365*5+1):(365*16)],color='black', linewidth=1.2 ,label='obs')
     ax1.plot(dis[(365*5+1):(365*16)],linewidth=1.0,color= col, label = simulation)    
     ax2.plot(meas[(365*16+1):(365*27)],color='black', linewidth=1.2 ,label='obs')
     ax2.plot(dis[(365*16+1):(365*27)],linewidth=1.0,color= col , label = simulation) 
     ax3.plot(meas[(365*27+1):(365*38)],color='black', linewidth=1.2 ,label='obs')
     ax3.plot(dis[(365*27+1):(365*38)],linewidth=1.0,color= col, label = simulation) 
     ax4.plot(meas[(365*38+1):(365*49)],color='black', linewidth=1.2 ,label='obs')
     ax4.plot(dis[(365*38+1):(365*49)],linewidth=1.0,color=col, label = simulation) 
     ax1.legend()  
     titleplot = stationID +' - '+simulation +' - KGE= '+ str(round(kge_components[0],3)) +' - corr= '+ str(round(kge_components[1],3))
     ax1.set_title(titleplot)
     savefig1 = 'HEADCATCHMENTS_noLakes_noRes_minKGE041_plots/' + stationID +'_' +simulation+'.png'
     ##plt.show()
     fig1.savefig(savefig1)
     plt.close(fig1)
     
     
     
     KGE41ycal=kge_components[0]
     bias41ycal=kge_components[2]
     corr41ycal=kge_components[1]
     spread41ycal=kge_components[3]
     sae41ycal=kge_components[4]
     
     JDKGE=jdkge_components[0]
     JSD=jdkge_components[5]
     
     
     '''
     split_date_file=foldernamedata+stationID+'/station/station_data.csv'
     stationdata = pandas.read_csv(split_date_file, sep=",", index_col=0)
     data=stationdata[stationID]
     spdt=stationdata.index[-2]
     print(data[spdt])
     r=stationdata.index[10]
     c=stationdata.index[2]
     river=data[r]
     countryname=data[c]cd ..
     #print(river)
     #print(countryname)
     ar=stationdata.index[14]
     areaLDD=str(data[ar])
     LISFLOODranksFile=foldernamedata+stationID+'/pHistoryWRanks.csv' 
     model_eval = pandas.read_csv(LISFLOODranksFile, sep=",", index_col=0)
     KGE1=model_eval['Kling Gupta Efficiency']
     KGE=np.array(KGE1)
     print(KGE[0])
     corr1=model_eval['Correlation']
     corr=np.array(corr1)
     print(corr[0])
     bias1=model_eval['Signal ratio (s/o) (Bias)']
     spread1=model_eval['Noise ratio (s/o) (Spread)']
     bias=np.array(bias1)
     print(bias[0])
     spread=np.array(spread1)
     print(spread[0])
     sae=model_eval['sae']
     sae=np.array(sae)
     print(sae[0])
     
     start_calib=datetime.strptime( data[spdt], '%d/%m/%Y %H:%M')
     meas_calib=np.zeros((14975,1)) + np.nan
     time=M['Timestamp']
     print(len(time))
     obs=M[stationID]
     for t in np.arange(len(time)):       
         date_string=time[t]
         a=datetime.strptime( date_string, '%d/%m/%Y %H:%M')
         b=(a-start_calib).days
         meas_calib[b]=obs[t]
     
     aa=datetime.strptime('02/01/1979 00:00', '%d/%m/%Y %H:%M')
     cc=(start_calib-aa).days
     #print(cc)
     dis_calib=np.zeros((14975,1)) + np.nan
     dis_calib[0:(14975-cc-1)]=dis[cc:-1]
     #print(dis_calib)
     #print(meas_calib)
     my_kge_calib = evaluator(kgeprime, dis_calib, meas_calib)
     
     kge_components_calib = hydro_stats.fKGE(s=dis_calib, o=meas_calib)
     
     KGElrCAL=kge_components_calib[0]
     biaslrCAL=kge_components_calib[2]
     corrlrCAL=kge_components_calib[1]
     spreadlrCAL=kge_components_calib[3]
     saelrCAL=kge_components_calib[4]
     
     #print(my_kge_calib)
     print('kge_components_calibperiod')
     print(kge_components_calib)
     '''
     if simulation == 'calibrated':
      for id in np.arange(len(dfwrite['ID'])):
          if (dfwrite['ID'][id]==int(stationID)):
             print('writing into csv file....')
             print(id)
             print(int(stationID))
             print('-----------------------------------------------------------------------------------------')
             dfwrite['CALIB_JDKGE_longrun'][id] = JDKGE
             dfwrite.to_csv("experiment_results_HEADCATCHMENTS_noLakes_noRes_minKGE041_geography_geographyclimate_JDKGE_calib.csv", index=False)           
             dfwrite['CALIB_JSD_longrun'][id] = JSD
             dfwrite.to_csv("experiment_results_HEADCATCHMENTS_noLakes_noRes_minKGE041_geography_geographyclimate_JDKGE_calib.csv", index=False)   
     '''
     if simulation == 'defaultparam':
      for id in np.arange(len(dfwrite['ID'])):

     if simulation == 'geography':
      for id in np.arange(len(dfwrite['ID'])):

  
     if simulation == 'geographyclimate':
      for id in np.arange(len(dfwrite['ID'])):
          if (dfwrite['ID'][id]==int(stationID)):
             print('writing into csv file....')
             print(id)
             print(int(stationID))
             print('-----------------------------------------------------------------------------------------')
             dfwrite['geographyclimate_JDKGE_longrun'][id] = JDKGE
             dfwrite.to_csv("experiment_results_HEADCATCHMENTS_noLakes_noRes_minKGE041_geography_geographyclimate_JDKGE_JSD.csv.csv", index=False)           
             dfwrite['geographyclimate_JSD_longrun'][id] = JSD
             dfwrite.to_csv("experiment_results_HEADCATCHMENTS_noLakes_noRes_minKGE041_geography_geographyclimate_JDKGE_JSD.csv.csv", index=False)           
     '''  