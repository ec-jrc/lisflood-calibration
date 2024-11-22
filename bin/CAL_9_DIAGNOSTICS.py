import numpy as np
import os
import sys
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from hydroeval import *
import pandas
from pandas import ExcelFile
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime as dt
from datetime import date
from datetime import datetime
from datetime import timedelta
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15

iniFile = os.path.normpath(sys.argv[1])
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
if ver.find('3.') > -1:
    parser = ConfigParser()  # python 3.8
else:
    parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

# USAGE python CAL_9_optional_diagnostic_plots.py settings_plots.txt CatchmentsToProcess_XX.txt

SubCatchmentPath=parser.get('Main', 'SubCatchmentPath')
#catchments=np.arange(1,6300) # ALL the GloFAS IDs
#catchments=[461] # selection of subcatchments IDs
total_num_steps = int(parser.get('Main', 'total_num_steps')) # this is the number of computational steps between forcings start and forcings end, be mindfull of the daily and 6hours options
plots_storage_folder = parser.get('Main', 'plots_storage_folder') # folder in which the plots are saved
suffix_fig_filename = parser.get('Main', 'suffix_fig_filename') # optional deatils of the file name

#catchments list is loaded from the txt file 'CatchmentsToProcess_XX.txt' added as an argument of the script
CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)
catchments=CatchmentsToProcess[0]


outputfilenames = ['rainUps', 'snowUps', 'snowMeltUps', 'frostUps', 'actEvapo', 'theta1total', 'theta2total', 'theta3total', 'dTopToSubUps', 'qUzUps', 'qLzUps', 'percUZLZUps', 'dSubToUzUps', 'prefFlowUps', 'infUps', 'surfaceRunoffUps' , 'lossUps', 'lzUps']
# plots: 1'rainUpsX', 2'snowUpsX', 3'snowMeltUpsX', 4'frostUps', 5'actEvapo', 6'theta1totalX', 7'theta2totalX', 8'theta3totalX', 9'dTopToSubUpsX', 10'qUzUpsX', 11'qLzUpsX', 12'percUZLZUpsX', 13'dSubToUzUpsX', 14'prefFlowUpsX', 15'infUpsX', 16'surfaceRunoffUpsX' , 17'gwLossUpsX', 18'lzUpsX']
num_var = len(outputfilenames)
UpsXtss=np.zeros((total_num_steps,num_var))

for index in np.arange(len(catchments)):
 obsID = str(catchments[index])
 path_subcatch = os.path.join(SubCatchmentPath,obsID)
 if os.path.exists(os.path.join(path_subcatch,"out","streamflow_simulated_best.csv")):
    print("streamflow_simulated_best.csv for subcatchment ID "+ obsID + " exists: we can plot the results!")  
    nv = - 1
    for outfn in outputfilenames:    
     nv = nv + 1        
     tssfile=SubCatchmentPath+obsID+'/out/X/'+ outfn + '.tss'
     tssfile_data = pandas.read_csv(tssfile, index_col=0, sep=",", skiprows=3, header=None, skipinitialspace=True, engine='python')
     aa=-1           
     index_gauge=[]
     index_time=[]    
     dummy =  np.zeros((total_num_steps,))
     for ii in np.arange(len(tssfile_data.index)):
        spdtw=[]
        CC=[]
        AA=tssfile_data.index[ii]
        CC=AA.split()
        if len(CC)==1:
           index_gauge.append(CC)
        else:
           index_time.append(CC[0])
           aa=aa+1
           for gg in np.arange(len(index_gauge)): 
               dummy[aa]=CC[gg+1]
     UpsXtss[:,nv] = dummy
    

    # measurements, split date, area LDD
    observedstreamflow = pandas.read_csv(SubCatchmentPath+obsID+'/station/observations.csv', sep=",", index_col=0)         
    observed_streamflow = observedstreamflow[obsID]
    stationfile=SubCatchmentPath+obsID+'/station/station_data.csv'
    stationdata = pandas.read_csv(stationfile, sep=",", index_col=0)
    data=stationdata[obsID]
    splitdate=stationdata.index[-2]
    basin=data[stationdata.index[10]]
    countryname=data[stationdata.index[2]]
    areaLDD=str(data[stationdata.index[14]])
    
    # calibrated parameters and objective function
    LISFLOODpHistoryWRanksFile=SubCatchmentPath+obsID+'/pHistoryWRanks.csv' 
    ScoresParamValues = pandas.read_csv(LISFLOODpHistoryWRanksFile, sep=",", index_col=0)
    # scores for calibration period
    KGE=np.array(ScoresParamValues['Kling Gupta Efficiency'])[0]
    corr=np.array(ScoresParamValues['Correlation'])[0]
    bias=np.array(ScoresParamValues['Signal ratio (s/o) (Bias)'])[0]
    spread=np.array(ScoresParamValues['Noise ratio (s/o) (Spread)'])[0]
    # calibrated parameters
    gwperc_cal=np.array(ScoresParamValues['GwPercValue'])[0]
    bX_cal=np.array(ScoresParamValues['b_Xinanjiang'])[0]
    cpref_cal=np.array(ScoresParamValues['PowerPrefFlow'])[0]
    gwloss_cal=np.array(ScoresParamValues['GwLoss'])[0]   
    lztc_cal=np.array(ScoresParamValues['LowerZoneTimeConstant'])[0]
       
    # results long run with calibrated parameters
    dischargelongruncalib=SubCatchmentPath+obsID+'/out/streamflow_simulated_best.csv' 
    modelled_streamflow = pandas.read_csv(dischargelongruncalib, sep=",", index_col=0)   
     
    # dates
    startdate = observed_streamflow.index[0]
    enddate = observed_streamflow.index[-1]
    startdatemodel = modelled_streamflow.index[0]
    enddatemodel = modelled_streamflow.index[-1]          
    observed_streamflow = observed_streamflow[startdate:enddate] 
    modelled_streamflow_forcingsperiod = modelled_streamflow[startdatemodel:enddatemodel]
    modelled_streamflow_obsperiod = modelled_streamflow[startdate:enddate] # Keep only the observation period
    spdt=data[splitdate]
    spdtw=np.int(np.argwhere(modelled_streamflow_obsperiod.index==spdt))
    splitdateplot = observed_streamflow.index[spdtw]
    xaxistimevalues = [dt.datetime.strptime(d,'%d/%m/%Y %H:%M').date() for d in modelled_streamflow_obsperiod.index]
    xaxistimevalues1 = [dt.datetime.strptime(d,'%d/%m/%Y %H:%M').date() for d in modelled_streamflow_forcingsperiod.index]
 
    # FIGURES
    # figure 1: modelled vs observed discharge, focus on observation period
    fig1, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,tight_layout=True)     
    titleFig1='Country: '+countryname+', Basin: '+basin+' \n ObsID '+obsID+', areaLDDkm2='+areaLDD+ '\n CAL KGE='+str(round(KGE,4))+'; corr='+str(round(corr,4))+'; bias='+str(round(bias,4))+'; var='+str(round(spread,4))
    plt.suptitle( titleFig1) 
    ax1.plot(xaxistimevalues,observed_streamflow[startdate:enddate],color='black', linewidth=0.5 ,label='observations')
    ax1.plot(xaxistimevalues,modelled_streamflow[startdate:enddate],color='red',linewidth=0.5 ,label='model')    
    ax1.plot([xaxistimevalues[spdtw],xaxistimevalues[spdtw]],[0,modelled_streamflow[startdate:enddate].max()],color='y',linewidth=1.5 ,label='split date')
    ax1.legend()    
    lineStart=observed_streamflow[startdate:enddate].min()
    lineEnd=observed_streamflow[startdate:enddate].max()
    ax2.plot(observed_streamflow[splitdateplot:enddate],modelled_streamflow[splitdateplot:enddate],marker='o',markersize=2,linestyle='None', color = 'blue',label='calibration')
    ax2.plot(observed_streamflow[startdate:splitdateplot],modelled_streamflow[startdate:splitdateplot],marker='o',markersize=2,linestyle='None', color = 'y',label='outside of calib. period')
    ax2.plot([lineStart, lineEnd], [lineStart, lineEnd],color = 'm', linewidth=1.0 ,label='1:1')
    ax2.grid(visible=None, which='major', axis='both')
    ax2.legend()
    plt.xlabel('observations', axes=ax2)
    plt.ylabel('model', axes=ax2)
    savefig1 = plots_storage_folder + obsID + '_DischargeObsPeriod_' + suffix_fig_filename + '.png' 
    fig1.savefig(savefig1)
    plt.close(fig1)

    
    # figure 2: modelled vs observed discharge, same time interval of the forcings
    fig2, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,tight_layout=True) 
    titleFig2='Country: '+countryname+', Basin: '+basin+' \n ObsID '+obsID+', areaLDDkm2='+areaLDD+ '\n CAL KGE='+str(round(KGE,4))+'; corr='+str(round(corr,4))+'; bias='+str(round(bias,4))+'; var='+str(round(spread,4))
    plt.suptitle( titleFig2) 
    ax1.plot(xaxistimevalues,observed_streamflow[startdate:enddate],color='black', linewidth=0.5 ,label='observations')
    ax1.plot(xaxistimevalues1,modelled_streamflow[startdatemodel:enddatemodel],color='red',linewidth=0.5 ,label='model')
    ax1.set_xlim([365*10,365*20])
    ax2.plot(xaxistimevalues,observed_streamflow[startdate:enddate],color='black', linewidth=0.5 ,label='observations')
    ax2.plot(xaxistimevalues1,modelled_streamflow[startdatemodel:enddatemodel],color='red',linewidth=0.5 ,label='model')
    ax2.set_xlim([365*20,365*30])   
    ax3.plot(xaxistimevalues,observed_streamflow[startdate:enddate],color='black', linewidth=0.5 ,label='observations')
    ax3.plot(xaxistimevalues1,modelled_streamflow[startdatemodel:enddatemodel],color='red',linewidth=0.5 ,label='model')
    ax3.set_xlim([365*30,365*40])    
    ax4.plot(xaxistimevalues,observed_streamflow[startdate:enddate],color='black', linewidth=0.5 ,label='observations')
    ax4.plot(xaxistimevalues1,modelled_streamflow[startdatemodel:enddatemodel],color='red',linewidth=0.5 ,label='model')
    ax4.set_xlim([365*40,365*50])  
    ax1.legend()     
    lineStart=modelled_streamflow[startdate:enddate].min()
    lineEnd=modelled_streamflow[startdate:enddate].max()
    plt.xlabel('time', axes=ax2)
    plt.ylabel('discharge [m3/sec]', axes=ax2)
    savefig2=plots_storage_folder + obsID + '_Discharge41years_' + suffix_fig_filename + '.png' 
    fig2.savefig(savefig2)
    plt.close(fig2)
    
    
    # figure 3: meteo inputs and variables -->  1'rainUpsX', 2'snowUpsX', 3'snowMeltUpsX', 4'frostUps', 5'actEvapo',
    init=np.int(np.argwhere(modelled_streamflow.index=='01/01/1982 00:00'))
    fig3, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,tight_layout=True) 
    ax1.plot(xaxistimevalues1,UpsXtss[:,0],color='black', linewidth=0.5 ,label='rain (total) [mm/day]')
    ax1.plot(xaxistimevalues1,UpsXtss[:,1],color='red', linewidth=0.5 ,label='snow (total) [mm/day]')
    ax1.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,0].max()],color='m',linewidth=1.5 ,label='init.')
    ax2.plot(xaxistimevalues1,UpsXtss[:,2],color='red', linewidth=0.5 ,label='snow melt (total) [mm/day]')
    ax2.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,2].max()],color='m',linewidth=1.5 ,label='init.')
    ax3.plot(xaxistimevalues1,UpsXtss[:,3],color='green', linewidth=0.5 ,label='frost index (total) [-]')
    ax3.plot([xaxistimevalues1[0],xaxistimevalues1[-1]],[56,56],color='black', linestyle='--', linewidth=0.5 ,label='threshold [-]')
    ax3.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,3].max()],color='m',linewidth=1.5 ,label='init.')
    ax4.plot(xaxistimevalues1,UpsXtss[:,4],color='blue', linewidth=0.5 ,label='actEvapo (total) [mm/day]')
    ax4.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[UpsXtss[:,4].min(),UpsXtss[:,4].max()],color='m',linewidth=1.5 ,label='init.')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    titleFig3='Country: '+countryname+', Basin: '+basin+' \n ObsID '+obsID+', areaLDDkm2='+areaLDD+', CAL KGE='+str(round(KGE,4))+'\n cpref='+str(round(cpref_cal,3))+'; b_X='+str(round(bX_cal,3))+'; GwPerc='+str(round(gwperc_cal,3))+'; GwLoss='+str(round(gwloss_cal,3))+'; LZTC='+str(round(lztc_cal,1))
    plt.suptitle( titleFig3) 
    savefig3= plots_storage_folder + obsID + '_meteo_' + suffix_fig_filename + '.png' 
    fig3.savefig(savefig3)
    plt.close(fig3)  
    
    # figure 4: soil and groundwater fluxes  --> 9'dTopToSubUpsX', 10'qUzUpsX', 11'qLzUpsX', 12'percUZLZUpsX', 13'dSubToUzUpsX'
    fig4, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,tight_layout=True) 
    ax1.plot(xaxistimevalues1,UpsXtss[:,8],color='black', linewidth=0.5 ,label='TopToSub (total) [mm/day]')
    ax1.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,8].max()],color='m',linewidth=1.5 ,label='init.')
    ax2.plot(xaxistimevalues1,UpsXtss[:,12],color='red', linewidth=0.5 ,label='SubToUz (total) [mm/day]')
    ax2.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,12].max()],color='m',linewidth=1.5 ,label='init.')
    ax3.plot(xaxistimevalues1,UpsXtss[:,11],color='green', linewidth=0.5 ,label='percUZLZ (total) [mm/day]')
    ax3.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,11].max()],color='m',linewidth=1.5 ,label='init.')
    ax4.plot(xaxistimevalues1,UpsXtss[:,9],color='blue', linewidth=0.5 ,label='qUz (total) [mm/day]')
    ax4.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[UpsXtss[:,9].min(),UpsXtss[:,4].max()],color='m',linewidth=1.5 ,label='init.')
    ax5.plot(xaxistimevalues1,UpsXtss[:,10],color='y', linewidth=0.5 ,label='qLz (total) [mm/day]')
    ax5.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[UpsXtss[:,10].min(),UpsXtss[:,4].max()],color='m',linewidth=1.5 ,label='init.')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    titleFig4='Country: '+countryname+', Basin: '+basin+' \n ObsID '+obsID+', areaLDDkm2='+areaLDD+', CAL KGE='+str(round(KGE,4))+'\n cpref='+str(round(cpref_cal,3))+'; b_X='+str(round(bX_cal,3))+'; GwPerc='+str(round(gwperc_cal,3))+'; GwLoss='+str(round(gwloss_cal,3))+'; LZTC='+str(round(lztc_cal,1))
    plt.suptitle( titleFig4) 
    savefig4= plots_storage_folder + obsID + '_soilgroundwatefluxes_' + suffix_fig_filename + '.png' 
    fig4.savefig(savefig4)
    plt.close(fig4)     
       
    # figure 5: runoff, infiltration, preferential flow, gwloss, lz  -->  14'prefFlowUpsX', 15'infUpsX', 16'surfaceRunoffUpsX' , 17'gwLossUpsX', 18'lzUpsX' 
    fig5, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,tight_layout=True) 
    ax1.plot(xaxistimevalues1,UpsXtss[:,15],color='black', linewidth=0.5 ,label='surface runoff (total) [mm/day]')
    ax1.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,15].max()],color='m',linewidth=1.5 ,label='init.')
    ax2.plot(xaxistimevalues1,UpsXtss[:,13],color='red', linewidth=0.5 ,label='preferential flow (total) [mm/day]')
    ax2.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,13].max()],color='m',linewidth=1.5 ,label='init.')
    ax3.plot(xaxistimevalues1,UpsXtss[:,14],color='green', linewidth=0.5 ,label='infiltration (total) [mm/day]')
    ax3.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[0,UpsXtss[:,14].max()],color='m',linewidth=1.5 ,label='init.')
    ax4.plot(xaxistimevalues1,UpsXtss[:,17],color='blue', linewidth=0.5 ,label='lower zone gw (total) [mm]')
    ax4.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[UpsXtss[:,17].min(),UpsXtss[:,17].max()],color='m',linewidth=1.5 ,label='init.')
    ax5.plot(xaxistimevalues1,UpsXtss[:,16],color='blue', linewidth=0.5 ,label='gw loss (total) [mm/day]')
    ax5.plot([xaxistimevalues1[init],xaxistimevalues1[init]],[UpsXtss[:,16].min(),UpsXtss[:,10].max()],color='m',linewidth=1.5 ,label='init.')  
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    titleFig5='Country: '+countryname+', Basin: '+basin+' \n ObsID '+obsID+', areaLDDkm2='+areaLDD+', CAL KGE='+str(round(KGE,4))+'\n cpref='+str(round(cpref_cal,3))+'; b_X='+str(round(bX_cal,3))+'; GwPerc='+str(round(gwperc_cal,3))+'; GwLoss='+str(round(gwloss_cal,3))+'; LZTC='+str(round(lztc_cal,1))
    plt.suptitle( titleFig5) 
    savefig5 = plots_storage_folder + obsID + '_PrefInfRunoffGWstorage_' + suffix_fig_filename + '.png' 
    fig5.savefig(savefig5)
    plt.close(fig5)                          
    
    # figure 6: theta --> 6'theta1totalX', 7'theta2totalX', 8'theta3totalX'
    fig6, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,tight_layout=True) 
    ax1.plot(xaxistimevalues1,UpsXtss[:,5],color='black', linewidth=0.5 ,label='theta1 (total) [mm/mm]')
    ax2.plot(xaxistimevalues1,UpsXtss[:,6],color='red', linewidth=0.5 ,label='theta2 (total) [mm/mm]')
    ax3.plot(xaxistimevalues1,UpsXtss[:,7],color='green', linewidth=0.5 ,label='theta3 (total) [mm/mm]')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    titleFig6='Country: '+countryname+', Basin: '+basin+' \n ObsID '+obsID+', areaLDDkm2='+areaLDD+', CAL KGE='+str(round(KGE,4))+'\n cpref='+str(round(cpref_cal,3))+'; b_X='+str(round(bX_cal,3))+'; Theta RES. value 0.179(S<2) or 0.041(S>=2)'
    plt.suptitle(titleFig6) 
    savefig6 = plots_storage_folder + obsID + '_theta_' + suffix_fig_filename + '.png' 
    fig6.savefig(savefig6)
    plt.close(fig6) 
