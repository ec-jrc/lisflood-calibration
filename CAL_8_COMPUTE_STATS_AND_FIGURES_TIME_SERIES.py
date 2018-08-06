# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import HydroStats
import numpy as np
import datetime
import pdb
import time
import pandas
import math
import glob
import string
from ConfigParser import SafeConfigParser

from matplotlib import rcParams
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams.update({'figure.autolayout': True})

t = time.time()


########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])
(drive, path) = os.path.splitdrive(iniFile)
(path, fil)  = os.path.split(path)
print ">> Reading settings file ("+fil+")..."

file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

parser = SafeConfigParser()
parser.read(iniFile)

ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%m/%d/%Y")  # Start of forcing
ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%m/%d/%Y")  # Start of forcing

WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')

path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
path_result = parser.get('Path', 'Result')

Qtss_csv = parser.get('CSV', 'Qtss')
Qgis_csv = parser.get('CSV', 'Qgis')


########################################################################
#   Loop through catchments and perform calibration
########################################################################

print ">> Reading Qgis2.csv file..."
stationdata = pandas.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

stationdata_sorted['KGE_cal'] = np.nan
stationdata_sorted['KGE_val'] = np.nan
stationdata_sorted['NSE_cal'] = np.nan
stationdata_sorted['NSE_val'] = np.nan

for index, row in stationdata_sorted.iterrows():
	Series = CatchmentsToProcess.ix[:,0]
	if len(Series[Series==str(row["ID"])]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
		continue
	path_subcatch = os.path.join(SubCatchmentPath,row['ID'])
	
	print row['ID']+" "+row['Val_Start'] # For debugging

	#if index!=805: # Runs only this catchment
	#	continue
		
	# Make figures directory
	try:
		os.stat(os.path.join(path_subcatch,"FIGURES"))
	except:
		os.mkdir(os.path.join(path_subcatch,"FIGURES"))
		
	# Delete contents of figures directory
	for filename in glob.glob(os.path.join(path_subcatch,"FIGURES",'*.*')):
		os.remove(filename)

	# Compute the time steps at which the calibration should start and end
	if row['Val_Start'][:10]!="Streamflow": # Check if Q record is long enough for validation
		Val_Start = datetime.datetime.strptime(row['Val_Start'],"%m/%d/%Y")
		Val_End = datetime.datetime.strptime(row['Val_End'],"%m/%d/%Y")
		Val_Start_Step = (Val_Start-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
		Val_End_Step = (Val_End-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
	else:
		Val_Start = []
		Val_End = []
		Val_Start_Step = []
		Val_End_Step = []
	Cal_Start = datetime.datetime.strptime(row['Cal_Start'],"%m/%d/%Y")
	Cal_End = datetime.datetime.strptime(row['Cal_End'],"%m/%d/%Y")
	Cal_Start_Step = (Cal_Start-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
	Cal_End_Step = (Cal_End-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
	Forcing_End_Step = (ForcingEnd-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!

	# Load observed streamflow
	streamflow_data = pandas.read_csv(Qtss_csv,sep=",", parse_dates=True, index_col=0, infer_datetime_format=True)
	Qobs = streamflow_data[row['ID']]
	Qobs[Qobs<0] = np.NaN
	
	# Load streamflow of best run
	if os.path.isfile(os.path.join(path_subcatch,"streamflow_simulated_best.csv")):
		print "Making figures for catchment "+row['ID']+", size "+str(row['CatchmentArea'])+" pixels..."
		Qsim = pandas.read_csv(os.path.join(path_subcatch,"streamflow_simulated_best.csv"), sep=",", parse_dates=True, index_col=0, header=None)
	else:
		print "streamflow_simulated_best.csv missing for catchment"+row['ID']+"..."
		continue
		
	Qsim = Qsim.ix[:,1] # from dataframe to series

	# Make dataframe with aligned Qsim and Qobs columns
	Q = pandas.concat([Qsim, Qobs], axis=1)#.reset_index()
	
	
	########################################################################
	#   Make figure consisting of several subplots
	########################################################################
	
	fig = plt.figure()
	gs = plt.GridSpec(13,6)
	
	# TEXT OF CALIBRATION RESULTS
	ax0 = plt.subplot(gs[0,:])
	texts = r"\huge \bfseries "+str(row["ID"])+": "+str(row["RiverName"])+" at "+str(row["StationName"])
	texts = texts.replace("_","\_")
	texts_filtered = filter(lambda x: x in string.printable, texts)
	ax0.text(0.5, 0.0, texts_filtered, verticalalignment='top',horizontalalignment='center', transform=ax0.transAxes)
	plt.axis('off')

	
	# FIGURE OF CALIBRATION PERIOD TIME SERIES
	Dates_Cal = Q.loc[Cal_Start:Cal_End].index
	Q_sim_Cal = Q.loc[Cal_Start:Cal_End].ix[:,0].values
	Q_obs_Cal = Q.loc[Cal_Start:Cal_End].ix[:,1].values
	ax1 = plt.subplot(gs[1:4,:])
	ax1.plot(Dates_Cal,Q_sim_Cal,'r',Dates_Cal,Q_obs_Cal,'b')
	ax1.set_title('(a) Streamflow time series for calibration period')
	locs, labels = plt.xticks()
	plt.setp(labels,rotation=0)
	plt.ylabel(r'Streamflow [m$^{3}$ s$^{-1}$]')
	statsum = r" " \
		+"KGE$="+"{0:.2f}".format(HydroStats.KGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
		+"$, NSE$="+"{0:.2f}".format(HydroStats.NS(s=Q_sim_Cal,o=Q_obs_Cal,warmup=WarmupDays)) \
		+"$, $R="+"{0:.2f}".format(HydroStats.correlation(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
		+"$, $B="+"{0:.2f}".format(HydroStats.pc_bias2(s=Q_sim_Cal,o=Q_obs_Cal,warmup=WarmupDays)) \
		+"$ \%"
	ax1.text(0.025, 0.93, statsum, verticalalignment='top',horizontalalignment='left', transform=ax1.transAxes)
	leg = ax1.legend(['Simulated', 'Observed'], fancybox=True, framealpha=0.8,prop={'size':12},labelspacing=0.1)
	leg.get_frame().set_edgecolor('white')
	stationdata_sorted.loc[index,'KGE_cal'] = HydroStats.KGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)
	stationdata_sorted.loc[index,'NSE_cal'] = HydroStats.NS(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)
		
	# FIGURE OF VALIDATION PERIOD TIME SERIES
	if row['Val_Start'][:10]!="Streamflow": # Check if Q record is long enough for validation
		Dates_Val = Q.loc[Val_Start:Val_End].index
		Q_sim_Val = Q.loc[Val_Start:Val_End].ix[:,0].values
		Q_obs_Val = Q.loc[Val_Start:Val_End].ix[:,1].values
		if len(Q_obs_Val[~np.isnan(Q_obs_Val)])>(365*2):
			ax2 = plt.subplot(gs[4:7,:])
			ax2.plot(Dates_Val,Q_sim_Val,'r',Dates_Val,Q_obs_Val,'b')
			ax2.set_title('(b) Streamflow time series for validation period')
			plt.ylabel(r'Streamflow [m$^3$ s$^{-1}$]')
			statsum = r" " \
				+"KGE$="+"{0:.2f}".format(HydroStats.KGE(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)) \
				+"$, NSE$="+"{0:.2f}".format(HydroStats.NS(s=Q_sim_Val,o=Q_obs_Val,warmup=WarmupDays)) \
				+"$, $R="+"{0:.2f}".format(HydroStats.correlation(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)) \
				+"$, $B="+"{0:.2f}".format(HydroStats.pc_bias2(s=Q_sim_Val,o=Q_obs_Val,warmup=WarmupDays)) \
				+"$ \%"
			ax2.text(0.025, 0.93, statsum, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes)
			stationdata_sorted.loc[index,'KGE_val'] = HydroStats.KGE(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)
			stationdata_sorted.loc[index,'NSE_val'] = HydroStats.NS(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)	
	   
		
	# FIGURE OF MONTHLY CLIMATOLOGY FOR CALIBRATION PERIOD
	Q_obs_clim_Cal = np.zeros(shape=(12,1))*np.NaN
	Q_sim_clim_Cal = np.zeros(shape=(12,1))*np.NaN
	Q_obs_clim_Cal_stddev = np.zeros(shape=(12,1))*np.NaN
	Q_sim_clim_Cal_stddev = np.zeros(shape=(12,1))*np.NaN
	for month in range(1,13):
		mask = ~np.isnan(Q_obs_Cal) & ~np.isnan(Q_sim_Cal)
		Q_obs_clim_Cal[month-1] = np.mean(Q_obs_Cal[(Dates_Cal.month==month) & mask])
		Q_sim_clim_Cal[month-1] = np.mean(Q_sim_Cal[(Dates_Cal.month==month) & mask])
		Q_obs_clim_Cal_stddev[month-1] = np.std(Q_obs_Cal[(Dates_Cal.month==month) & mask])
		Q_sim_clim_Cal_stddev[month-1] = np.std(Q_sim_Cal[(Dates_Cal.month==month) & mask])
	ax3 = plt.subplot(gs[7:10,0:3])
	months = np.array([9,10,11,12,1,2,3,4,5,6,7,8,9,10]) # water year
	ax3.fill_between(np.arange(0,14),(Q_sim_clim_Cal[months-1]+0.5*Q_sim_clim_Cal_stddev[months-1]).reshape(-1),(Q_sim_clim_Cal[months-1]-0.5*Q_sim_clim_Cal_stddev[months-1]).reshape(-1),facecolor='red',alpha=0.1,edgecolor='none')
	ax3.fill_between(np.arange(0,14),(Q_obs_clim_Cal[months-1]+0.5*Q_obs_clim_Cal_stddev[months-1]).reshape(-1),(Q_obs_clim_Cal[months-1]-0.5*Q_obs_clim_Cal_stddev[months-1]).reshape(-1),facecolor='blue',alpha=0.1,edgecolor='none')
	ax3.plot(range(0,14),Q_sim_clim_Cal[months-1],'r',range(0,14),Q_obs_clim_Cal[months-1],'b')
	ax3.set_title('(c) Monthly $Q$ climatology cal.\ period')
	#leg2 = ax3.legend(['121', '122','343','334'], loc='best', fancybox=True, framealpha=0.5,prop={'size':12},labelspacing=0.1)
	plt.xticks(range(0,14), months)
	plt.xlim([0.5,12.5])
	plt.ylabel(r'Streamflow [m$^3$ s$^{-1}$]')
	plt.xlabel(r'Month')
	leg.get_frame().set_edgecolor('white')
		
	# FIGURE OF MONTHLY CLIMATOLOGY FOR VALIDATION PERIOD
	if row['Val_Start'][:10]!="Streamflow": # Check if Q record is long enough for validation
		if len(Q_obs_Val[~np.isnan(Q_obs_Val)])>365:
			Q_obs_clim_Val = np.zeros(shape=(12,1))*np.NaN
			Q_sim_clim_Val = np.zeros(shape=(12,1))*np.NaN
			Q_obs_clim_Val_stddev = np.zeros(shape=(12,1))*np.NaN
			Q_sim_clim_Val_stddev = np.zeros(shape=(12,1))*np.NaN
			for month in range(1,13):
				mask = ~np.isnan(Q_obs_Val) & ~np.isnan(Q_sim_Val)
				Q_obs_clim_Val[month-1] = np.mean(Q_obs_Val[(Dates_Val.month==month) & mask])
				Q_sim_clim_Val[month-1] = np.mean(Q_sim_Val[(Dates_Val.month==month) & mask])
				Q_obs_clim_Val_stddev[month-1] = np.std(Q_obs_Val[(Dates_Val.month==month) & mask])
				Q_sim_clim_Val_stddev[month-1] = np.std(Q_sim_Val[(Dates_Val.month==month) & mask])
			ax4 = plt.subplot(gs[7:10,3:6])
			months = np.array([9,10,11,12,1,2,3,4,5,6,7,8,9,10]) # water year            
			ax4.fill_between(np.arange(0,14),(Q_sim_clim_Val[months-1]+0.5*Q_sim_clim_Val_stddev[months-1]).reshape(-1),(Q_sim_clim_Val[months-1]-0.5*Q_sim_clim_Val_stddev[months-1]).reshape(-1),facecolor='red',alpha=0.1,edgecolor='none')
			ax4.fill_between(np.arange(0,14),(Q_obs_clim_Val[months-1]+0.5*Q_obs_clim_Val_stddev[months-1]).reshape(-1),(Q_obs_clim_Val[months-1]-0.5*Q_obs_clim_Val_stddev[months-1]).reshape(-1),facecolor='blue',alpha=0.1,edgecolor='none')
			ax4.plot(range(0,14),Q_sim_clim_Val[months-1],'r',range(0,14),Q_obs_clim_Val[months-1],'b')
			ax4.set_title('(d) Monthly $Q$ climatology val.\ period')
			plt.xticks(range(0,14), months)
			plt.xlim([0.5,12.5])
			plt.ylabel(r'Streamflow [m$^3$ s$^{-1}$]')
			plt.xlabel(r'Month')
	
	# FIGURES OF CALIBRATION EVOLUTION
	front_history = pandas.read_csv(os.path.join(path_subcatch,"front_history.csv"),sep=",", parse_dates=True, index_col=0)    
	
	ax10 = plt.subplot(gs[10:13,0:2])
	x = front_history['gen'].values
	#plt.fill_between(x,front_history["effavg_R"].values-0.5*front_history["effstd_R"].values, front_history["effavg_R"].values+0.5*front_history["effstd_R"].values, facecolor='0.8',edgecolor='none')
	#plt.hold(True)
	plt.plot(x,front_history['effavg_R'].values,'black')
	ax10.set_title('(e) KGE evolution')
	ax = plt.gca()
	plt.ylabel(r"KGE")
	plt.xlabel(r"Generation")
	#p = plt.Rectangle((0, 0), 0, 0, facecolor='0.8',edgecolor='none')
	#ax.add_patch(p)
	#leg2 = ax10.legend(['Mean', 'Std.\ dev.'], loc=4, fancybox=True, framealpha=0.8,prop={'size':12},labelspacing=0.1)
	#leg2.get_frame().set_edgecolor('white')
	##leg2.draw_frame(False)
	
	"""
	ax11 = plt.subplot(gs[10:13,2:4])
	plt.fill_between(x,front_history["effavg_R"].values-0.5*front_history["effstd_R"].values, front_history["effavg_R"].values+0.5*front_history["effstd_R"].values, facecolor='0.8',edgecolor='none')
	plt.hold(True)
	plt.plot(x,front_history['effavg_R'].values,'black')
	ax11.set_title('(f) Pareto front KGE')
	ax = plt.gca()
	plt.ylabel(r"KGE [$-$]")
	plt.xlabel(r"Generation")
	
	ax12 = plt.subplot(gs[10:13,4:6])
	plt.fill_between(x,front_history["effavg_B"].values-0.5*front_history["effstd_B"].values, front_history["effavg_B"].values+0.5*front_history["effstd_B"].values, facecolor='0.8',edgecolor='none')
	plt.hold(True)
	plt.plot(x,front_history['effavg_B'].values,'black')
	ax12.set_title('(g) Pareto front $|B|$')
	ax = plt.gca()
	plt.ylabel(r"$|B|$ [\%]")
	plt.xlabel(r"Generation")
	"""
	
	adjustprops = dict(left=0.1, bottom=0, right=1, top=1, wspace=-0.2, hspace=0.0)
	fig.subplots_adjust(**adjustprops)

	plt.draw()
	
	#gs.tight_layout(fig,rect=[0,0.03,1,0.95]) #pad=0.1, w_pad=0.1, h_pad=0.1
	
	fig.set_size_inches(22/2.54,30/2.54)
	
	fig.savefig(os.path.join(path_subcatch,"FIGURES",row['ID']+'_summary.pdf'), format='PDF')
	fig.savefig(os.path.join(path_subcatch,"FIGURES",row['ID']+'_summary.png'), dpi=300, format='PNG')
	
	plt.close("all")
	
	#plt.show()
	#pdb.set_trace()
	
stationdata_sorted.to_csv(os.path.join(path_result,"Qgis3.csv"),',',append=False)