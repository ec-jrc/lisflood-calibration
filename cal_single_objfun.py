# -*- coding: utf-8 -*-
import gdal
import os
import sys
import HydroStats
import array
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import pandas
import re
import pdb
import multiprocessing as mp
import time
import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15import glob
from scoop import futures
import subprocess
import stat
from datetime import datetime, timedelta
from dask.distributed import Client, LocalCluster
import cProfile
import traceback


# DD test to check the calibration algo is converging to the simulated streamflow of the model. This eliminates problems
# stemming from forcings and observations
testConvergence = False
fastDebug = False

########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])

if ver.find('3.') > -1:
    parser = ConfigParser()  # python 3.8
else:
    parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

ObservationsStart = datetime.strptime(parser.get('DEFAULT', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
ObservationsEnd = datetime.strptime(parser.get('DEFAULT', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
ForcingStart = datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
ForcingEnd = datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing

WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')
ParamRangesPath = parser.get('Path','ParamRanges')
MeteoDataPath = parser.get('Path','MeteoData')
lisfloodRoot = parser.get('Path', 'lisfloodPath')
sys.path.insert(0, lisfloodRoot)
import lisf1

path_temp = parser.get('Path', 'Temp')
path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
path_result = parser.get('Path', 'Result')

Qtss_csv = parser.get('CSV', 'Qtss')
calibrationFreq = parser.get('DEFAULT', 'calibrationFreq')

LISFLOODSettings_template = parser.get('Templates','LISFLOODSettings')
RunLISFLOOD_template = parser.get('Templates','RunLISFLOOD')

use_multiprocessing = int(parser.get('DEAP','use_multiprocessing'))
numCPUs = int(parser.get('DEAP','numCPUs'))
minGen = int(parser.get('DEAP','minGen'))
maxGen = int(parser.get('DEAP','maxGen'))
pop = int(parser.get('DEAP','pop'))
mu = int(parser.get('DEAP','mu'))
lambda_ = int(parser.get('DEAP','lambda_'))

pcraster_path = parser.get('Path', 'PCRHOME')

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
    config[execname] = getPCrasterPath(pcraster_path,iniFile,execname)

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]


########################################################################
#   Preparation for calibration
########################################################################

stationdata = pandas.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata['ObsID'] = stationdata.index

with open(sys.argv[2], "r") as f:
    catchmentIndex = int(f.readline().replace("\n", ""))
row = stationdata.loc[catchmentIndex]

path_subcatch = os.path.join(SubCatchmentPath,str(row['ObsID']))

# For some reason this version of LISFLOOD doesn't work with outlet map,
# hence have to supply gauge coordinates
gaugeloc_txt = os.path.join(path_subcatch,"maps","gaugeloc.txt")
with open(gaugeloc_txt,"r") as f:
    for line in f.readlines():
        (X,Y,value) = line.split()
gaugeloc = str(float(X))+" "+str(float(Y))
# print(gaugeloc)

# Check if inflow.tss file is present
# inflow.tss is created using CAL_5_PERFORM_CAL.py
inflow_tss = os.path.join(path_subcatch,"inflow","chanq.tss")
inflow_tss_last_run = os.path.join(path_subcatch,"inflow","chanq_last_run.tss")
inflow_tss_cal = os.path.join(path_subcatch,"inflow","chanq_cal.tss")
inflowflag = str(0)
if os.path.isfile(inflow_tss):
    inflowflag = str(1)


if fastDebug:
    # Turn this on for debugging faster. You can speed up further by setting maxGen = 1
    WarmupDays = 0
    if row['CAL_TYPE'] == 'HIST_6h':
        Cal_Start = datetime.strptime(row['StartDate_hist_6'], '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
    elif row['CAL_TYPE'] == 'HIST_24h':
        Cal_Start = datetime.strptime(row['StartDate_hist_24'], '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
    elif row['CAL_TYPE'] == 'NRT_6h':
        Cal_Start = datetime.strptime(row['StartDate_nrt_6'], '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
    elif row['CAL_TYPE'] == 'NRT_24h':
        Cal_Start = datetime.strptime(row['StartDate_nrt_24'], '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
    Cal_End = (datetime.strptime(Cal_Start, '%Y-%m-%d %H:%M') + timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
    ForcingStart = datetime.strptime(Cal_Start, '%Y-%m-%d %H:%M')
    ForcingEnd = datetime.strptime(Cal_End, '%Y-%m-%d %H:%M')
else:
    # Compute the time steps at which the calibration should start and end
    Cal_Start = datetime.strptime(row['Cal_Start'],'%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
    Cal_End = datetime.strptime(row['Cal_End'],'%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M') # DD original

# Load xml and .bat template files
f = open(os.path.join('templates',RunLISFLOOD_template),"r")

template_bat = f.read()
f.close()

f = open(os.path.join('templates',LISFLOODSettings_template),"r")
template_xml = f.read()
f.close()

# Load paramranges file
ParamRanges = pandas.read_csv(ParamRangesPath,sep=",",index_col=0)

# Load observed streamflow # DD Much faster IO with npy despite being more complicated (<1s vs 22s)
if testConvergence:
    try:
        streamflow_data = pandas.read_csv(SubCatchmentPath + "/" + str(row['ObsID']) + "/convergenceTester.csv", sep=",", index_col=0, header=None)
        streamflow_data.index = pandas.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
        observed_streamflow = streamflow_data[ForcingStart:ForcingEnd]
        genConvergenceTimeSeries = False
    except IOError:
        observed_streamflow = 0.0
        genConvergenceTimeSeries = True
        pass
else:
    genConvergenceTimeSeries = False
    if os.path.exists(Qtss_csv.replace(".csv", ".npy")) and os.path.getsize(Qtss_csv) > 0:
        streamflow_data = pandas.DataFrame(np.load(Qtss_csv.replace(".csv", ".npy"), allow_pickle=True))
        streamflow_datetimes = np.load(Qtss_csv.replace(".csv", "_dates.npy"), allow_pickle=True).astype('string_')
        try:
            streamflow_data.index = [datetime.strptime(i, "%Y-%m-%dT%H:%M:%S.000000000") for i in streamflow_datetimes]
        except ValueError:
            streamflow_data.index = [datetime.strptime(i, "%d/%m/%Y %H:%M") for i in streamflow_datetimes]
        streamflow_data.columns = np.load(Qtss_csv.replace(".csv", "_catchments.npy"), allow_pickle=True)
    else:
        streamflow_data = pandas.read_csv(Qtss_csv, sep=",", index_col=0)
        # streamflow_data = pandas.read_csv(Qtss_csv, sep=",", index_col=0, parse_dates=True) # DD WARNING buggy unreliable parse_dates! Don't use it!
        np.save(Qtss_csv.replace(".csv", ".npy"), streamflow_data)
        np.save(Qtss_csv.replace(".csv", "_dates.npy"), streamflow_data.index)
        np.save(Qtss_csv.replace(".csv", "_catchments.npy"), streamflow_data.columns.values)
    observed_streamflow = streamflow_data[str(row['ObsID'])]
    observed_streamflow = observed_streamflow[ForcingStart:ForcingEnd] # Keep only the part for which we run LISFLOOD
    print("PLACE BREAKPOINT HERE")

# DD Use a multiprocessing shared Value type to keep track of the generations so we can access it in the RunModel function
if use_multiprocessing == True:
    gen = mp.Value('i')
    with gen.get_lock():
        gen.value = -1

    runNumber = mp.Value("i")
    with runNumber.get_lock():
        runNumber.value = -1

# DD Function to profile the multiprocessing children
def profiler(Individual, mapLoadOnly=None):
    profNum = int(random.random()*1e6)
    print("run profiler " + str(profNum))
    ret = []
    cProfile.runctx('wrapper(ret, Individual, mapLoadOnly=None)', globals(), locals(), 'prof%d.cprof' %profNum)
    return ret[0]

# DD Wrapper function to retrieve result of the profiled function
def wrapper(ret, Individual, mapLoadOnly=None):
    ret.append(RunModel(Individual, mapLoadOnly=None))

def findBestSAERuns(numOfRuns, population, allowDuplicates=True):
    pHistory = pandas.read_csv(os.path.join(path_subcatch, "paramsHistory.csv"), sep=",")[3:]
    saes = []
    for ind in population:
        # Make sure we find at least one element in the table, incl. the fact we can only represent 12 digit numbers in, so try to find a match with decreasing precition until a match is found
        indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-11]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-10]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-9]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-8]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-7]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-7]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-6]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-5]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-4]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-3]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-2]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-1]                
        if len(indIndex) == 0:
            print("BLAAT")
            print(ind.fitness.values)
            for ii, i in enumerate(pHistory["Kling Gupta Efficiency"]):
                print(ii, i, abs(i - ind.fitness.values[0]), abs(i - ind.fitness.values[0]) <= 1e-11, abs(i - ind.fitness.values[0]) <= 1e-10, abs(i - ind.fitness.values[0]) <= 1e-9, abs(i - ind.fitness.values[0]) <= 1e-8, abs(i - ind.fitness.values[0]) <= 1e-7, abs(i - ind.fitness.values[0]) <= 1e-6, abs(i - ind.fitness.values[0]) <= 1e-5)
            print( "MEEUUHH")
        indSae = pHistory['sae'][indIndex]
        indKGE = pHistory["Kling Gupta Efficiency"][indIndex]
        # Round the SAE as well to a precision of 1e4
        saes.append([round(indSae[indIndex[0]] * 1e4) / 1e4, indKGE[indIndex[0]], ind])
    if allowDuplicates:
        uniques = sorted([ind[0] for ind in saes])
    else:
        uniques = sorted(set([ind[0] for ind in saes]))
    saesNoDups = []
    for u in uniques:
        saesNoDups.append([ind for ind in saes if ind[0] == u])
    # by appending automatically, the values are sorted
    children = []
    for i in range(numOfRuns):
        children.append(saesNoDups[i][0][2])
    return children

########################################################################
#   Function for running the model, returns objective function scores
########################################################################
#@profile
def RunModel(Individual, mapLoadOnly=None):
    Cal_Start_Local = Cal_Start
    if mapLoadOnly is None:
        Cal_End_Local = Cal_End
    elif mapLoadOnly:
        Cal_End_Local = datetime.strptime(Cal_Start, "%Y-%m-%d %H:%M") + timedelta(days=0.25) #datetime.strptime(row['Cal_Start'], '%d/%m/%Y %H:%M') + timedelta(days=0.25)
        Cal_End_Local = Cal_End_Local.strftime('%Y-%m-%d %H:%M')
    elif genConvergenceTimeSeries:
        Cal_Start_Local = ForcingStart.strftime('%Y-%m-%d %H:%M')
        Cal_End_Local = ForcingEnd.strftime('%Y-%m-%d %H:%M')

    # Convert scaled parameter values ranging from 0 to 1 to usncaled parameter values
    Parameters = [None] * len(ParamRanges)
    for ii in range(len(ParamRanges)):
        Parameters[ii] = Individual[ii]*(float(ParamRanges.iloc[ii,1])-float(ParamRanges.iloc[ii,0]))+float(ParamRanges.iloc[ii,0])

    # Note: The following code must be identical to the code near the end where LISFLOOD is run
    # using the "best" parameter set. This code:
    # 1) Modifies the settings file containing the unscaled parameter values amongst other things
    # 2) Makes a .bat file to run LISFLOOD
    # 3) Runs LISFLOOD and loads the simulated streamflow
    # Random number is appended to settings and .bat files to avoid simultaneous editing
    run_rand_id = str(int(random.random()*1e10)).zfill(12)
    template_xml_new = template_xml
    for ii in range(0,len(ParamRanges)):
        ## DD Special Rule for the SAVA
        if str(row['ObsID']) == '851' and (ParamRanges.index[ii] == "adjust_Normal_Flood" or ParamRanges.index[ii] == "ReservoirRnormqMult"):
            template_xml_new = template_xml_new.replace('%adjust_Normal_Flood',"0.8")
            template_xml_new = template_xml_new.replace('%ReservoirRnormqMult',"1.0")
        template_xml_new = template_xml_new.replace("%"+ParamRanges.index[ii],str(Parameters[ii]))
    template_xml_new = template_xml_new.replace('%gaugeloc',gaugeloc) # Gauge location
    template_xml_new = template_xml_new.replace('%CalStart', Cal_Start_Local) # Date of Cal starting
    template_xml_new = template_xml_new.replace('%CalEnd', Cal_End_Local)  # Time step of forcing at which to end simulation
    template_xml_new = template_xml_new.replace('%CatchmentDataPath',CatchmentDataPath) # Directory with forcing for the whole region
    template_xml_new = template_xml_new.replace('%SubCatchmentPath',path_subcatch)
    template_xml_new = template_xml_new.replace('%MeteoDataPath',MeteoDataPath)
    template_xml_new = template_xml_new.replace('%run_rand_id',run_rand_id)
    template_xml_new = template_xml_new.replace('%inflowflag',inflowflag)

    template_xml_new2 = template_xml_new
    template_xml_new = template_xml_new.replace('%InitLisflood',"1")
    f = open(os.path.join(path_subcatch,os.path.basename(LISFLOODSettings_template[:-4]+'-PreRun'+run_rand_id+'.xml')), "w")
    f.write(template_xml_new)
    f.close()
    template_xml_new2 = template_xml_new2.replace('%InitLisflood',"0")
    f = open(os.path.join(path_subcatch,os.path.basename(LISFLOODSettings_template[:-4]+'-Run'+run_rand_id+'.xml')), "w")
    f.write(template_xml_new2)
    f.close()

    # DD Do not run lisflood twice in a bash script, instead, import lisflood and keep everything in memory to reduce IO
    currentdir = os.getcwd()

    templatePathPreRun = os.path.join(path_subcatch, os.path.basename(LISFLOODSettings_template[:-4] + '-PreRun' + run_rand_id + '.xml'))
    templatePathRun = os.path.join(path_subcatch, os.path.basename(LISFLOODSettings_template[:-4] + '-Run' + run_rand_id + '.xml'))
    # Cleanest way to import lisflood by importing only the main function.
    # from lisf1 import main as lisfloodMain
    # lisfloodMain([template, template])
    # This method crashes on the second run (explained below + fix)
    # OR
    # Easiest way to import lisflood. This keeps lisf1 into memory, including all subobjects such as
    # e.g. lisf1.binding. The problem is that the latter is edited in Lisflood_initial.py when
    # self.output_module = outputTssMap(self) is run, which in turns runs some code in output.py.
    # This code seems to add a 'Catchments' and a '1' key to the binding dictionary.
    # If we then later want to rerun lisflood, since everything is kept in memory, the code tries to interpret
    # lfuser and lfbinding from the settings xml files, and suddenly encounters this 'Catchments' key which actually doesn't exist
    # in the xml code. This makes the second run crash. To avoid this without modifying whatever P.Burek coded
    # in output.py, just clean up this binding that was in memory. Besides, as we rerun Lisflood, e.g. in calibration suite
    # we want to reread the xml settings anew, but we only overwrite whatever is in memory. So making sure that this binding
    # variable is clean is crucial in case not all memory contents gets overwritten in this process.
    # Alternatively we could try to entirely delete the lisf1 object, but that would defeat the purpose of this code which tries
    # to minimise disk IO in between two Lisflood runs by keeping as much as possible into memory.
    # DD sync the loaded maps into the lisf1 object so lisflood won't have to reread the files from disk
    try:
        if mapLoadOnly is None:
            try:
                del lisf1.binding['Catchments']
                del lisf1.binding['1']
            except KeyError:
                pass
        lisf1.main(lisfloodRoot + "dummy.txt", templatePathPreRun, '-v', '-t')
        if mapLoadOnly is None:
            try:
                del lisf1.binding['Catchments']
                del lisf1.binding['1']
            except KeyError:
                pass
            lisf1.main(lisfloodRoot + "dummy.txt", templatePathRun, '-v', '-t')
        elif genConvergenceTimeSeries and not mapLoadOnly:
            try:
                del lisf1.binding['Catchments']
                del lisf1.binding['1']
            except KeyError:
                pass
            lisf1.main(lisfloodRoot + "dummy.txt", templatePathRun, '-v', '-t')
            Qsim_tss = os.path.join(path_subcatch, "out", 'dis' + run_rand_id + '.tss')
            simulated_streamflow = pandas.read_csv(Qsim_tss, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
            simulated_streamflow[1][simulated_streamflow[1] == 1e31] = np.nan
            Qsim = simulated_streamflow[1].values
            print( ">> Saving simulated streamflow with default parameters(convergenceTester.csv)")
            Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(ForcingStart, periods=len(Qsim), freq='6H'))
            Qsim.to_csv(os.path.join(path_subcatch, "convergenceTester.csv"), ',', header="")
            return Qsim
        elif mapLoadOnly:
            return
    except:
        traceback.print_exc()
        raise Exception("")

    os.chdir(currentdir)

    # DD Extract simulation
    Qsim_tss = os.path.join(path_subcatch,"out",'dis'+run_rand_id+'.tss')
    print('rand_id',run_rand_id)
    if os.path.isfile(Qsim_tss)==False:
        print("run_rand_id: "+str(run_rand_id))
        raise Exception("No simulated streamflow found. Probably LISFLOOD failed to start? Check the log files of the run!")
    simulated_streamflow = pandas.read_csv(Qsim_tss,sep=r"\s+",index_col=0,skiprows=4,header=None,skipinitialspace=True)
    simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan
    simulated_streamflow.index = [datetime.strptime(Cal_Start_Local, "%Y-%m-%d %H:%M") + timedelta(hours=6*(i-1)) for i in simulated_streamflow.index]

    # DD: To save us from this error:
    # Exception: run_rand_id: 008968777033: observed and simulated streamflow arrays have different number of elements (3127 and 7292 elements, respectively)
    # DD Make dataframe with aligned Qsim and Qobs columns
    Q = pandas.concat({"Sim": simulated_streamflow[1], "Obs": observed_streamflow}, axis=1)  # .reset_index()

    # Finally, extract equal-length arrays from it
    Qobs = np.array(Q['Obs'][Cal_Start:Cal_End]) #.values+0.001
    Qsim = np.array(Q['Sim'][Cal_Start:Cal_End])

    # DD: Add flag to select calibration frequency. We always load 6-hourly observations, so we need to aggregate them to daily obs
    if not testConvergence:
        if calibrationFreq == r"6-hourly":
            # DD: Check if daily or 6-hourly observed streamflow is available
            # DD: Aggregate 6-hourly simulated streamflow to daily ones
            if row["CAL_TYPE"].find("_24h") > -1:
                # DD: Overwrite index with date range so we can use Pandas' resampling + mean function to easily average 6-hourly to daily data
                Qsim = simulated_streamflow
                Qsim.index = pandas.date_range(Cal_Start, Cal_End, freq="360min")
                Qsim = Qsim.resample('24H', label="right", closed="right").mean()
                Qsim = np.array(Qsim) #[1].values + 0.001
                # Same for Qobs
                Qobs = observed_streamflow[Cal_Start:Cal_End]
                Qobs.index = pandas.date_range(Cal_Start, Cal_End, freq="360min")
                Qobs = Qobs.resample('24H', label="right", closed="right").mean()
                Qobs = np.array(Qobs) #[1].values + 0.001
                # Trim nans
                Qsim = Qsim[~np.isnan(Qobs)]
                Qobs = Qobs[~np.isnan(Qobs)]
        elif calibrationFreq == r"daily":
            # DD Untested code! DEBUG TODO
            Qobs = observed_streamflow[Cal_Start:Cal_End]
            Qobs.index = pandas.date_range(Cal_Start, Cal_End, freq="360min")
            Qobs = Qobs.resample('24H', label="right", closed="right").mean()
            Qobs = np.array(Qobs) #[1].values + 0.001

    # Check that returns error
    if len(Qobs) != len(Qsim):
        raise Exception("run_rand_id: "+str(run_rand_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

    # Compute objective function score
    fKGEComponents = HydroStats.fKGE(s=Qsim, o=Qobs, warmup=WarmupDays, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False)
    KGE = fKGEComponents[0]

    # retrieve the array in shared memory
    if use_multiprocessing:
        with runNumber.get_lock():
            runNumber.value += 1
            if runNumber.value == max(pop, lambda_):
                runNumber.value = 0
    print("   run_rand_id: "+str(run_rand_id)+", KGE: "+"{0:.3f}".format(KGE))

    lock.acquire()
    with open(os.path.join(path_subcatch,"runs_log.csv"), "a") as myfile:
        myfile.write(str(run_rand_id)+","+str(KGE)+"\n")

    # DD We want to check that the parameter space is properly sampled. Write them out to file now
    paramsHistoryFilename = os.path.join(path_subcatch, "paramsHistory.csv")
    if not os.path.exists(paramsHistoryFilename) or os.path.getsize(paramsHistoryFilename) == 0:
        paramsHistoryFile = open(paramsHistoryFilename, "w")
        # Headers
        paramsHistory = "randId,"
        for i in [str(ip) + "," for ip in ParamRanges.index.values]:
            paramsHistory += i
        for i in [str(ip) + "," for ip in ["Kling Gupta Efficiency", "Correlation", "Signal ratio (s/o) (Bias)", "Noise ratio (s/o) (Spread)", "sae", "generation", "runNumber"]]:
            paramsHistory += i
        paramsHistory += "\n"
        # Minimal values
        paramsHistory += str(ParamRanges.head().columns.values[0]) + ","
        for i in [str(ip) + "," for ip in ParamRanges[str(ParamRanges.head().columns.values[0])].values]:
            paramsHistory += i
        paramsHistory += "\n"
        # Default values
        paramsHistory += str(ParamRanges.head().columns.values[2]) + ","
        for i in [str(ip) + "," for ip in ParamRanges[str(ParamRanges.head().columns.values[2])].values]:
            paramsHistory += i
        paramsHistory += "\n"
        # Maximal values
        paramsHistory += str(ParamRanges.head().columns.values[1]) + ","
        for i in [str(ip) + "," for ip in ParamRanges[str(ParamRanges.head().columns.values[1])].values]:
            paramsHistory += i
        paramsHistory += "\n\n"
    else:
        paramsHistoryFile = open(paramsHistoryFilename, "a")
        paramsHistory = ""
    paramsHistory += str(run_rand_id) + ","
    for i in [str(ip) + "," for ip in Parameters]:
        paramsHistory += i
    for i in [str(ip) + "," for ip in fKGEComponents]:
        paramsHistory += i
    if use_multiprocessing:
        paramsHistory += str(HydroStats.sae(s=Qsim, o=Qobs, warmup=WarmupDays)) + ","
        paramsHistory += str(gen.value) + ","
        paramsHistory += str(runNumber.value)
    paramsHistory += "\n"
    paramsHistoryFile.write(paramsHistory)
    paramsHistoryFile.close()
    lock.release()

    return KGE, # If using just one objective function, put a comma at the end!!!


########################################################################
#   Perform calibration using the DEAP module
########################################################################

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.uniform, 0, 1)

# Structure initializers
toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(ParamRanges))
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

def checkBounds(min, max):
    def decorator(func):
        def wrappper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrappper
    return decorator

toolbox.register("evaluate", RunModel)
toolbox.register("mate", tools.cxBlend, alpha=0.15)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
toolbox.register("select", tools.selNSGA2)

toolbox.decorate("mate", checkBounds(0, 1))
toolbox.decorate("mutate", checkBounds(0, 1))

#@profile
def main():
    if os.path.exists(os.path.join(path_subcatch,"pareto_front.csv"))==False:
        t = time.time()

        # DD Run Lisflood a first time before forking to load the stacks into memory
        # try:
        minParams = ParamRanges[str(ParamRanges.head().columns.values[0])].values
        maxParams = ParamRanges[str(ParamRanges.head().columns.values[1])].values
        defaultParams = ParamRanges[str(ParamRanges.head().columns.values[2])].values

        ## DD uncomment to generate a synthetic run with default parameters to converge to
        if genConvergenceTimeSeries:
            RunModel((defaultParams - minParams) / (maxParams - minParams), mapLoadOnly=False)
            print("Finished generating default run. Please relaunch the calibration. It will now try to converge to this default run.")
            sys.exit(0)
        else:
            RunModel((defaultParams-minParams)/(maxParams-minParams), mapLoadOnly=True)

        if use_multiprocessing==True:
            global lock
            lock = mp.Lock()
            pool_size = numCPUs #mp.cpu_count() * 1 ## DD just restrict the number of CPUs to use manually
            pool = mp.Pool(processes=pool_size, initargs=(lock,))
            toolbox.register("map", pool.map)

        #cxpb = 0.9 # For someone reason, if sum of cxpb and mutpb is not one, a lot less Pareto optimal solutions are produced
        # DD: These values are used as percentage probability, so they should add up to 1, to determine whether to mutate or cross. The former finds the parameter for the next generation by taking the average of two parameters. This could lead to convergence to a probability set by the first generation as a biproduct of low first-generation parameter spread (they are generated using a uniform-distribution random generator.
        #mutpb = 0.1
        cxpb = 0.6
        mutpb = 0.4

        # Initialise statistics arrays
        effmax = np.zeros(shape=(maxGen + 1, 1)) * np.NaN
        effmin = np.zeros(shape=(maxGen + 1, 1)) * np.NaN
        effavg = np.zeros(shape=(maxGen + 1, 1)) * np.NaN
        effstd = np.zeros(shape=(maxGen + 1, 1)) * np.NaN

        # Start generational process setting all stopping conditions to false
        conditions = {"maxGen": False, "StallFit": False}

        # Start a new hall of fame
        halloffame = tools.ParetoFront()

        # Attempt to open a previous parameter history
        try:
            # Open the paramsHistory file from previous runs
            paramsHistory = pandas.read_csv(os.path.join(path_subcatch, "paramsHistory.csv"), sep=",")[3:]
            print("Restoring previous calibration state")
            def updatePopulationFromHistory(pHistory):
                n = len(pHistory)
                paramvals = np.zeros(shape=(n, len(ParamRanges)))
                paramvals[:] = np.NaN
                invalid_ind = []
                fitnesses = []
                for ind in range(n):
                    for ipar, par in enumerate(ParamRanges.index):
                        # # scaled to unscaled conversion
                        # paramvals[ind][ipar] = pHistory.iloc[ind][par] * (float(ParamRanges.iloc[ipar,1]) - \
                        #   float(ParamRanges.iloc[ipar,0]))+float(ParamRanges.iloc[ipar,0])
                        # unscaled to scaled conversion
                        paramvals[ind][ipar] = (pHistory.iloc[ind][par] - float(ParamRanges.iloc[ipar, 0])) / \
                          (float(ParamRanges.iloc[ipar, 1]) - float(ParamRanges.iloc[ipar, 0]))
                    # Create a fresh individual with the restored parameters
                    # newInd = toolbox.Individual() # creates an individual with random numbers for the parameters
                    newInd = creator.Individual(list(paramvals[ind]))  # creates a totally empty individual
                    invalid_ind.append(newInd)
                    # WARNING: Change the following line when using multi-objective functions
                    # also load the old KGE the individual had (works only for single objective function)
                    fitnesses.append((pHistory.iloc[ind][len(ParamRanges) + 1],))
                # update the score of each
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                return invalid_ind

            # Initiate the generations counter
            with gen.get_lock():
                gen.value = 0

            population = None
            # reconstruct the generational evoluation
            for igen in range(int(paramsHistory["generation"].iloc[-1])+1):
                # retrieve the generation's data
                parsHistory = paramsHistory[paramsHistory["generation"] == igen]
                # reconstruct the invalid individuals array
                valid_ind = updatePopulationFromHistory(parsHistory)
                # Update the hall of fame with the generation's parameters
                halloffame.update(valid_ind)
                # prepare for the next stage
                if population is not None:
                    population[:] = toolbox.select(population + valid_ind, mu)
                else:
                    population = valid_ind

                # Loop through the different objective functions and calculate some statistics from the Pareto optimal population
                for ii in range(1):
                    effmax[gen.value, ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                    effmin[gen.value, ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                    effavg[gen.value, ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                    effstd[gen.value, ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                print(">> gen: " + str(gen.value) + ", effmax_KGE: " + "{0:.3f}".format(effmax[gen.value, 0]))

                # Terminate the optimization after maxGen generations
                if gen.value >= maxGen:
                    print(">> Termination criterion maxGen fulfilled.")
                    conditions["maxGen"] = True

                if gen.value >= minGen:
                    # DD attempt to stop early with different criterion
                    if (effmax[gen.value, 0] - effmax[gen.value - 3, 0]) < 0.003:
                        print(">> Termination criterion no-improvement KGE fulfilled.")
                        conditions["StallFit"] = True
                with gen.get_lock():
                    gen.value += 1

        # No previous parameter history was found, so start from scratch
        except IOError:

            # Start with a fresh population
            population = toolbox.population(n=pop)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid] # DD this filters the population or children for uncalculated fitnesses. We retain only the uncalculated ones to avoid recalculating those that already had a fitness. Potentially this can save time, especially if the algorithm tends to produce a child we already ran.
            with gen.get_lock():
                gen.value = 0
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) # DD this runs lisflood and calculates the fitness, here KGE
            for ind, fit in zip(invalid_ind, fitnesses): # DD this updates the fitness (=KGE) for the individuals in the global pool of individuals which we just calculated. ind are
                ind.fitness.values = fit

            halloffame.update(population) # DD this selects the best one

            # Loop through the different objective functions and calculate some statistics from the Pareto optimal population
            for ii in range(1):
                effmax[0,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                effmin[0,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                effavg[0,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                effstd[0,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            print(">> gen: "+str(gen.value)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen.value,0]))

            # Update the generation to the first
            with gen.get_lock():
                gen.value = 1

        # Resume the generational process from wherever we left off
        while not any(conditions.values()):

            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) # DD this runs lisflood
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Loop through the different objective functions and calculate some statistics
            # from the Pareto optimal population
            for ii in range(1):
                effmax[gen.value,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                effmin[gen.value,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                effavg[gen.value,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
                effstd[gen.value,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            print(">> gen: "+str(gen.value)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen.value,0]))

            # Terminate the optimization after maxGen generations
            if gen.value >= maxGen:
              print(">> Termination criterion maxGen fulfilled.")
              conditions["maxGen"] = True

            if gen.value >= minGen:
                if (effmax[gen.value,0]-effmax[gen.value-3,0]) < 0.003:
                    print(">> Termination criterion no-improvement KGE fulfilled.")
                    conditions["StallFit"] = True
            with gen.get_lock():
                gen.value += 1




        # Finito
        if use_multiprocessing == True:
            pool.close()
        elapsed = time.time() - t
        print(">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s")


        ########################################################################
        #   Save calibration results
        ########################################################################

        # Save history of the change in objective function scores during calibration to csv file
        front_history = pandas.DataFrame()
        front_history['gen'] = range(gen.value)
        front_history['effmax_R'] = effmax[0:gen.value,0]
        front_history['effmin_R'] = effmin[0:gen.value,0]
        front_history['effstd_R'] = effstd[0:gen.value,0]
        front_history['effavg_R'] = effavg[0:gen.value,0]
        front_history.to_csv(os.path.join(path_subcatch,"front_history.csv"))

        # DD We found that there really are 4 aspects of the discharge time series we'd like to optimise the calibration for:
        # timeliness of events (peaks), bias and spread, but also the total discharged volume, which is given by the SAE.
        # By including the SAE ratio (s/o) in the KGE, we optimise all of these 4 aspects. However, we found by looking at the actual hydrographs
        # that even including the SAE ratio in the KGE, there are compensating errors which give more weight to high discharge peaks,
        # which makes the low flows events badly timed. The lisflood run giving minimal SAE seems to alleviate this problem quite well.
        # Hence, we replace here the pareto_front of the halloffame with the one that gives the smallest SAE as this makes more sense
        # hydrologically. The run is chosen throughout all generations, and since we know that the SAE is strongly correlated with KGE,
        # it's safe to assume we won't end up with a run with very low SAE and crappy KGE overall.
        # We conclude that the KGE perhaps is a good skill score, but not so great for actual calibration with a single objective functino.
        # We now found that selecting purely the best SAE can also lead to bad selections. Therefore, we implemented a new method
        # of selecting the pareto-optimal solution based on a multi-variate distribution of ranks for 3 aspects:
        # KGE, correlation and SAE. We give equal weight to all 3. We know that the KGE includes correlation as well, but found
        # that the variability of the two other components (bias and noise ratio) dominate the KGE. Also, the timeliness of peaks
        # is most important hydrologically speaking. Thus we give a bit more importance to correlation.
        pHistory = pandas.read_csv(os.path.join(path_subcatch, "paramsHistory.csv"), sep=",")[3:]
        # Keep only the best 10% of the runs for the selection of the parameters for the next generation
        pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
        pHistory = pHistory.head(int(round(len(pHistory) * 0.1)))
        n = len(pHistory)
        # Give ranking scores to corr
        pHistory = pHistory.sort_values(by="Correlation", ascending=False)
        pHistory["corrRank"] = [float(i + 1) / n for i, ii in enumerate(pHistory["Correlation"].values)]
        # Give ranking scores to sae
        pHistory = pHistory.sort_values(by="sae", ascending=True)
        pHistory["saeRank"] = [float(i + 1) / n for i, ii in enumerate(pHistory["sae"].values)]
        # Give ranking scores to KGE
        pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
        pHistory["KGERank"] = [float(i + 1) / n for i, ii in enumerate(pHistory["Kling Gupta Efficiency"].values)]
        # Give pareto score
        pHistory["paretoRank"] = pHistory["corrRank"].values * pHistory["saeRank"].values * pHistory["KGERank"].values
        pHistory = pHistory.sort_values(by="paretoRank", ascending=True)
        # Select the best pareto candidate
        bestParetoIndex \
            = pHistory["paretoRank"].nsmallest(1).index
        # Save the pareto front
        paramvals = np.zeros(shape=(1,len(ParamRanges)))
        paramvals[:] = np.NaN
        for ipar, par in enumerate(ParamRanges.index):
            paramvals[0][ipar] = pHistory.ix[bestParetoIndex][par]
        pareto_front = pandas.DataFrame({'effover': pHistory["Kling Gupta Efficiency"].ix[bestParetoIndex], 'R': pHistory["Kling Gupta Efficiency"].ix[bestParetoIndex]}, index=[0])
        for ii in range(len(ParamRanges)):
            pareto_front["param_"+str(ii).zfill(2)+"_"+ParamRanges.index[ii]] = paramvals[0,ii]
        pareto_front.to_csv(os.path.join(path_subcatch,"pareto_front.csv"),',')
        return

    paramvals = pandas.read_csv(os.path.join(path_subcatch,"pareto_front.csv"),sep=",")

    name_params= paramvals.columns
    names=name_params[3:]
    print('names',names)
    Parameters=list()
    for indx in range(0,len(names)):
        print('name[idx]', names[indx],'paramvals',paramvals[names[indx]])
        Parameters.append(paramvals[names[indx]].values[0])

    print('param',Parameters)

    print(">> Running LISFLOOD using the \"best\" parameter set")
    # Note: The following code must be identical to the code near the end where LISFLOOD is run
    # using the "best" parameter set. This code:
    # 1) Modifies the settings file containing the unscaled parameter values amongst other things
    # 2) Makes a .bat file to run LISFLOOD
    # 3) Runs LISFLOOD and loads the simulated streamflow
    # Random number is appended to settings and .bat files to avoid simultaneous editing
    if os.path.isfile(inflow_tss) or os.path.isfile(inflow_tss_cal):
        print(inflow_tss)
        print(inflow_tss_cal)
        print(inflow_tss_last_run)
        os.rename(inflow_tss,inflow_tss_cal)
        os.rename(inflow_tss_last_run,inflow_tss)


    run_rand_id = str(int(random.random()*10000000000)).zfill(12)
    template_xml_new = template_xml
    for ii in range(0,len(ParamRanges)):
        ## DD Special Rule for the SAVA
        if str(row['ObsID']) == '851' and (ParamRanges.index[ii] == "adjust_Normal_Flood" or ParamRanges.index[ii] == "ReservoirRnormqMult"):
            template_xml_new = template_xml_new.replace('%adjust_Normal_Flood',"0.8")
            template_xml_new = template_xml_new.replace('%ReservoirRnormqMult',"1.0")
            # os.system("cp %s %s" % (ParamRangesPath.replace(".csv", "851Only.csv"), ParamRangesPath))
        template_xml_new = template_xml_new.replace("%"+ParamRanges.index[ii],str(Parameters[ii]))        
    template_xml_new = template_xml_new.replace('%gaugeloc',gaugeloc) # Gauge location

    # DD DEBUG check timestep vs calendar functionality 
    template_xml_new = template_xml_new.replace('%CalStart', ForcingStart.strftime('%Y-%m-%d %H:%M')) # Time step of forcing at which to start simulation
    template_xml_new = template_xml_new.replace('%CalEnd', ForcingEnd.strftime('%Y-%m-%d %H:%M')) # Time step of forcing at which to end simulation
    template_xml_new = template_xml_new.replace('%CatchmentDataPath',CatchmentDataPath) # Directory with forcing for the whole region
    template_xml_new = template_xml_new.replace('%SubCatchmentPath',path_subcatch) # Directory with data for subcatchments
    template_xml_new = template_xml_new.replace('%MeteoDataPath',MeteoDataPath)
    template_xml_new = template_xml_new.replace('%run_rand_id',run_rand_id)
    template_xml_new = template_xml_new.replace('%inflowflag',inflowflag)

    template_xml_new2 = template_xml_new
    template_xml_new = template_xml_new.replace('%InitLisflood',"1")
    f = open(os.path.join(path_subcatch,os.path.basename(LISFLOODSettings_template[:-4]+'-PreRun'+run_rand_id+'.xml')), "w")
    f.write(template_xml_new)
    f.close()
    template_xml_new2 = template_xml_new2.replace('%InitLisflood',"0")
    f = open(os.path.join(path_subcatch,os.path.basename(LISFLOODSettings_template[:-4]+'-Run'+run_rand_id+'.xml')), "w")
    f.write(template_xml_new2)
    f.close()
   
    currentdir = os.getcwd()
   
    templatePathPreRun = os.path.join(path_subcatch, os.path.basename(LISFLOODSettings_template[:-4] + '-PreRun' + run_rand_id + '.xml'))
    templatePathRun = os.path.join(path_subcatch, os.path.basename(LISFLOODSettings_template[:-4] + '-Run' + run_rand_id + '.xml'))
    try:
        del lisf1.binding['Catchments']
        del lisf1.binding['1']
    except KeyError:
        pass
    lisf1.main(lisfloodRoot + "dummy.txt", templatePathPreRun, '-v', '-t')
    del lisf1.binding['Catchments']
    del lisf1.binding['1']
    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 to avoid overwriting the bestrun avgdis.end.nc
    cmd = "cp " + path_subcatch + "/out/avgdis" + run_rand_id + ".end.nc " + path_subcatch + "/out/avgdis" + run_rand_id + ".end.nc.bak"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()
    cmd = "cp " + path_subcatch + "/out/lzavin" + run_rand_id + ".end.nc " + path_subcatch + "/out/lzavin" + run_rand_id + ".end.nc.bak"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()
    lisf1.main(lisfloodRoot + "dummy.txt", templatePathRun, '-v', '-t')
    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 restore the backup
    cmd = "mv " + path_subcatch + "/out/avgdis" + run_rand_id + ".end.nc.bak " + path_subcatch + "/out/avgdis" + run_rand_id + ".simulated_best.end.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()
    cmd = "mv " + path_subcatch + "/out/lzavin" + run_rand_id + ".end.nc.bak " + path_subcatch + "/out/lzavin" + run_rand_id + ".simulated_best.end.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()
    cmd = "rm " + path_subcatch + "/out/avgdis" + run_rand_id + ".end.nc " + path_subcatch + "/out/lzavin" + run_rand_id + ".end.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()

    os.chdir(currentdir)
    print("BLAAT")
    cmd = "ls -lrt " + path_subcatch + "/out"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()

    # Save simulated streamflow to disk
    Qsim_tss = os.path.join(path_subcatch, "out", 'dis' + run_rand_id + '.tss')
    timer = 0
    simulated_streamflow = pandas.read_csv(Qsim_tss,sep=r"\s+",index_col=0,skiprows=4,header=None,skipinitialspace=True)
    simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan
    Qsim = simulated_streamflow[1].values
    print(">> Saving \"best\" simulated streamflow (streamflow_simulated_best.csv)")
    Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(ForcingStart, periods=len(Qsim), freq='6H'))
    Qsim.to_csv(os.path.join(path_subcatch,"streamflow_simulated_best.csv"),',',header="")
    try:
        os.remove(os.path.join(path_subcatch, "out", 'streamflow_simulated_best.tss'))
    except:
        pass
    os.rename(Qsim_tss, os.path.join(path_subcatch,"out",'streamflow_simulated_best.tss'))

    # DD Modification for efas-ec 2.12.6
    # Save instantaneous discharge in channel to disk to use as inflow for the next catchment
    chanQ_tss = os.path.join(path_subcatch, "out", 'chanq' + run_rand_id + '.tss')
    timer = 0
    chanQpd = pandas.read_csv(chanQ_tss,sep=r"\s+",index_col=0,skiprows=4,header=None,skipinitialspace=True)
    chanQpd[1][chanQpd[1]==1e31] = np.nan
    chanQ = chanQpd[1].values
    print(">> Saving \"inflow\")")
    chanQ = pandas.DataFrame(data=chanQ, index=pandas.date_range(ForcingStart, periods=len(chanQ), freq='6H'))
    chanQ.to_csv(os.path.join(path_subcatch, "chanq_simulated_best.csv"), ',', header="")
    try:
        os.remove(os.path.join(path_subcatch, "out", 'chanq_simulated_best.tss'))
    except:
        pass
    os.rename(chanQ_tss, os.path.join(path_subcatch, "out", 'chanq_simulated_best.tss'))

if __name__=="__main__":
    main()
