
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import HydroStats
import array
import random
import numpy as np
import datetime
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import pandas
import re
import pdb
import multiprocessing
import time
import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath
from ConfigParser import SafeConfigParser
import glob
from scoop import futures
from subprocess import Popen, PIPE
import stat


########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])

parser = SafeConfigParser()
parser.read(iniFile)

ForcingStart = parser.get('DEFAULT','ForcingStart')  # Start of forcing
ForcingEnd = parser.get('DEFAULT','ForcingEnd')  # Start of forcing

WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')
ParamRangesPath = parser.get('Path','ParamRanges')
MeteoDataPath = parser.get('Path','MeteoData')

path_temp = parser.get('Path', 'Temp')
path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
path_result = parser.get('Path', 'Result')

Qtss_csv = parser.get('CSV', 'Qtss')

LISFLOODSettings_template = parser.get('Templates','LISFLOODSettings')
RunLISFLOOD_template = parser.get('Templates','RunLISFLOOD')

use_multiprocessing = int(parser.get('DEAP','use_multiprocessing'))
ngen = int(parser.get('DEAP','ngen'))
mu = int(parser.get('DEAP','mu'))
lambda_ = int(parser.get('DEAP','lambda_'))

pcraster_path = parser.get('Path', 'PCRHOME')

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
    config[execname] = getPCrasterPath(pcraster_path,execname)

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]


########################################################################
#   Preparation for calibration
########################################################################

stationdata = pandas.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=True)

row = stationdata.loc[int(sys.argv[2])]

path_subcatch = os.path.join(SubCatchmentPath,row['ID'])

# For some reason this version of LISFLOOD doesn't work with outlet map, 
# hence have to supply gauge coordinates
gaugeloc_txt = os.path.join(path_subcatch,"maps","gaugeloc.txt")
f = file(gaugeloc_txt,"r")
for line in f.readlines():
    (X,Y,value) = line.split()
gaugeloc = str(float(X))+" "+str(float(Y))
print gaugeloc 

# Check if inflow.tss file is present
# inflow.tss is created using CAL_5_PERFORM_CAL.py
inflow_tss = os.path.join(path_subcatch,"inflow","inflow.tss")
inflowflag = str(0)
if os.path.isfile(inflow_tss):
    inflowflag = str(1)

# Check if lakes or reservoirs are present
# Lakes map must be called lakes.map and reservoir map res.map!!!
#simulateLakes = str(0)
#simulateReservoirs = str(0)
#tmp_txt = os.path.join(path_temp,"tmp.txt")
#pcrasterCommand(map2col + " F0 F1"  , {"F0": os.path.join(path_subcatch,"maps","res.map"), "F1":tmp_txt})
#if os.path.getsize(tmp_txt)!=0: # Check if empty
#	simulateReservoirs = str(1)
#pcrasterCommand(map2col + " F0 F1"  , {"F0": os.path.join(path_subcatch,"maps","lakes.map"), "F1":tmp_txt})
#if os.path.getsize(tmp_txt)!=0: # Check if empty
#	simulateLakes = str(1)

# Compute the time steps at which the calibration should start and end
#Val_Start = datetime.datetime.strptime(row['Val_Start'],"%m/%d/%Y")
#Val_End = datetime.datetime.strptime(row['Val_End'],"%m/%d/%Y")
#al_Start = str(datetime.datetime.strptime(row['Cal_Start'],"%d/%m/%Y %H:%M"))
Cal_Start = row['Cal_Start']
#al_End = str(datetime.datetime.strptime(row['Cal_End'],"%d/%m/%Y %H:%M"))
Cal_End = row['Cal_End']
#Val_Start_Step = (Val_Start-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
#Val_End_Step = (Val_End-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
#Cal_Start_Step = (Cal_Start-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
#Cal_End_Step = (Cal_End-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!
#Forcing_End_Step = (ForcingEnd-ForcingStart).days+1 # For LISFLOOD, not for indexing in Python!!!

# Load xml and .bat template files
print 'ao',(os.path.join('templates',RunLISFLOOD_template))
f = open(os.path.join('templates',RunLISFLOOD_template),"r")

template_bat = f.read()
f.close()

f = open(os.path.join('templates',LISFLOODSettings_template),"r")
template_xml = f.read()
f.close()

# Load paramranges file
ParamRanges = pandas.read_csv(ParamRangesPath,sep=",",index_col=0)

# Load observed streamflow
streamflow_data = pandas.read_csv(Qtss_csv,sep=",", parse_dates=True, index_col=0)
observed_streamflow = streamflow_data[row['ID']]
observed_streamflow = observed_streamflow[ForcingStart:ForcingEnd] # Keep only the part for which we run LISFLOOD


########################################################################
#   Function for running the model, returns objective function scores
########################################################################

def RunModel(Individual):

    # Convert scaled parameter values ranging from 0 to 1 to usncaled parameter values
    Parameters = [None] * len(ParamRanges)
    for ii in range(0,len(ParamRanges)):
        Parameters[ii] = Individual[ii]*(float(ParamRanges.iloc[ii,1])-float(ParamRanges.iloc[ii,0]))+float(ParamRanges.iloc[ii,0])

    # Note: The following code must be identical to the code near the end where LISFLOOD is run
    # using the "best" parameter set. This code:
    # 1) Modifies the settings file containing the unscaled parameter values amongst other things
    # 2) Makes a .bat file to run LISFLOOD
    # 3) Runs LISFLOOD and loads the simulated streamflow
    # Random number is appended to settings and .bat files to avoid simultaneous editing
    run_rand_id = str(int(random.random()*10000000000)).zfill(12)
    template_xml_new = template_xml
    for ii in range(0,len(ParamRanges)):
        template_xml_new = template_xml_new.replace("%"+ParamRanges.index[ii],str(Parameters[ii]))
    template_xml_new = template_xml_new.replace('%gaugeloc',gaugeloc) # Gauge location
    #template_xml_new = template_xml_new.replace('%ForcingStart',ForcingStart.date().strftime('%d/%m/%Y %H:%M')) # Date of forcing start
    template_xml_new = template_xml_new.replace('%CalStart', Cal_Start) # Date of Cal starting
        #print 'calstartsingle', datetime.datetime.strptime(row['Cal_Start'],"%m/%d/%Y").strftime('%d/%m/%Y %H:%M')
    #template_xml_new = template_xml_new.replace('%StepStart',str(Cal_Start_Step)) # Time step of forcing at which to start simulation
    template_xml_new = template_xml_new.replace('%CalEnd',Cal_End) # Time step of forcing at which to end simulation
    template_xml_new = template_xml_new.replace('%CatchmentDataPath',CatchmentDataPath) # Directory with forcing for the whole region
    template_xml_new = template_xml_new.replace('%SubCatchmentPath',path_subcatch)
    template_xml_new = template_xml_new.replace('%MeteoDataPath',MeteoDataPath)
    template_xml_new = template_xml_new.replace('%run_rand_id',run_rand_id)
    template_xml_new = template_xml_new.replace('%inflowflag',inflowflag)
    #template_xml_new = template_xml_new.replace('%simulateLakes',simulateLakes)
    #template_xml_new = template_xml_new.replace('%simulateReservoirs',simulateReservoirs)
    template_xml_new2 = template_xml_new
    template_xml_new = template_xml_new.replace('%InitLisflood',"1")
    f = open(os.path.join(path_subcatch,LISFLOODSettings_template[:-4]+'-PreRun'+run_rand_id+'.xml'), "w")
    f.write(template_xml_new)
    f.close()
    template_xml_new2 = template_xml_new2.replace('%InitLisflood',"0")
    f = open(os.path.join(path_subcatch,LISFLOODSettings_template[:-4]+'-Run'+run_rand_id+'.xml'), "w")
    f.write(template_xml_new2)
    f.close()
    template_bat_new = template_bat
    template_bat_new = template_bat_new.replace('%prerun',LISFLOODSettings_template[:-4]+'-PreRun'+run_rand_id+'.xml')
    template_bat_new = template_bat_new.replace('%run',LISFLOODSettings_template[:-4]+'-Run'+run_rand_id+'.xml')
    f = open(os.path.join(path_subcatch,RunLISFLOOD_template[:-4]+run_rand_id+'.bat'), "w")
    f.write(template_bat_new)
    f.close()
        
    currentdir = os.getcwd()
    print currentdir
    os.chdir(path_subcatch)
    print "path_subcatch",path_subcatch
    print RunLISFLOOD_template[:-4]+run_rand_id+'.bat'
    shutil.move(RunLISFLOOD_template[:-4]+run_rand_id+'.bat', path_subcatch)
    st = os.stat(path_subcatch+'/runLF_linu'+run_rand_id+'.bat')
    os.chmod(path_subcatch+'/runLF_linu'+run_rand_id+'.bat', st.st_mode | stat.S_IEXEC)
    print path_subcatch+'/runLF_linu'+run_rand_id+'.bat'
    p = Popen(path_subcatch+'/runLF_linu'+run_rand_id+'.bat', stdout=PIPE, stderr=PIPE, bufsize=16*1024*1024)
    output, errors = p.communicate()
    f = open("log"+run_rand_id+".txt",'w')
    content = "OUTPUT:\n"+output+"\nERRORS:\n"+errors
    f.write(content)
    f.close()
    os.chdir(currentdir)
    Qsim_tss = os.path.join(path_subcatch,"out",'dis'+run_rand_id+'.tss')
    if os.path.isfile(Qsim_tss)==False:
        print "run_rand_id: "+str(run_rand_id)
        raise Exception("No simulated streamflow found. Probably LISFLOOD failed to start? Check the log files of the run!")
    simulated_streamflow = pandas.read_table(Qsim_tss,sep=r"\s+",index_col=0,skiprows=4,header=None,skipinitialspace=True)
    simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan
    Qobs = observed_streamflow[Cal_Start:Cal_End].values+0.001
    Qsim = simulated_streamflow[1].values+0.001
    #Qsim = Qsim[Cal_Start_Step-1:Cal_End_Step+1]
    if len(Qobs) != len(Qsim):
        raise Exception("run_rand_id: "+str(run_rand_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

    # Compute objective function score
    KGE = HydroStats.KGE(s=Qsim,o=Qobs,warmup=WarmupDays)

    print "   run_rand_id: "+str(run_rand_id)+", KGE: "+"{0:.3f}".format(KGE)

    with open(os.path.join(path_subcatch,"runs_log.csv"), "a") as myfile:
        myfile.write(str(run_rand_id)+","+str(KGE)+"\n")

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

if __name__ == "__main__":

    t = time.time()

    if use_multiprocessing==True:
        pool_size = multiprocessing.cpu_count() * 1
        pool = multiprocessing.Pool(processes=pool_size)
        toolbox.register("map", pool.map)
        print pool_size

    #random.seed(64) # Don't use or the ID numbers used in the RunModel function will not be unique

    cxpb = 0.9 # For someone reason, if sum of cxpb and mutpb is not one, a lot less Pareto optimal solutions are produced
    mutpb = 0.1

    population = toolbox.population(n=mu)
    halloffame = tools.ParetoFront()

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    print 'invalid_ind',invalid_ind
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    halloffame.update(population)

    # Loop through the different objective functions and calculate some statistics
    # from the Pareto optimal population
    effmax = np.zeros(shape=(ngen+1,1))*np.NaN
    effmin = np.zeros(shape=(ngen+1,1))*np.NaN
    effavg = np.zeros(shape=(ngen+1,1))*np.NaN
    effstd = np.zeros(shape=(ngen+1,1))*np.NaN
    for ii in range(1):
        effmax[0,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effmin[0,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effavg[0,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effstd[0,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
    gen = 0
    print ">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0])

    # Begin the generational process
    conditions = {"ngen" : False, "StallFit" : False}
    gen = 1

    while not any(conditions.values()):

        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
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
            effmax[gen,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effmin[gen,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effavg[gen,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effstd[gen,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        print ">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0])

# Terminate the optimization after ngen generations
        if gen >= ngen:
          print(">> Termination criterion ngen fulfilled.")
          conditions["ngen"] = True
           
        if gen >= 2:
            if (effmax[gen,0]-effmax[gen-1,0]) < 0.003:
                print(">> Termination criterion no-improvement fulfilled.")
                conditions["StallFit"] = True   
        gen += 1




    # Finito
    if use_multiprocessing == True:
        pool.close()
    elapsed = time.time() - t
    print ">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s"


    ########################################################################
    #   Save calibration results
    ########################################################################

    # Save history of the change in objective function scores during calibration to csv file
    front_history = pandas.DataFrame()
    front_history['gen'] = range(gen)
    print range(gen)
    print effmax[0:gen,0]
    front_history['effmax_R'] = effmax[0:gen,0]
    front_history['effmin_R'] = effmin[0:gen,0]
    front_history['effstd_R'] = effstd[0:gen,0]
    front_history['effavg_R'] = effavg[0:gen,0]
    front_history.to_csv(os.path.join(path_subcatch,"front_history.csv"))
    
    # Compute overall efficiency scores from the objective function scores for the
    # solutions in the Pareto optimal front
    # The overall efficiency reflects the proximity to R = 1, NSlog = 1, and B = 0 %
    front = np.array([ind.fitness.values for ind in halloffame])
    effover = 1 - np.sqrt((1-front[:,0]) ** 2)
    best = np.argmax(effover)

    # Convert the scaled parameter values of halloffame ranging from 0 to 1 to unscaled parameter values
    paramvals = np.zeros(shape=(len(halloffame),len(halloffame[0])))
    paramvals[:] = np.NaN
    for kk in range(len(halloffame)):
        for ii in range(len(ParamRanges)):
            paramvals[kk][ii] = halloffame[kk][ii]*(float(ParamRanges.iloc[ii,1])-float(ParamRanges.iloc[ii,0]))+float(ParamRanges.iloc[ii,0])

    # Save Pareto optimal solutions to csv file
    # The table is sorted by overall efficiency score
    print ">> Saving Pareto optimal solutions (pareto_front.csv)"
    ind = np.argsort(effover)[::-1]
    pareto_front = pandas.DataFrame({'effover':effover[ind],'R':front[ind,0]})
    for ii in range(len(ParamRanges)):
        pareto_front["param_"+str(ii).zfill(2)+"_"+ParamRanges.index[ii]] = paramvals[ind,ii]
    pareto_front.to_csv(os.path.join(path_subcatch,"pareto_front.csv"),',')

    # Select the "best" parameter set and run LISFLOOD for the entire forcing period
    Parameters = paramvals[best,:]

    print ">> Running LISFLOOD using the \"best\" parameter set"
    # Note: The following code must be identical to the code near the end where LISFLOOD is run
    # using the "best" parameter set. This code:
    # 1) Modifies the settings file containing the unscaled parameter values amongst other things
    # 2) Makes a .bat file to run LISFLOOD
    # 3) Runs LISFLOOD and loads the simulated streamflow
    # Random number is appended to settings and .bat files to avoid simultaneous editing
    run_rand_id = str(int(random.random()*10000000000)).zfill(12)
    template_xml_new = template_xml
    for ii in range(0,len(ParamRanges)):
        template_xml_new = template_xml_new.replace("%"+ParamRanges.index[ii],str(Parameters[ii]))
    template_xml_new = template_xml_new.replace('%gaugeloc',gaugeloc) # Gauge location
#	template_xml_new = template_xml_new.replace('%ForcingStart',ForcingStart.date().strftime('%d/%m/%Y')) # Date of forcing start
    template_xml_new = template_xml_new.replace('%CalStart', ForcingStart) # Time step of forcing at which to start simulation
    template_xml_new = template_xml_new.replace('%CalEnd', ForcingEnd) # Time step of forcing at which to end simulation
    template_xml_new = template_xml_new.replace('%CatchmentDataPath',CatchmentDataPath) # Directory with forcing for the whole region
    template_xml_new = template_xml_new.replace('%SubCatchmentPath',path_subcatch) # Directory with data for subcatchments
    template_xml_new = template_xml_new.replace('%MeteoDataPath',MeteoDataPath)
    template_xml_new = template_xml_new.replace('%run_rand_id',run_rand_id)
    template_xml_new = template_xml_new.replace('%inflowflag',inflowflag)
    #template_xml_new = template_xml_new.replace('%simulateLakes',simulateLakes)
    #template_xml_new = template_xml_new.replace('%simulateReservoirs',simulateReservoirs)

    template_xml_new2 = template_xml_new
    template_xml_new = template_xml_new.replace('%InitLisflood',"1")
    f = open(os.path.join(path_subcatch,LISFLOODSettings_template[:-4]+'-PreRun'+run_rand_id+'.xml'), "w")
    f.write(template_xml_new)
    f.close()
    template_xml_new2 = template_xml_new2.replace('%InitLisflood',"0")
    f = open(os.path.join(path_subcatch,LISFLOODSettings_template[:-4]+'-Run'+run_rand_id+'.xml'), "w")
    f.write(template_xml_new2)
    f.close()
    

    f = open(os.path.join(path_subcatch,LISFLOODSettings_template[:-4]+'-Run'+run_rand_id+'.xml'), "w")
    f.write(template_xml_new2)
    f.close()
    template_bat_new = template_bat
    template_bat_new = template_bat_new.replace('%prerun',LISFLOODSettings_template[:-4]+'-PreRun'+run_rand_id+'.xml')
    template_bat_new = template_bat_new.replace('%run',LISFLOODSettings_template[:-4]+'-Run'+run_rand_id+'.xml')
    f = open(os.path.join(path_subcatch,RunLISFLOOD_template[:-4]+run_rand_id+'.bat'), "w")
    f.write(template_bat_new)
    f.close()
        
    currentdir = os.getcwd()
    print currentdir
    os.chdir(path_subcatch)
    print "path_subcatch",path_subcatch
    print RunLISFLOOD_template[:-4]+run_rand_id+'.bat'
    shutil.move(RunLISFLOOD_template[:-4]+run_rand_id+'.bat', path_subcatch)
    st = os.stat(path_subcatch+'/runLF_linu'+run_rand_id+'.bat')
    os.chmod(path_subcatch+'/runLF_linu'+run_rand_id+'.bat', st.st_mode | stat.S_IEXEC)
    print path_subcatch+'/runLF_linu'+run_rand_id+'.bat'
    p = Popen(path_subcatch+'/runLF_linu'+run_rand_id+'.bat', stdout=PIPE, stderr=PIPE, bufsize=16*1024*1024)
    output, errors = p.communicate()
    f = open("log"+run_rand_id+".txt",'w')
    content = "OUTPUT:\n"+output+"\nERRORS:\n"+errors
    f.write(content)
    f.close()
    
    
    os.chdir(currentdir)
    Qsim_tss = os.path.join(path_subcatch, "out", 'dis' + run_rand_id + '.tss')

    simulated_streamflow = pandas.read_table(Qsim_tss,sep=r"\s+",index_col=0,skiprows=4,header=None,skipinitialspace=True)
    simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan
    Qsim = simulated_streamflow[1].values

    # Save simulated streamflow to disk
    print ">> Saving \"best\" simulated streamflow (streamflow_simulated_best.csv)"
    Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(ForcingStart, periods=len(Qsim), freq='d'))
    Qsim.to_csv(os.path.join(path_subcatch,"streamflow_simulated_best.csv"),',',header="")
    try: os.remove(os.path.join(path_subcatch,"out",'streamflow_simulated_best.tss'))
    except: pass
    os.rename(Qsim_tss, os.path.join(path_subcatch,"out",'streamflow_simulated_best.tss'))

    # Delete all .xml, .bat, .tmp, and .txt files created for the runs
    for filename in glob.glob(os.path.join(path_subcatch,"*.xml")):
        os.remove(filename)
    for filename in glob.glob(os.path.join(path_subcatch,"*.bat")):
        os.remove(filename)
    for filename in glob.glob(os.path.join(path_subcatch,"*.tmp")):
        os.remove(filename)
    #for filename in glob.glob(os.path.join(path_subcatch,"*.txt")):
    #	os.remove(filename)
    for filename in glob.glob(os.path.join(path_subcatch,"out","lzavin*.map")):
        os.remove(filename)
    for filename in glob.glob(os.path.join(path_subcatch,"out","dis*.tss")):
        os.remove(filename)
