This is a collection of scripts for calibrating the Lisflood model.

This Readme text explains how to use the scripts with the LISFLOOD hydrological model against streamflow observations in an automated manner for multiple catchments. The scripts loop through the catchments in ascending order of catchment area, calibrating LISFLOOD for each \interstation region" (i.e., the catchment area excluding the area of upstream catchments) using a genetic algorithm (https://github.com/DEAP/deap).  

The calibration tool was created by Hylke Beck 2014 (JRC, Princeton) hylkeb@princeton.edu 
Modified by Feyera Aga Hirpa in 2015 (JRC) feyera.hirpa@ouce.ox.ac.uk 
Modified by Valerio Lorini (valerio.lorini@ec.europa.eu) and Alfieri Lorenzo (lorenzo.alfieri@ec.europa.eu) in 2018
The submodule Hydrostats was created 2011 by Sat Kumar Tomer (modified by Hylke Beck) 

The calibration procedure consists of several Python scripts to be run consecutively   
<details><summary>Before you start.</summary>

Software required
- pcraster > 0.41 http://pcraster.geo.uu.nl/
- Python 2.7
- postscript (for figures, you can avoid it though) 

Python packages
- netCDF4  https://pypi.org/project/netCDF4/
- pcraster http://pcraster.geo.uu.nl/
- pandas  (pip install ...)
- numpy (pip install ...)
- matplotlib (pip install ...)
- deap (https://deap.readthedocs.io/en/master/installation.html)

You MUST run them in the correct order because they're outcomes depend on each other.

Order:
- CAL_1_CAL_VAL_PERIODS
- CAL_2_PREP_MISC_REMOVE_SMALL_CATCH
- CAL_3_PREP_MISC
- CAL_4_SPREAD_WORKLOAD
- CAL_5_PREP_FORCING
- CAL_6_CUTMAPS_LAUNCH (that then runs CAL_6_CUT_MAPS)
- CAL_7_PERFORM_CAL
- CAL_8_CLEAN_SUBCATCH_DIRS
- CAL_9_PARAMETER_MAPS
- CAL_10_COMPUTE_STATS_AND_FIGURES_TIME_SERIES

### N.B. Run the scripts from the folder Root using absolute path , else if your write only relative path the job will not work when submitted to the queue.

this ... is OK  
python /PATH_TO_CALIBRATION_SCRIPTS/CAL_5_PERFORM_CAL.py   /PATH_TO_CALIBRATION_SCRIPTS/settings_calibration.txt   /PATH_TO_CALIBRATION_SCRIPTS/CatchmentsToProcess_01.txt 

this in NOT OK  
python  ./CAL_8_COMPUTE_STATS_AND_FIGURES_TIME_SERIES.py ./settings_reforecasts_9616_testColombia.txt ./CatchmentsToProcess_01.txt  
</details>

<details><summary>Settings File</summary>  

**Root** = /PATH_TO_YOUR_CALIBRATION/  
**ForcingStart** = 1/1/1986 00:00  /* Starting of Meteo Forcings Precipitation Evapotranspiration TAvg  
**ForcingEnd** = 31/12/2017 00:00  /* Ending of Meteo Forcings Precipitation Evapotranspiration TAvg  
**SubsetMeteoData** = 0 
**WarmupDays** = 366  /* Number of days for the Warmup period of the Model  
**MinQlength** = 4 /* Catchments with streamflow records shorter than the number of years specifed by MinQlength in settings.txt will not be processed 
**No_of_calibration_lists** = 21 /* Number of lists of catchments to porcess in parallel. i.e. If one agrees 10 nodes for running the calibration, a maximum of 10 (or less depending on direct links between subcatchments) lists will be generated with the name CatchmentsToProcess_XX.txt and 10 will be the maximum number of jobs submitted at the same time.  
**MaxPercArea**=0.1  

[CSV]  
**Qgis** = /FLOODS/lisflood/CalibrationTest/Qgis_worldOutlet_Calib20180810_AllUps_w_Scores.csv /* File containing metadata of Stations available with observation  
**Qtss** = /FLOODS/lisflood/CalibrationTest/Qts_World_2018_08.csv /* Observed data   

[Path]  
**Temp** = %(Root)s/temp  
**Result** = %(Root)s/result  
**Templates** = %(Root)s/templates       
**SubCatchmentPath** = %(Root)s/catchments  
**ParamRanges** = %(Root)s/ParamRanges_LISFLOOD.csv   /* Values range for parameters to calibrate   
**CatchmentDataPath** = /FLOODS/lisflood/CalibrationTest/static_data  /* static maps for lisflood model (landuse wateruse area ldd etc)  
**MeteoData** = /FLOODS/glofas/meteo/ERA5/ /* path to netcdf forcing data  
**PCRHOME** = /ADAPTATION/usr/anaconda2/bin/ /* path to pcraster binaries  
**PYTHONCMD** = /ADAPTATION/usr/anaconda2/bin/python /* path to python executable  (in case of several versions)  

[Templates]  
**LISFLOODSettings** = %(Root)s/templates/settings_LF.xml /* Settings for Lisflood Model see documentation on Lisflood Repo  
**RunLISFLOOD** = %(Root)s/templates/runLF_linux_cut.sh  /* Script for launching PreRun and Run for every parameters combination during genetic algorithm runs   

[DEAP]   /* for socumentation refers to link provided above)  
**use_multiprocessing** = 1 /* Flag for using multiprocessing, meaning running several lisflood runs on several cores (each using 1 core)  
**ngen** = 16 /* number of MAX generation to run  
**mu** = 16  /* initial population  
**lambda_** = 32 /* size of generation of offsprings  

</details>


<details><summary>python CAL_1_CAL_VAL_PERIODS.py settings.txt</summary>
The 1rst script reads the settings  (settings.txt in this case) and the Qgis and Qtss specifed in settings.txt. Computes calibration and validation periods based on the available streamflow data. Catchments with streamflow records shorter than the number of years specifed by MinQlength in settings.txt will not be processed. If the record length is twice MinQlength, the available streamflow record is split into equally long validation and calibration parts. If the record length is less than twice MinQlength, MinQlength is used for calibration and the remaining part for validation. In all cases, the first part of the record is used for validation and the second part for calibration. If necessary, the calibration and validation periods can be changed manually by editing Qgis2.csv. However, be sure to use the correct
date format (DD/MM/YYYY).
</details>

<details><summary>python CAL_2_PREP_MISC_REMOVE_SMALL_CATCH.py settings.txt</summary>
This script reads the settings file (settings.txt in this case) and subsequently the Qgis and Qtss files specified in settings.txt. Stations from the Qgis list are first associated to the corresponding model river network, to identify the upstream/downstream relation among stations lying in the same river basin. 
  Then, if the parameter MaxPercArea is set to a number larger than 0, the script eliminates from the calibration list all stations lying within MaxPercArea times the upstream area of each river station. If MaxPercArea=0.1 (default value), all river stations with upstream area up to 10% larger than that of the upstream station will be removed. For example, if 5 stations are available along the same river, with upstream area respectively of A1=100, A2=104, A3=107, A4=111, A5=118 km^2, the script will remove the 2nd, 3rd and the 5th station, leaving the 1st and the 4th in the calibration list. The idea behind this is to avoid calibrating clusters of stations, where the downstream ones bring little benefit and often assumes anomaluos calibrated parameter values, as they are heavily affected by the simulated inflow of the upstream station. 
</details>

<details><summary>python CAL_3_PREP_MISC.py settings.txt</summary>
The second script reads the settings file (settings.txt in this case) and subsequently the Qgis and Qtss files specified in settings.txt. Computes maps with station locations (outlet.map), interstation regions (interstation regions.map),sampling frequency (sampling frequency.map), and in ow locations (inlets.map).  
In addition, generates a CSV file with direct station linkages (direct links.csv), and a CSV file listing the catchment area (in number of pixels) and the numeric portion of the catchment identifier (Qgis2.csv).  
</details>


<details><summary>python CAL_4_SPREAD_WORKLOAD.py settings.txt</summary>  
This script subdivides the modeling domain to allow running the calibration in parallel. Considered the time needed for running the calibration, we opted for spreading the total amount of catchments on multiple lists that can be used to spread the run of the calibration over several medium (PCs , nodes on grid engine, single PC). 
  
Each list is independent from the others, this let you decide where to run each list. In our case we 
  datetime.datetime.strptime(row['Cal_Start'],"%d/%m/%Y %H:%M")
        f=open(Root+'/runLF.sh','w')
        print 'open'
        cmd = python_cmd+' '+Root+'/cal_single_objfun.py '+sys.argv[1]+' '+str(index)
        f.write("#!/bin/sh \n")
        f.write(cmd)
        f.close()
        cmd="qsub -l nodes=1:ppn=32 -q high -N LF_calib "+Root+"/runLF.sh"
        
        timerqsub = 0
        
        while not int(subprocess.Popen('qstat | grep LF_calib | wc -l',shell=True,stdout=subprocess.PIPE).stdout.read()) < int(nmax) and timerqsub<=72000:
            time.sleep(1)
            timerqsub+=1
        
        if timerqsub>72000:
            print '20 hrs waiting for job submission, something's wrong')
            raise Exception('too much time')
        
        print ">> Calling \""+cmd+"\""
        os.system(cmd)
  
</details>

<details><summary>python CAL_5_PREP_FORCING.py settings.txt CatchmentsToProcess_XX.txt</summary>  
This script should be run on each PC, so the entire directory should be copied to each PC. The script reads the settings file (settings.txt in this case) and the Qgis2.csv file that was generated by the previous script. Loops through the stations in CatchmentsToProcess X.txt, makes directories for each station, and produces mask and station maps subsetted to the interstation regions of the respective catchments.  
</details>

<details><summary>python CAL_6_CUTMAPS_LAUNCH.py settings.txt CatchmentsToProcess_XX.txt</summary>  
This script should be run on each PC, so the entire directory should be copied to each PC. The script reads the settings file (settings.txt in this case) and the Qgis2.csv file that was generated by the previous script. Loops through the stations in CatchmentsToProcess X.txt, makes directories for each station, and produces mask and station maps subsetted to the interstation regions of the respective catchments.  
</details>


<details><summary>python CAL_7_PERFORM_CAL.py settings.txt CatchmentsToProcess_XX.txt</summary>  
This script should be run on each PC. It reads the settings file (settings.txt in this case), the direct links.csv and Qgis2.csv files, and the Qtss file specified in settings.txt. Loops through the stations listed in CatchmentsToProcess X.txt (in ascending order of catchment area), creates an inflow.tss file if there are directly linked upstream stations, and runs the script cal single objfun.py settings.txt x" (use if there is a single objective function) or cal multiple objfun.py settings.txt x" (use if there are multiple objective functions), where x denotes the numeric portion of the station identifier as listed in Qgis2.csv.  
The **cal_single_objfun.py**  script performs the calibration for each station using the Non-dominated Sorting Genetic Algorithm-II (NSGA-II) multi-objective genetic algorithm (Deb et al., 2002). Any objective function can be used by modifying the scripts. The LISFLOOD runs are distributed to the available cores if use multiprocessing is set to one in the settings.txt file. Outputs for each catchment are CSV files with objective function scores of each LISFLOOD run (runs log.csv), objective function statistics for the Pareto optimal solutions of each generation (front history.csv), final Pareto optimal solutions and corresponding model parameter values (pareto front.csv), and streamflow time series of the \best" run (streamflow simulated best.csv and streamflow simulated best.tss).
</details>  

<details><summary>python CAL_8_CLEAN_SUBCATCH_DIRS.py settings.txt CatchmentsToProcess_XX.txt</summary>  
This script should be run on each PC. Reads the settings file (settings.txt in this case), loops through the catchments, and deletes unnecessary files created with each run (e.g., prerun and run XML files and lzavin maps). After this script the files from each PC should be copied to a single PC on which you run the following scripts.
</details>

<details><summary>python CAL_9_PARAMETER_MAPS.py settings.txt</summary>  
This script reads the settings file (settings.txt in this case) and produces, for each parameter, a map with the calibrated parameter values assigned to the corresponding interstation regions. Ungauged regions are assigned the default parameter values taken from the CSV file specified in the settings file.
</details>

<details><summary>python CAL_10_COMPUTE_STATS_AND_FIGURES_TIME_SERIES.py settings.txt</summary>  
Reads the settings file (settings.txt in this case) and loops through the catchments to create figures of the calibration and validation results. 
</details>

<details><summary>Tips & Tricks</summary>  
To ensure that LISFLOOD runs as fast as possible, check that the meteorological forcing NetCDF fles use chunking, are uncompressed, and do not have time as an unlimited dimension. You can check this by opening the files using Panoply (http://www.giss.nasa.gov/tools/panoply/). In addition, the forcing data should be stored on the local disk instead of over the network.  
</details>
