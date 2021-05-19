import sys
import os
import pandas
import numpy as np


def pcrasterCommand(cmd = '', files = None, workdir = None, var2export = {},DebugMe = False):
    """ BPAN: this is copied from simone's script.
    His version is not working on windows, so I kicked out the environment parts, and just replaced by a os.system
    Eventually this should point back to Simones routines."""
    '''this is a generic function that runs generic pcraster commands; at the moment pcrcalc and legend.
    cmd is a string that contains pcraster operators, filenames and aliases to be then
    substituted.

    files can be a single filename, a list of filenames or a dictionary that contains
    couples of keys/values where the key is the alias and the value is the filename

    This function takes care of replacing the aliases inside the cmd with the correct
    filenames, it adds the missing characters needed for a correct interpretation
    of the shell of every filename.
    eg:
    /home/username/this is a filename >> "/home/username/this is a filename"

    Another thing is that it sources the pcraster environment file before executing any command
    Then, as in the shellcmd library, there are two extra parameters to define a workdir, that
    makes enter into a certain directory before calling the command, and var2export that lets
    define variables to be set in the environment before running the real command. Note that
    when an external program, like pcrcalc, is called from python a completely new shell is
    created from zero, so it's sometimes necessary to create a particular starting environment.

    (**) in the code means needed to avoid cross references, see class TestUsingPcrcalc test 05

    BPAN: DEBUG will show all pcraster commands to the screen, DebugMe only the one where variable is passed as True
    '''
    DEBUG = False
    if DebugMe: print(cmd)
    if files is not None:
        cmd_orig = cmd
        from random import choice #(**)
        import string
        charset = string.punctuation

        aliases = []
        random_aliases = [] # (**)
        realnames = []

        if type(files) is str: files = [files]
        if type(files) is list:
            for i in range(len(files)):
                aliases.append('F'+str(i))
                random_alias = ''
                for j in range(6):
                    random_alias += choice(charset)
                random_aliases.append(random_alias)
                realnames.append(files[i])

        elif type(files) is dict:
            for alias, realname in files.items():
                aliases.append(alias)
                random_alias = ''
                for j in range(6):
                    random_alias += choice(charset)
                random_aliases.append(random_alias)
                realnames.append(realname)

        else:
            raise Exception(str(type(files))+' :unhandled format')

        #replace alias with random_alias **
        for i in range(len(aliases)):
            alias = aliases[i]
            random_alias = random_aliases[i]
            if alias not in cmd:
                err = "Can't find "+alias+'\n'
                err += cmd_orig+'\n'
                err += cmd
                raise Exception(err)
            cmd = cmd.replace(alias, random_alias)

        #replace random_alias with filenames **
        for i in range(len(random_aliases)):
            random_alias = random_aliases[i]
            realname = realnames[i]
            if random_alias not in cmd:
                err = "Can't find "+random_alias+'\n'
                err += cmd_orig+'\n'
                err += cmd+'\n'
                err += str(aliases)+'\n'
                err += str(random_aliases)+'\n'
                err += str(files)
                raise Exception(err)
            cmd = cmd.replace(random_alias, '"'+realname+'"')

    if (sys.platform.upper()=="WIN32"):
        # on windows, get rid of these ennoying single quotes that pcraster doesn't like.
        cmd = cmd.replace("'","")
        cmd = cmd.replace('/', '\\')

    if DEBUG or DebugMe: print("PCRCOMMAND => " + cmd)
    if DEBUG or DebugMe: executedCmd = os.system(cmd)



    ####else:     executedCmd = os.system(cmd + " 2>>stdout.txt") # COMMENTED OUT BY HYLKE BECAUSE PREVENTS RUNNING IN PARALLEL
    else: executedCmd = os.system(cmd)
    if DEBUG or DebugMe: print("END")

    return cmd

def getPCrasterPath(pcraster_path,settingsFile,alias = ""):
  
    '''get the path for a pcrastercommand'''
    alias = alias.strip().lower()
    #pcraster_path = PCRHOME
    # Checking platform
    if (sys.platform.upper()=="WIN32"):
        execsuf = ".exe"
    #    pcraster_path = "C:/PcRaster/apps"
    else:
        execsuf = ""
        ##pcraster_path = c.get("pcrasterdir")       # directory where pcrastercommands can be found
    #pcraster_path = "/ADAPTATION/usr/anaconda2/bin/"
    if alias=="pcrcalc": return os.path.join(pcraster_path,"pcrcalc" + execsuf)
    elif alias=="map2asc": return os.path.join(pcraster_path,"map2asc" + execsuf)
    elif alias=="asc2map": return os.path.join(pcraster_path,"asc2map" + execsuf)
    elif alias=="col2map": return os.path.join(pcraster_path,"col2map" + execsuf)
    elif alias=="map2col": return os.path.join(pcraster_path,"map2col" + execsuf)
    elif alias=="mapattr": return os.path.join(pcraster_path,"mapattr" + execsuf)
    elif alias=="resample": return os.path.join(pcraster_path,"resample" + execsuf)
    else:
        PCRpaths = { "pcrcalc": os.path.join(pcraster_path,"pcrcalc" + execsuf),
                     "map2asc": os.path.join(pcraster_path,"map2asc" + execsuf),
                     "asc2map": os.path.join(pcraster_path,"asc2map" + execsuf),
                     "col2map": os.path.join(pcraster_path,"col2map" + execsuf),
                     "map2col": os.path.join(pcraster_path,"map2col" + execsuf),
                     "mapattr": os.path.join(pcraster_path,"mapattr" + execsuf),
                     "resample": os.path.join(pcraster_path,"resample" + execsuf)}
        return PCRpaths
    return



def create_gauge_loc(cfg, path_subcatch):
    # For some reason this version of LISFLOOD doesn't work with outlet map,
    # hence have to supply gauge coordinates
    gaugeloc_txt = os.path.join(path_subcatch, "maps", "gaugeloc.txt")
    with open(gaugeloc_txt,"r") as f:
        for line in f.readlines():
            (X,Y,value) = line.split()
    gaugeloc = str(float(X))+" "+str(float(Y))

    pcrasterCommand(cfg.pcraster_cmd['map2col'] + " F0 F1"  , {"F0": os.path.join(path_subcatch, "maps", "outlet.map"), "F1":gaugeloc_txt})

    return gaugeloc

def prepare_inflows(cfg, path_subcatch, index):

    # Copy simulated streamflow from upstream catchments
    # Change inlet map by replacing the numeric ID's with 1, 2, ...
    print("Upstream station(s): ")
    direct_links = pandas.read_csv(os.path.join(cfg.path_result, "direct_links.csv"), sep=",", index_col=0)
    #inflow_tss is created according to the cal_start cal_end parameyters, script removes steps before and after and it reindex the steps
    
    inflow_tss = os.path.join(path_subcatch, "inflow", "chanq.tss")
    #inflow_tss_lastrun is for when after the optimal combination of parameters is found , when we run the full forcing period
    inflow_tss_last_run = os.path.join(path_subcatch, "inflow", "chanq_last_run.tss")
    
    try: del big_one
    except: pass
    try: 
        os.remove(inflow_tss)
        os.remove(inflow_tss_last_run)
    except: pass
    upstream_catchments = [int(i) for i in direct_links.loc[index].values if not np.isnan(i)]
    cnt = 1
    subcatchinlets_map = os.path.join(path_subcatch, "inflow", "inflow.map")
    # subcatchinlets_new_map = os.path.join(path_subcatch,"inflow","inflow_new.map")
    subcatchinlets_cut_map = os.path.join(path_subcatch, "inflow", "inflow_cut.map")
    smallsubcatchmask_map = os.path.join(path_subcatch, "maps", "masksmall.map")
    
    # pcrasterCommand(pcrcalc + " 'F0 = F1*0.0'", {"F0":subcatchinlets_new_map,"F1":subcatchinlets_map})
    header = ""
    for subcatchment in upstream_catchments:
        
        subcatchment = str(subcatchment)

        print(subcatchment+" ")
                        
        Qsim_tss = os.path.join(cfg.subcatchment_path, subcatchment, "out", "chanq_simulated_best.tss")
                
        if not os.path.exists(Qsim_tss) or os.path.getsize(Qsim_tss) == 0:
            raise Exception("ERROR: Missing " + Qsim_tss)

        try:
            # DD The shift_time.days is not correctly read for 6-hourly. Using time stamps to make it timesteps invariant
            simulated_streamflow_tmp = pandas.read_csv(Qsim_tss, sep=r"\s+", index_col=False, skiprows=4, header=None, usecols=[1])
            simulated_streamflow_tmp.index = pandas.date_range(cfg.forcing_start, periods=len(simulated_streamflow_tmp), freq='6H')
            # DD comment the following line if you want to make the inflow the complete period
            # simulated_streamflow_tmp = simulated_streamflow_tmp.loc[datetime.datetime.strptime(row['Cal_Start'], "%d/%m/%Y %H:%M"):datetime.datetime.strptime(row['Cal_End'], '%d/%m/%Y %H:%M')]
            simulated_streamflow_tmp.index = [i+1 for i in range(len(simulated_streamflow_tmp))]
            simulated_streamflow_lastrun = pandas.read_csv(Qsim_tss,sep=r"\s+",index_col=0,skiprows=4,header=None)
        except:
            print("Could not find streamflow_simulated_best.tss for upstream catchment "+subcatchment+", hence cannot run this catchment...")
            raise Exception("Stopping...")
                
        simulated_streamflow = simulated_streamflow_tmp
        print('got it')
        if cnt==1: 
            big_one = simulated_streamflow  # type: object
            big_one_lastrun = simulated_streamflow_lastrun
        else:
            big_one[str(cnt)] = simulated_streamflow.values
            big_one_lastrun[str(cnt)] = simulated_streamflow_lastrun.values
#            if cnt==1: big_one = simulated_streamflow  # type: object
#            else: big_one[str(cnt)] = simulated_streamflow.values
        # DD don't need this as it causes inflow points to be deleted in inflow.py
#             numeric_only = re.compile(r'[^\d.]+')
        # hhh = str(int(numeric_only.sub('',subcatchment)))
        # pcrasterCommand(pcrcalc + " 'F0 = F0+scalar(F1=="+hhh+")*"+str(cnt)+"'", {"F0": subcatchinlets_new_map,"F1":subcatchinlets_map})
        cnt += 1
        header = header+subcatchment+"\n"

    # DD If the following commands give an error, then replace it with the proper method to cut pcraster maps without getting the error
    # In addition, there is no point in converting points to indices from 1 to 5 if they are later removed in inflow.py.
    # So instead, just clip the map with the original catchment numbers
    # pcrasterCommand(pcrcalc + " 'F1 = if(scalar(boolean(F0))>0,nominal(F0))'", {"F0": subcatchinlets_new_map,"F1": subcatchinlets_new2_map})
    # pcrasterCommand(resample + " F0 F1 --clone F2 " , {"F0": subcatchinlets_new2_map, "F1":subcatchinlets_new3_map, "F2":smallsubcatchmask_map})
    #print("(note that despite memory error, inflow_new3.map is being created, strange...)")
    # pcrasterCommand(pcrcalc + " 'F1 = if(F0>=0,F0)'", {"F0": subcatchinlets_map,"F1": subcatchinlets_new_map})
    pcrasterCommand(cfg.pcraster_cmd['resample'] + " --clone F2 F0 F1" , {"F0": subcatchinlets_map, "F1":subcatchinlets_cut_map, "F2":smallsubcatchmask_map})
    # map = pcraster.readmap(subcatchinlets_cut_map)
    # mapNpyInt = int(pcraster.pcr2numpy(map, -9999))
    # mapN = pcraster.numpy2pcr(pcraster.Nominal, map, -9999)

    if ("big_one" in globals()) or ("big_one" in locals()):

        big_one_lastrun.to_csv(inflow_tss_last_run,sep=' ',header=False)
        #simulated_streamflow_lastrun.to_csv(inflow_tss_last_run,sep=' ',header=False)
        f = open(inflow_tss_last_run,'r+')
        content = f.read()
        content = 'timeseries scalar\n'+str(cnt)+'\n'+'timestep\n'+header+content
        f.seek(0,0)
        f.write(content)
        f.close()
        
        big_one.to_csv(inflow_tss,sep=' ',header=False)
        f = open(inflow_tss,'r+')
        content = f.read()
        content = 'timeseries scalar\n'+str(cnt)+'\n'+'timestep\n'+header+content
        f.seek(0,0)
        f.write(content)
        f.close()
    else:
        sys.stdout.write("No upstream inflow needed\n")
    sys.stdout.write("\n")

    inflowflag = str(0)
    if os.path.isfile(inflow_tss):
        inflowflag = str(1)

    return inflowflag