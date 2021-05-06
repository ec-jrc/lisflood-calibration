import sys
import os



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
