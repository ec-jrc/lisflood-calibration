#!/bin/python
import sys
import os
import re
import subprocess
import pandas
from datetime import datetime, timedelta

from liscal import hydro_model, calibration, templates, config,  subcatchment


def runCmd(cmd):
    res = subprocess.run(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        out = res.stdout[0:-1]
    else:
        Exception(res.stderr)
        out = res.stderr
    return (res.returncode, out)


# For checking CAL_7_PERFORM_CAL.py and cal_single_objfun.py functioning bypassing uncertainty with deap
def test_calib_launcher(cfg, catchment_index, target1, target2, tol):
  # assert tail -1 front_history.csv | cut -d "," -f6
    with open(os.path.join(cfg.subcatchment_path, str(catchment_index), 'front_history.csv')) as f:
        # KGE = float(subprocess.check_output("tail -1 " + f.name + " | cut -d ',' -f6", stderr=subprocess.STDOUT, shell=True)[0:-1])
        # DD: Better alternative with error handling
        ret, KGE = runCmd("tail -1 " + f.name + " | cut -d ',' -f6")
        KGE = float(KGE)
        assertion = abs(KGE - target1) < tol and KGE > target2
    assert assertion, "Target not reached! abs({} - {}) = {} is > {} or {} < {}".format(KGE, target1, abs(KGE-target1), tol, KGE, target2)


def saveOutput(cfg, catchment_index, data):
    subcatch_dir = os.path.join(cfg.subcatchment_path, str(catchment_index))
    exp_dir = os.path.join(subcatch_dir, 'convExperiments', "{}_{}_{}_{}_{}_{}_{}_{}".format(*data))
    ret, res = runCmd("mkdir -p {}".format(exp_dir))
    ret, res = runCmd("mv {}/settings*.xml {}/".format(subcatch_dir, exp_dir))
    ret, res = runCmd("mv {}/out {}/".format(subcatch_dir, exp_dir))
    ret, res = runCmd("mv {}/*.csv {}/".format(subcatch_dir, exp_dir))
    ret, res = runCmd("cp {}/convergenceTester.csv {}/.".format(exp_dir, subcatch_dir))


def deleteOutput(cfg, catchment_index):
    subcatch_dir = os.path.join(cfg.subcatchment_path, str(catchment_index))
    ret, res = runCmd("rm -f {}/settings*.xml".format(subcatch_dir))
    ret, res = runCmd("rm -rf {}/out".format(subcatch_dir))
    ret, res = runCmd("find {} -name '*.csv' -not -name 'convergenceTester.csv' -delete".format(subcatch_dir))
    # ret, res = runCmd("for f in $(find {} -name '*.csv -a -not -wholename 'convergenceTester.csv'); do rm -rf $f; done".format(destCatchFolder))


# fast test
def test_calib_aggregation(cfg, catchment_index):
    # assert tail -1 front_history.csv | cut -d "," -f6
    with open(os.path.join(cfg.subcatchment_path, str(catchment_index), 'front_history.csv')) as f:
        subprocess.run(["ls", "foo bar"], check=True)
        line = subprocess.check_output(["tail -1 ", f.name, " | cut -d ',' -f6"])  # ['tail', '-1', filename])
        print(line)
        # assert qsim and qobs tseries are matching with benchmark (RMS or SumError, SquaredSum) < threshold


# inplace string replacements
def inplacements(inFile, strings, precise=False):
    # read original
    file = open(inFile, 'r')
    settings = file.read()
    file.close()
    # do each replacement
    for s in strings:
        if precise:
            settings = re.sub("{}".format(s[0]), "{}".format(s[0], s[1]), settings)
        else:
            settings = re.sub("{} = .*".format(s[0]), "{} = {}".format(s[0], s[1]), settings)
    # Overwrite with buffer
    file = open(inFile, 'w')
    file.write(settings)
    file.close()

if __name__ == '__main__':

    cfg = config.Config(sys.argv[1])

    with open(sys.argv[2], "r") as catchmentFile:
      obsid = int(catchmentFile.readline().replace("\n", ""))

    print(">> Reading Qmeta2.csv file...")
    stations = pandas.read_csv(os.path.join(cfg.path_result,"Qmeta2.csv"), sep=",", index_col=0)

    try:
        station_data = stations.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    # hack shorter period
    station_data['Cal_Start'] = (cfg.forcing_end - timedelta(days=335+cfg.WarmupDays)).strftime('%Y-%m-%d %H:%M')
    station_data['Cal_End'] = cfg.forcing_end.strftime('%Y-%m-%d %H:%M')

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data)
    if os.path.exists(os.path.join(subcatch.path, "pareto_front.csv")):
        deleteOutput(cfg, obsid)
    ret, res = runCmd("mkdir -p {}/out".format(os.path.join(subcatch.path)))

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

    lock_mgr = calibration.LockManager()

    tol = 1e-4

    objective = hydro_model.ObjectiveDischargeTest(cfg, subcatch, tol)

    model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, objective)

    # Performing calibration with external call, to avoid multiprocessing problems
    calibration.run_calibration(cfg, subcatch, model, lock_mgr)

    test_calib_launcher(cfg, obsid, target1=0.9999, target2=0.99, tol=tol)

    deleteOutput(cfg, obsid)
