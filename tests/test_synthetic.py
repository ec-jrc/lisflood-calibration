#!/bin/python
import sys
import os
import re
import subprocess
import pandas

sys.path.insert(0, '/home/ma/macw/git/lisflood-calibration')

import CAL_7_PERFORM_CAL as calib
import cal_single_objfun


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

# slow test running 96h
def test_fullCalib():
  print()
  # assert KGE is above 0.99

    # os.path.join(os.path.dirname(sys.argv[0]), 'cal_single_objfun.py')
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


cfg = calib.Config(sys.argv[1])

with open(sys.argv[2], "r") as catchmentFile:
  obsid = int(catchmentFile.readline().replace("\n", ""))

ret, res = runCmd("mkdir -p {}/out".format(os.path.join(cfg.subcatchment_path, str(obsid))))

print(">> Reading Qmeta2.csv file...")
stations = pandas.read_csv(os.path.join(cfg.path_result,"Qmeta2.csv"), sep=",", index_col=0)

try:
    station_data = stations.loc[obsid]
except KeyError as e:
    raise Exception('Station {} not found in stations file'.format(obsid))

print("=================== "+str(obsid)+" ====================")
path_subcatch = os.path.join(cfg.subcatchment_path, str(obsid))
if os.path.exists(os.path.join(path_subcatch, "streamflow_simulated_best.csv")):
    deleteOutput(cfg, obsid)
print(">> Starting calibration of catchment "+str(obsid))

gaugeloc = calib.create_gauge_loc(cfg, path_subcatch)

inflowflag = calib.prepare_inflows(cfg, path_subcatch, obsid)

lis_template = calib.LisfloodSettingsTemplate(cfg, path_subcatch, obsid, gaugeloc, inflowflag)

lock_mgr = cal_single_objfun.LockManager()

tol = 1e-4

model = cal_single_objfun.HydrologicalModelTest(cfg, obsid, path_subcatch, station_data, lis_template, lock_mgr, tol=tol)

# Performing calibration with external call, to avoid multiprocessing problems
if os.path.exists(os.path.join(path_subcatch, "pareto_front.csv"))==False:
    cal_single_objfun.run_calibration(cfg, obsid, path_subcatch, station_data, model, lock_mgr)

test_calib_launcher(cfg, obsid, target1=0.9999, target2=0.99, tol=tol)

# deleteOutput(cfg, obsid)
