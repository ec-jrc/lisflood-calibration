#!/bin/python
import sys
import os
import re
import subprocess

sys.path.insert(0, '/home/ma/macw/git/lisflood-calibration')

import CAL_7_PERFORM_CAL as calib


def roundn(x, n, num_digits=4):
  num_digits = n - int(math.floor(math.log10(abs(x)))) - 1
  return round(x * 10 ** num_digits) / 10 ** num_digits


def floorn(x, n, num_digits=4):
  num_digits = n - int(math.floor(math.log10(abs(x)))) - 1
  return math.floor(x * 10 ** num_digits) / 10 ** num_digits


def ceiln(x, n, num_digits=4):
  num_digits = n - int(math.floor(math.log10(abs(x)))) - 1
  return math.ceil(x * 10 ** num_digits) / 10 ** num_digits


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
  catchment_index = int(catchmentFile.readline().replace("\n", ""))

ret, res = runCmd("mkdir -p {}/out".format(os.path.join(cfg.subcatchment_path, str(catchment_index))))

# Run a second time to run the actual calib
if not os.path.isfile(os.path.join(cfg.subcatchment_path, str(catchment_index), 'pareto_front.csv')) or os.path.getsize(os.path.join(cfg.subcatchment_path, str(catchment_index), 'pareto_front.csv')) == 0:
  calib.calibrate_subcatchment(cfg, catchment_index)

test_calib_launcher(cfg, catchment_index, target1=0.9999, target2=0.99, tol=1e-4)

deleteOutput(cfg, catchment_index)
