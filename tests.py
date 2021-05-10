#!/bin/python
import sys
import os
import re
import subprocess
import unittest
import pytest
lisfloodRoot = "/perm/rd/nedd/EFAS/efasCalib/input/src/lisflood-develop/src/"
sys.path.insert(0, lisfloodRoot)
print(sys.path)
import CAL_7_PERFORM_CAL as calib
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15
import math
import pandas as pd
import time
import importlib

if ver.find('3.') > -1:
  parser = ConfigParser()  # python 3.8
else:
  parser = SafeConfigParser()  # python 2.7-15
parser.read(sys.argv[1])

SubCatchmentPath = parser.get('Path', 'SubCatchmentPath')

with open(sys.argv[2], "r") as catchmentFile:
  catchmentIndex = int(catchmentFile.readline().replace("\n", ""))


def roundn(x, n):
  numDigits = n - int(math.floor(math.log10(abs(x)))) - 1
  return round(x * 10 ** numDigits) / 10 ** numDigits


def floorn(x, n):
  numDigits = n - int(math.floor(math.log10(abs(x)))) - 1
  return math.floor(x * 10 ** numDigits) / 10 ** numDigits


def ceiln(x, n):
  numDigits = n - int(math.floor(math.log10(abs(x)))) - 1
  return math.ceil(x * 10 ** numDigits) / 10 ** numDigits


def runCmd(cmd):
  res = subprocess.run(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if res.returncode == 0:
    out = res.stdout[0:-1]
  else:
    Exception(res.stderr)
    out = res.stderr
  return (res.returncode, out)

# For checking CAL_7_PERFORM_CAL.py and cal_single_objfun.py functioning bypassing uncertainty with deap
def test_calib_launcher():
  # assert tail -1 front_history.csv | cut -d "," -f6
  with open(os.path.join(SubCatchmentPath, str(catchmentIndex), 'front_history.csv')) as f:
    # KGE = float(subprocess.check_output("tail -1 " + f.name + " | cut -d ',' -f6", stderr=subprocess.STDOUT, shell=True)[0:-1])
    # DD: Better alternative with error handling
    ret, KGE = runCmd("tail -1 " + f.name + " | cut -d ',' -f6")
    KGE = float(KGE)
  try:
    assertion = abs(KGE - 0.9999) < 1e-5 and KGE > 0.99
    assert assertion, "KGE = " + str(KGE) + "; abs(KGE - 0.9999) = " + str(abs(KGE-0.9999)) + " should be < 1e-5"
    retcode = 0
  except AssertionError as e:
    print("ERROR in test_calib_launcher(): " + str(e))
    retcode = 1
  return retcode, KGE, abs(KGE - 0.9999)

n = 4
for x in [
  1.216846,
  5868.51986,
  13548643.15,
  2,
  1.9999999,
  3.21354131,
  0.00021658468,
  0.065498432,
  0.16843584185e12,
]:
  print(x, floorn(x,n), ceiln(x,n))
print("done")

def saveOutput(data):
  destCatchFolder = os.path.join(SubCatchmentPath, str(catchmentIndex))
  destFolder = os.path.join(destCatchFolder, 'convExperiments', "{}_{}_{}_{}_{}_{}_{}_{}".format(*data))
  ret, res = runCmd("mkdir -p {}".format(destFolder))
  ret, res = runCmd("mv {}/settings*.xml {}/".format(destCatchFolder, destFolder))
  ret, res = runCmd("mv {}/out {}/".format(destCatchFolder, destFolder))
  ret, res = runCmd("mv {}/*.csv {}/".format(destCatchFolder, destFolder))
  ret, res = runCmd("mkdir -p {}/out".format(destCatchFolder))
  ret, res = runCmd("cp {}/convergenceTester.csv {}/.".format(destFolder, destCatchFolder))


# fast test
def test_calib_aggregation():
  # assert tail -1 front_history.csv | cut -d "," -f6
  with open(os.path.join(SubCatchmentPath, str(catchmentIndex), 'front_history.csv')) as f:
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

# do the calibration for different configurations
print("numDigits, round(numLambda / 2), numLambda, numPop\n--------------------------------------------------")

ct = 0
experiments = pd.DataFrame(columns=['NumDigits', 'mu', 'lambda', 'pop', 'KGE', 'KGEDiff', 'runtime', 'retCode'])
for numDigits in range(1,7,1):
  # inplacements(os.path.join(os.path.dirname(sys.argv[1]), 'cal_single_objfun.py'), [("numDigits", numDigits)])
  # importlib.reload(calib)
  for numLambda in range(2,9,1):
  # for numLambda in [4]:
    # only even # of children
    if numLambda % 2 == 0:
      for numPop in range(2,17,1):
      # for numPop in [4]:
        # only keep # of children when they are double or single times the starting population size
        # This is critical for the functioning of the crossover and blending generational mutations
        if numLambda == numPop / 2 or numLambda == numPop:
          # for numMonths in range(2,6*12,1):
          inplacements(sys.argv[1],
            [
              ("numCPUs", 2),
              ("minGen", 1),
              ("maxGen", 1),
              ("mu", max(round(numLambda / 2), 2)),
              ("lambda_", numLambda),
              ("pop", numPop),
              ("numDigitsTests", numDigits)
            ]
          )
          # CM proposed to turn off reservoirs so as to have 3 variable params while we can keep the others constant
          # DD testing if simple prescription cannot be done
          # inplacements(sys.argv[1],
          #   [
          #     ('<setoption choice="1" name="simulateLakes"/>', '<setoption choice="0" name="simulateLakes"/>'),
          #     ('<setoption choice="1" name="simulateReservoirs"/>', '<setoption choice="0" name="simulateReservoirs"/>'),
          #   ], precise=True
          # )
          print(numDigits, max(round(numLambda / 2), 2),numLambda,numPop)
          importlib.reload(calib)
          print("checkpoint")
          bTime = time.time()
          # Run a first time to generate the synthetic time series we use as obs
          if not os.path.isfile(os.path.join(SubCatchmentPath, str(catchmentIndex), 'convergenceTester.csv')) or os.path.getsize(os.path.join(SubCatchmentPath, str(catchmentIndex), 'convergenceTester.csv')) == 0:
            try:
              calib.main(sys.argv[1:])
            except SystemExit:
              pass
          # Run a second time to run the actual calib
          if not os.path.isfile(os.path.join(SubCatchmentPath, str(catchmentIndex), 'pareto_front.csv')) or os.path.getsize(os.path.join(SubCatchmentPath, str(catchmentIndex), 'pareto_front.csv')) == 0:
            calib.main(sys.argv[1:])
          # # Run a 3rd time to generate the simulated best streamflow
          # calib.main(sys.argv[1:])
          runtime = time.time() - bTime
          retCode, KGE, KGEDiff = test_calib_launcher()
          experiments.loc[ct] = [numDigits, round(numLambda / 2), numLambda, numPop, KGE, KGEDiff, runtime, retCode]
          saveOutput([numDigits, max(round(numLambda / 2), 2), numLambda, numPop, KGE, KGEDiff, runtime, retCode])
          ct += 1

print(ct)
experiments.to_csv(os.path.join(SubCatchmentPath, str(catchmentIndex), 'convergenceExperiments.csv'))