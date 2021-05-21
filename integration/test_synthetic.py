#!/bin/python
import sys
import os
import re
import subprocess
import pandas
from datetime import datetime, timedelta

from liscal import hydro_model, calibration, templates, config, subcatchment, objective, utils


# For checking CAL_7_PERFORM_CAL.py and cal_single_objfun.py functioning bypassing uncertainty with deap
def test_calib_launcher(cfg, catchment_index, target1, target2, tol):
  # assert tail -1 front_history.csv | cut -d "," -f6
    with open(os.path.join(cfg.subcatchment_path, str(catchment_index), 'front_history.csv')) as f:
        # KGE = float(subprocess.check_output("tail -1 " + f.name + " | cut -d ',' -f6", stderr=subprocess.STDOUT, shell=True)[0:-1])
        # DD: Better alternative with error handling
        ret, KGE = utils.run_cmd("tail -1 " + f.name + " | cut -d ',' -f6")
        KGE = float(KGE)
        assertion = abs(KGE - target1) < tol and KGE > target2
    assert assertion, "Target not reached! abs({} - {}) = {} is > {} or {} < {}".format(KGE, target1, abs(KGE-target1), tol, KGE, target2)


def saveOutput(cfg, catchment_index, data):
    subcatch_dir = os.path.join(cfg.subcatchment_path, str(catchment_index))
    exp_dir = os.path.join(subcatch_dir, 'convExperiments', "{}_{}_{}_{}_{}_{}_{}_{}".format(*data))
    ret, res = utils.run_cmd("mkdir -p {}".format(exp_dir))
    ret, res = utils.run_cmd("mv {}/settings*.xml {}/".format(subcatch_dir, exp_dir))
    ret, res = utils.run_cmd("mv {}/out {}/".format(subcatch_dir, exp_dir))
    ret, res = utils.run_cmd("mv {}/*.csv {}/".format(subcatch_dir, exp_dir))
    ret, res = utils.run_cmd("cp {}/convergenceTester.csv {}/.".format(exp_dir, subcatch_dir))


def deleteOutput(cfg, catchment_index):
    subcatch_dir = os.path.join(cfg.subcatchment_path, str(catchment_index))
    ret, res = utils.run_cmd("rm -f {}/settings*.xml".format(subcatch_dir))
    ret, res = utils.run_cmd("rm -rf {}/out".format(subcatch_dir))
    ret, res = utils.run_cmd("find {} -name '*.csv' -not -name 'convergenceTester.csv' -delete".format(subcatch_dir))
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


class ObjectiveDischargeTest(objective.ObjectiveDischarge):

    def __init__(self, cfg, subcatch, tol):

        self.tol = tol

        super().__init__(cfg, subcatch)

    def read_observed_streamflow(self):
        cfg = self.cfg
        obsid = self.subcatch.obsid

        # Load observed streamflow # DD Much faster IO with npy despite being more complicated (<1s vs 22s)
        streamflow_data = pandas.read_csv(cfg.subcatchment_path + "/" + str(obsid) + "/convergenceTester.csv", sep=",", index_col=0, header=None)
        # streamflow_data.index = pandas.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
        #streamflow_data.index = pandas.date_range(start=ForcingStart, end=ForcingEnd, periods=len(streamflow_data))
        streamflow_data.index = pandas.date_range(start=streamflow_data.index[0], end=streamflow_data.index[-1], periods=len(streamflow_data))
        observed_streamflow = streamflow_data[cfg.forcing_start:cfg.forcing_end]
        return observed_streamflow

    def get_parameters(self, Individual):
        param_ranges = self.cfg.param_ranges
        parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
          ref = 0.5 * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])
          parameters[ii] = ref * (1+self.tol)
        return parameters


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
    ret, res = utils.run_cmd("mkdir -p {}/out".format(os.path.join(subcatch.path)))

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

    lock_mgr = calibration.LockManager()

    tol = 1e-4

    objective = ObjectiveDischargeTest(cfg, subcatch, tol)

    model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, objective)
    model.init_run() # load static maps into memory

    calib_deap = calibration.CalibrationDeap(cfg, model.run)
    calib_deap.run(subcatch.path, lock_mgr)

    test_calib_launcher(cfg, obsid, target1=0.9999, target2=0.99, tol=tol)

    deleteOutput(cfg, obsid)
