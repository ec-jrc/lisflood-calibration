#!/bin/python
import sys
import os
from os import path
import re
import argparse
import pandas
import numpy as np
from datetime import datetime, timedelta

from liscal import hydro_model, calibration, templates, config, subcatchment, objective, utils

ROOT_DIR = path.join(path.dirname(path.realpath(__file__)), '..')
TEST_DIR = path.join(ROOT_DIR, 'tests')
DATA_DIR = path.join(TEST_DIR, 'data')
OUT_DIR = path.join(TEST_DIR, 'outputs')


def test_calib_launcher(cfg, catchment_index, target1, target2, tol):
  # assert tail -1 front_history.csv | cut -d "," -f6
    with open(os.path.join(cfg.subcatchment_path, str(catchment_index), 'front_history.csv')) as f:
        # KGE = float(subprocess.check_output("tail -1 " + f.name + " | cut -d ',' -f6", stderr=subprocess.STDOUT, shell=True)[0:-1])
        # DD: Better alternative with error handling
        ret, KGE = utils.run_cmd("tail -1 " + f.name + " | cut -d ',' -f6")
        KGE = float(KGE)
        assertion = abs(KGE - target1) < tol and KGE > target2
    assert assertion, "Target not reached! abs({} - {}) = {} is > {} or {} < {}".format(KGE, target1, abs(KGE-target1), tol, KGE, target2)


def deleteOutput(subcatch_dir):
    ret, res = utils.run_cmd("rm -f {}/settings*.xml".format(subcatch_dir))
    ret, res = utils.run_cmd("rm -rf {}/out".format(subcatch_dir))
    ret, res = utils.run_cmd("rm -rf {}/*.csv".format(subcatch_dir))


class ObjectiveDischargeTest(objective.ObjectiveDischarge):

    def __init__(self, cfg, subcatch):

        super().__init__(cfg, subcatch)

    def read_observed_streamflow(self):
        cfg = self.cfg
        obsid = self.subcatch.obsid

        # Load observed streamflow # DD Much faster IO with npy despite being more complicated (<1s vs 22s)
        streamflow_data = pandas.read_csv(os.path.join(subcatch.path, 'out', 'convergenceTester.csv'), sep=",", index_col=0, header=None)
        # streamflow_data.index = pandas.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
        #streamflow_data.index = pandas.date_range(start=ForcingStart, end=ForcingEnd, periods=len(streamflow_data))
        streamflow_data.index = pandas.date_range(start=streamflow_data.index[0], end=streamflow_data.index[-1], periods=len(streamflow_data))
        observed_streamflow = streamflow_data[cfg.forcing_start:cfg.forcing_end]
        return observed_streamflow


class ObjectiveDischargeTestFast(ObjectiveDischargeTest):

    def __init__(self, cfg, subcatch, param_target, tol):

        self.tol = tol
        self.param_target = param_target

        super().__init__(cfg, subcatch)

    def get_parameters(self, Individual):
        param_ranges = self.cfg.param_ranges
        parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
            ref = self.param_target[ii] * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])
            parameters[ii] = ref * (1+self.tol)
        return parameters


class SlowDEAPParameters():
    min_gen = 6
    max_gen = 16
    pop = 72
    mu = 18
    lambda_ = 36
    cxpb = 0.6
    mutpb = 0.4


class FastDEAPParameters():
    min_gen = 1
    max_gen = 1
    pop = 2
    mu = 2
    lambda_ = 2
    cxpb = 0.6
    mutpb = 0.4


class ConfigTest():

    def __init__(self, data_dir, slow):

        # paths
        self.path_result = path.join(data_dir, 'result')
        self.subcatchment_path = path.join(data_dir, 'catchments')

        pcraster_path = ''
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname

        # deap
        if slow:
            self.deap_param = SlowDEAPParameters()
        else:
            self.deap_param = FastDEAPParameters()

        # Load param ranges file
        param_ranges_file = path.join(DATA_DIR, 'ParamRanges_LISFLOOD.csv')
        self.param_ranges = pandas.read_csv(param_ranges_file, sep=",", index_col=0)

        # template
        self.lisflood_template = path.join(ROOT_DIR, 'templates','settings_LF_CUT.xml')

        # Debug/test parameters
        self.fast_debug = False

        self.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")  # Start of forcing
        self.WarmupDays = 30
        self.calibration_freq = '6-hourly'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('obsid', help='Station obsid')
    parser.add_argument('data_dir', help='Static data folder')
    parser.add_argument('--slow', action='store_true', help='run slow test')
    parser.add_argument('--ncpus', default=1)
    args = parser.parse_args()

    print('  - obsid: {}'.format(args.obsid))
    print('  - ncpus: {}'.format(args.ncpus))
    print('  - data_dir: {}'.format(args.data_dir))
    cfg = ConfigTest(args.data_dir, args.slow)
    obsid = int(args.obsid)
    ncpus = int(args.ncpus)

    print(">> Reading Qmeta2.csv file...")
    stations_file = path.join(ROOT_DIR, 'integration', 'stations.csv')
    stations = pandas.read_csv(stations_file, sep=",", index_col=0)
    try:
        station_data = stations.loc[obsid]
    except KeyError as e:
        print(stations)
        raise Exception('Station {} not found in stations file'.format(obsid))

    # hack shorter period
    n_days_test = 365
    station_data.loc['Cal_Start'] = (cfg.forcing_end - timedelta(days=n_days_test)).strftime('%d/%m/%Y %H:%M')
    station_data.loc['Cal_End'] = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data)
    if os.path.exists(os.path.join(subcatch.path, "pareto_front.csv")):
        deleteOutput(subcatch.path)

    # create object to create lisflood settings file
    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

    # create lock manager, which will handle process mapping and locks
    lock_mgr = calibration.LockManager(ncpus)

    # first generate synthetic observations
    outfile = os.path.join(subcatch.path, 'out', 'convergenceTester.csv')
    param_target = 0.5*np.ones(len(cfg.param_ranges))
    hydro_model.generate_benchmark(cfg, subcatch, lis_template, param_target, outfile)

    # create objective (slow or fast) and hydro model
    if args.slow:
        tol = 1e-2
        obj = ObjectiveDischargeTest(cfg, subcatch)
    else:
        tol = 1e-4
        obj = ObjectiveDischargeTestFast(cfg, subcatch, param_target, tol)
    model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

    # load forcings and input maps in cache
    model.init_run()

    # create calib object and run
    calib_deap = calibration.CalibrationDeap(cfg, model.run)
    calib_deap.run(subcatch.path, lock_mgr)

    # process calibration results
    obj.process_results()

    # check KGE
    test_calib_launcher(cfg, obsid, target1=1., target2=0.99, tol=2*tol)

    # run long term run on full forcing dates
    if args.slow:
        hydro_model.generate_outlet_streamflow(cfg, subcatch, lis_template)

    # deleteOutput(subcatch.path)
