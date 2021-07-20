#!/bin/python3
import sys
import os
from os import path
import re
import argparse
import pandas
import numpy as np
from datetime import datetime, timedelta

from liscal import hydro_model, calibration, templates, config, subcatchment, objective, utils


def deleteOutput(subcatch_dir):
    ret, res = utils.run_cmd("rm -f {}/settings*.xml".format(subcatch_dir))
    # ret, res = utils.run_cmd("rm -rf {}/out".format(subcatch_dir))
    ret, res = utils.run_cmd("rm -rf {}/*.csv".format(subcatch_dir))


class ObjectiveKGETest(objective.ObjectiveKGE):

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('obsid', help='Station obsid')
    parser.add_argument('settings', help='Settings file')
    parser.add_argument('tol', help='KGE history file')
    args = parser.parse_args()

    print('  - obsid: {}'.format(args.obsid))
    print('  - settings file: {}'.format(args.settings))
    obsid = int(args.obsid)
    cfg = config.ConfigCalibration(args.settings)

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid)
    if os.path.exists(os.path.join(subcatch.path, "pareto_front.csv")):
        deleteOutput(subcatch.path)
        os.makedirs(subcatch.path_out, exist_ok=True)

    # create object to create lisflood settings file
    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

    # create lock manager, which will handle process mapping and locks
    lock_mgr = calibration.LockManager(cfg.deap_param.num_cpus)

    # create objective and hydro model
    tol = float(args.tol)
    param_target = 0.5*np.ones(len(cfg.param_ranges))
    obj = ObjectiveKGETest(cfg, subcatch, param_target, tol)
    model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

    # load forcings and input maps in cache
    model.init_run()

    # create calib object and run
    calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights)
    calib_deap.run(subcatch.path, lock_mgr)

    # process calibration results
    obj.process_results()
