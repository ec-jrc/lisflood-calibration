#!/bin/python3
import os
import pandas
import argparse
import traceback
import numpy as np
from datetime import datetime, timedelta

# lisflood
import lisf1
from lisflood.global_modules.decorators import Cache

from liscal import hydro_model, templates, config, subcatchment, calibration, objective


class ScalingModel():
    def __init__(self, cfg, subcatch, lis_template, lock_mgr, objective):

        self.cfg = cfg
        self.subcatch = subcatch

        self.lis_template = lis_template

        self.lock_mgr = lock_mgr

        self.objective = objective

        self.start = cfg.forcing_start.strftime('%d/%m/%Y %H:%M')
        self.end = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

    def init_run(self):

        # dummy Individual, doesn't matter here
        param_ranges = self.cfg.param_ranges
        Individual = 0.5*np.ones(len(param_ranges))

        cfg = self.cfg

        gen = self.lock_mgr.get_gen()
        run_id = str(gen)

        parameters = self.objective.get_parameters(Individual)

        prerun_file, run_file = self.lis_template.write_template(run_id, self.start, self.end, self.start, self.end, cfg.param_ranges, parameters)

        # -i option to exit after initialisation, we just load the inputs map in memory
        try:
            lisf1.main(prerun_file, '-i')
        except:
            traceback.print_exc()
            raise Exception("Lisflood failed!")

        # store lisflood cache size to make sure we don't load anything else after that
        self.lisflood_cache_size = Cache.size()


    def run(self, Individual):

        cfg = self.cfg

        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.increment_run()
        print('Generation {}, run {}'.format(gen, run))

        run_id = '{}_{}'.format(gen, run)

        parameters = self.objective.get_parameters(Individual)

        prerun_file, run_file = self.lis_template.write_template(run_id, self.start, self.end, self.start, self.end, cfg.param_ranges, parameters)

        try:
            lisf1.main(prerun_file, '-v')
            lisf1.main(run_file, '-v')
        except:
            traceback.print_exc()
            raise Exception("Lisflood failed!")

        # check lisflood cache size to make sure we don't load the same map multiple times
        cache_size = Cache.size()
        assert cache_size == self.lisflood_cache_size


def scaling_subcatchment(cfg, obsid, subcatch, n_runs):

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
    
    lock_mgr = calibration.LockManager(cfg.num_cpus)

    lock_mgr.set_gen(cfg.num_cpus)

    obj = objective.ObjectiveKGE(cfg, subcatch, read_observations=False)

    model = ScalingModel(cfg, subcatch, lis_template, lock_mgr, obj)

    # load forcings and input maps in cache
    # required in front of processing pool
    # otherwise each child will reload the maps
    model.init_run()

    dummy_params = [0.5*np.ones(len(cfg.param_ranges)) for i in range(n_runs)]

    scaling_map, pool = lock_mgr.create_mapping()

    mapped = list(scaling_map(model.run, dummy_params))

    if pool:
        pool.close()


class ConfigScaling(config.Config):

    def __init__(self, settings_file, num_cpus):
        super().__init__(settings_file)

        self.num_cpus = int(num_cpus)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # Date parameters
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.timestep = int(self.parser.get('Main', 'timestep'))  # in minutes
        self.prerun_timestep = 360  # in minutes
        if self.prerun_timestep != 360 and self.prerun_timestep != 1440:
            raise Exception('Pre-run timestep {} not supported'.format(self.prerun_timestep))
        
        # Load param ranges file
        self.param_ranges = pandas.read_csv(self.parser.get('Path','param_ranges'), sep=",", index_col=0)

        # template
        self.lisflood_template = self.parser.get('Templates','LISFLOODSettings')
        
        # pcraster commands
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings', help='Settings file')
    parser.add_argument('stations_data', help='Stations metadata CSV file')
    parser.add_argument('obsid', help='Station obsid')
    parser.add_argument('n_cpus', help='Station obsid')
    parser.add_argument('n_runs', help='Station obsid')
    args = parser.parse_args()

    print('  - obsid: {}'.format(args.obsid))
    print('  - settings file: {}'.format(args.settings))
    obsid = int(args.obsid)
    cfg = ConfigScaling(args.settings, args.n_cpus)

    print(">> Reading stations.csv file...")
    stations = pandas.read_csv(args.stations_data, sep=",", index_col=0)
    try:
        station_data = stations.loc[obsid]
    except KeyError as e:
        print(stations)
        raise Exception('Station {} not found in stations file'.format(obsid))

    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data, create_links=False)

    scaling_subcatchment(cfg, obsid, subcatch, int(args.n_runs))