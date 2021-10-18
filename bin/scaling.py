#!/bin/python3
from datetime import datetime
import os
import time
import pandas
import argparse
import traceback
import numpy as np
from datetime import datetime, timedelta

# lisflood
import lisf1
from lisflood import cache

from liscal import hydro_model, templates, config, subcatchment, calibration, objective, schedulers


class ScalingModel():
    def __init__(self, cfg, subcatch, lis_template, objective):

        self.cfg = cfg
        self.subcatch = subcatch

        self.lis_template = lis_template

        self.objective = objective

        spinup = int(float(subcatch.data['Spinup_days']))
        self.obs_start = datetime.strptime(subcatch.data['Split_date'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
        self.obs_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
        self.cal_start = (datetime.strptime(self.obs_start,"%d/%m/%Y %H:%M") - timedelta(days=spinup)).strftime('%d/%m/%Y %H:%M')
        self.cal_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

    def init_run(self, run_id):

        # dummy Individual, doesn't matter here
        param_ranges = self.cfg.param_ranges
        Individual = 0.5*np.ones(len(param_ranges))

        cfg = self.cfg

        parameters = self.objective.get_parameters(Individual)

        self.lis_template.write_template(run_id, self.cal_start, self.cal_end, cfg.param_ranges, parameters)

        prerun_file = self.lis_template.settings_path('-PreRun', run_id)

        # -i option to exit after initialisation, we just load the inputs map in memory
        try:
            lisf1.main(prerun_file, '-i')
        except:
            traceback.print_exc()
            raise Exception("Lisflood failed!")

        # store lisflood cache size to make sure we don't load anything else after that
        self.lisflood_cache_size = cache.cache_size()


    def run(self, individual):

        cfg = self.cfg

        gen = individual['gen']
        run = individual['run']
        size = individual['size']
        print('Generation {}, run {}'.format(gen, run))

        run_id = '{}_{}_{}'.format(size, gen, run)

        parameters = self.objective.get_parameters(individual['value'])

        self.lis_template.write_template(run_id, self.cal_start, self.cal_end, cfg.param_ranges, parameters)

        prerun_file = self.lis_template.settings_path('-PreRun', run_id)
        run_file = self.lis_template.settings_path('-Run', run_id)

        try:
            lisf1.main(prerun_file, '-v')
            lisf1.main(run_file, '-v')
        except:
            traceback.print_exc()
            raise Exception("Lisflood failed!")

        # check lisflood cache size to make sure we don't load the same map multiple times
        cache_size = cache.cache_size()
        assert cache_size == self.lisflood_cache_size

        return 1


def scaling_subcatchment(cfg, obsid, subcatch, scheduler, n_threads, n_runs):
    
    t0 = time.time()

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch, n_threads)
    
    obj = objective.ObjectiveKGE(cfg, subcatch, read_observations=False)

    model = ScalingModel(cfg, subcatch, lis_template, obj)

    # load forcings and input maps in cache
    # required in front of processing pool
    # otherwise each child will reload the maps
    scheduler.sequence(model.init_run, (str(scheduler.rank)))

    if scheduler.root():
        t1 = time.time()
        print('\nInit forcings done in {}\n'.format(t1-t0)) 
        f = open("timings.txt", "a")
        f.write('2 {}\n'.format(t1-t0))
        f.close()

    scaling_map = scheduler.create_mapping()

    for gen in range(2):

        individuals = []
        for i in range(n_runs):
            ind = {}
            ind['gen'] = gen
            ind['run'] = i
            ind['size'] = scheduler.size
            ind['value'] = 0.5*np.ones(len(cfg.param_ranges))
            individuals.append(ind)
        individuals = scheduler.chunk(individuals)
        
        mapped = scaling_map(model.run, individuals)
        
        if scheduler.root():
            print(mapped)
            tx = time.time()
        mapped = scheduler.gather(mapped)
        if scheduler.root():
            print('Gather done in {}'.format(time.time()-tx)) 
            print(mapped)

    if scheduler.root():
        t2 = time.time()
        print('\nCalib done in {}\n'.format(t2-t1)) 
        f = open("timings.txt", "a")
        f.write('3 {}\n'.format(t2-t1))
        f.close()

    scheduler.close()


class ConfigScaling(config.Config):

    def __init__(self, settings_file, num_cpus):
        super().__init__(settings_file)

        self.num_cpus = int(num_cpus)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # Date parameters
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.calibration_freq = self.parser.get('Main', 'calibration_freq')
        
        # Load param ranges file
        self.param_ranges = pandas.read_csv(self.parser.get('Path','param_ranges'), sep=",", index_col=0)

        # template
        self.lisflood_template = self.parser.get('Templates','LISFLOODSettings')
        
        # pcraster commands
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('settings', help='Settings file')
    parser.add_argument('stations_data', help='Stations metadata CSV file')
    parser.add_argument('obsid', help='Station obsid')
    parser.add_argument('n_cpus', help='Number of processes')
    parser.add_argument('n_threads', help='Number of threads')
    parser.add_argument('n_runs', help='Number of lisflood instances to run')
    args = parser.parse_args()

    scheduler = schedulers.get_scheduler('MPI', int(args.n_cpus))

    obsid = int(args.obsid)
    cfg = ConfigScaling(args.settings, args.n_cpus)

    if scheduler.root():
        f = open("timings.txt", "w")
        f.write('#step time\n')
        f.close()
        print('  - obsid: {}'.format(args.obsid))
        print('  - settings file: {}'.format(args.settings))
        cfg.info()

    subcatch = scheduler.sequence(subcatchment.SubCatchment, cfg, obsid, None, True, False)

    if scheduler.root():
        print(subcatch.info())
        t1 = time.time()
        print('\nConfig and subcatchment time: {}\n'.format(t1-t0))
        f = open("timings.txt", "a")
        f.write('1 {}\n'.format(t1-t0))
        f.close()

    scaling_subcatchment(cfg, obsid, subcatch, scheduler, args.n_threads, int(args.n_runs))

    if scheduler.root():
        tf = time.time()
        print('\nTotal time: {}\n'.format(tf-t0))
        f = open("timings.txt", "a")
        f.write('0 {}\n'.format(tf-t0))
        f.close()