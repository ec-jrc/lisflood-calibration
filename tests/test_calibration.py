from os import path
import pandas
import numpy as np
import random
from liscal import calibration


ROOT_DIR = path.join(path.dirname(path.realpath(__file__)), '..')
TEST_DIR = path.join(ROOT_DIR, 'tests')
DATA_DIR = path.join(TEST_DIR, 'data')
OUT_DIR = path.join(TEST_DIR, 'outputs')


class DummyDEAPParameters():

    def __init__(self):
        self.num_cpus = 1
        self.min_gen = 6
        self.max_gen = 32
        self.pop = 72
        self.mu = 18
        self.lambda_ = 36

        self.cxpb = 0.6
        self.mutpb = 0.4


class ModelSingleObj():

    def __init__(self, dummy_cfg, lock_mgr):

        self.observations = 0.1*np.ones(len(dummy_cfg.param_ranges))
        self.lock_mgr = lock_mgr

    def run(self, Individual):

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        self.lock_mgr.increment_run()
        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.get_run()

        parameters = Individual

        error = [1-np.sqrt(np.mean((parameters - self.observations)**2))]
        # error = [1-np.sqrt(((parameters - self.observations)**2))]  # could be use as test for multiobj

        return error


class ModelMultObj():

    def __init__(self, dummy_cfg, lock_mgr):

        self.observations = 0.1*np.ones(len(dummy_cfg.param_ranges))
        self.lock_mgr = lock_mgr

    def run(self, Individual):

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        self.lock_mgr.increment_run()
        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.get_run()

        parameters = Individual

        error = [1-np.sqrt(((parameters - self.observations)**2))]

        return error


def test_deap(dummy_cfg):

    print('Test calibration routines')

    dummy_cfg.deap_param = DummyDEAPParameters()
    lock_mgr = calibration.LockManager(dummy_cfg.deap_param.num_cpus)

    model = ModelSingleObj(dummy_cfg, lock_mgr)

    calib_deap = calibration.CalibrationDeap(dummy_cfg, model.run)
    target = calib_deap.run(dummy_cfg.path_result, lock_mgr)

    assert target[0] > 0.


# def test_deap_mult_obj(dummy_cfg):

#     print('Test calibration routines')

#     dummy_cfg.deap_param = DummyDEAPParameters()
#     lock_mgr = calibration.LockManager(dummy_cfg.deap_param.num_cpus)

#     model = ModelMultObj(dummy_cfg, lock_mgr)

#     calib_deap = calibration.CalibrationDeap(dummy_cfg, model.run, len(dummy_cfg.param_ranges))
#     target = calib_deap.run(dummy_cfg.path_result, lock_mgr)

#     assert target[0] > 0.9
