from os import path
import pandas
import numpy as np
import random
from liscal import calibration


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


class ModelDummy():

    def __init__(self, dummy_cfg, lock_mgr):

        self.observations = 0.1*np.ones(len(dummy_cfg.param_ranges))
        self.lock_mgr = lock_mgr

    def objectives(self, parameters):
        NotImplemented

    def run(self, Individual):

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        self.lock_mgr.increment_run()
        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.get_run()

        obj = self.objectives(Individual)

        return obj


class ModelSingleObj(ModelDummy):

    def __init__(self, dummy_cfg, lock_mgr):
        super().__init__(dummy_cfg, lock_mgr)

    def objectives(self, parameters):
        obj = [1-np.sqrt(np.mean((parameters - self.observations)**2))]
        return obj


class ModelMultObj(ModelDummy):

    def __init__(self, dummy_cfg, lock_mgr):
        super().__init__(dummy_cfg, lock_mgr)

    def objectives(self, parameters):
        obj = 1-np.sqrt(((parameters - self.observations)**2))
        return obj


def test_deap_single_obj(dummy_cfg):

    print('Test calibration routines')

    dummy_cfg.deap_param = DummyDEAPParameters()
    lock_mgr = calibration.LockManager(dummy_cfg.deap_param.num_cpus)

    model = ModelSingleObj(dummy_cfg, lock_mgr)

    calib_deap = calibration.CalibrationDeap(dummy_cfg, model.run)
    target = calib_deap.run(dummy_cfg.path_result, lock_mgr)

    assert target[0] > 0.9


def test_deap_mult_obj(dummy_cfg):

    print('Test calibration routines')

    dummy_cfg.deap_param = DummyDEAPParameters()
    lock_mgr = calibration.LockManager(dummy_cfg.deap_param.num_cpus)

    model = ModelMultObj(dummy_cfg, lock_mgr)

    calib_deap = calibration.CalibrationDeap(dummy_cfg, model.run, len(dummy_cfg.param_ranges))
    target = calib_deap.run(dummy_cfg.path_result, lock_mgr)

    assert target[0] > 0.99
