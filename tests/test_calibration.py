from os import path
import pandas
import numpy as np
import random
import pytest

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

    def __init__(self, lock_mgr, target):

        # target = 0.1 for all params
        self.target = target
        self.lock_mgr = lock_mgr

    def objectives(self, parameters):
        NotImplemented

    def run(self, Individual):

        self.lock_mgr.increment_run()
        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.get_run()

        obj = self.objectives(Individual)

        return obj


class ModelSingleObj(ModelDummy):

    def __init__(self, lock_mgr, target):
        super().__init__(lock_mgr, target)

    def objectives(self, parameters):
        obj = [1-np.sqrt(np.mean((parameters - self.target)**2))]
        return obj


class ModelMultObj(ModelDummy):

    def __init__(self, lock_mgr, target):
        super().__init__(lock_mgr, target)

    def objectives(self, parameters):
        obj = 1-np.sqrt(((parameters - self.target)**2))
        return obj


def test_deap_single_obj(dummy_cfg):

    print('Test calibration single objective')

    dummy_cfg.deap_param = DummyDEAPParameters()
    lock_mgr = calibration.LockManager(dummy_cfg.deap_param.num_cpus)
    
    n_param = len(dummy_cfg.param_ranges)
    target = np.arange(1, n_param+1)/n_param
    model = ModelSingleObj(lock_mgr, target)

    calib_deap = calibration.CalibrationDeap(dummy_cfg, model.run)
    target = calib_deap.run(dummy_cfg.path_result, lock_mgr)

    assert target[0] > 0.9


@pytest.mark.parametrize('value', [0.2, 0.4, 0.6, 0.8])
def test_deap_mult_obj(dummy_cfg, value):

    print('Test calibration multi objectives')

    dummy_cfg.deap_param = DummyDEAPParameters()
    lock_mgr = calibration.LockManager(dummy_cfg.deap_param.num_cpus)
    
    target = value*np.ones(len(dummy_cfg.param_ranges))
    model = ModelMultObj(lock_mgr, target)

    calib_deap = calibration.CalibrationDeap(dummy_cfg, model.run, len(dummy_cfg.param_ranges))
    target = calib_deap.run(dummy_cfg.path_result, lock_mgr)

    assert target[0] > 0.99
