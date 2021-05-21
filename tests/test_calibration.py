import numpy as np

from liscal import calibration


def test_deap(dummy_cfg, dummy_model):

    lock_mgr = calibration.LockManager()

    calib_deap = calibration.CalibrationDeap(dummy_cfg, dummy_model.run)
    calib_deap.run(dummy_cfg.path_result, lock_mgr)

    assert False
    