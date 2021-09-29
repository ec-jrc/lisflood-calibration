import time
import multiprocessing as mp

from liscal import calibration


def test_init():
    lock_mgr = calibration.MultiprocessingScheduler(num_cpus=4)
    assert lock_mgr.num_cpus == 4

def test_no_mp():
    lock_mgr = calibration.MultiprocessingScheduler(num_cpus=1)
    assert lock_mgr.num_cpus == 1
