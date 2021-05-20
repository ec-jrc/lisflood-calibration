import time
import multiprocessing as mp

from liscal import calibration


def inc_gen(lock_mgr):
    for i in range(20):
        time.sleep(0.01)
        lock_mgr.increment_gen()


def inc_run(lock_mgr):
    for i in range(10):
        time.sleep(0.01)
        lock_mgr.increment_run()


def test_init():
    lock_mgr = calibration.LockManager()
    assert lock_mgr.get_gen() == -1
    assert lock_mgr.get_run() == -1


def test_set_gen():
    lock_mgr = calibration.LockManager()
    lock_mgr.set_gen(10)
    assert lock_mgr.get_gen() == 10
    assert lock_mgr.get_run() == 0


def test_set_run():
    lock_mgr = calibration.LockManager()
    lock_mgr.set_run(20)
    assert lock_mgr.get_gen() == -1
    assert lock_mgr.get_run() == 20


def test_increment_gen():
    lock_mgr = calibration.LockManager()

    lock_mgr.set_gen(0)
    assert lock_mgr.get_gen() == 0
    assert lock_mgr.get_run() == 0
    with mp.Pool(processes=4) as pool:
        pool.map(inc_gen, [lock_mgr for i in range(10)])
    assert lock_mgr.get_gen() == 200
    assert lock_mgr.get_run() == 0


def test_increment_run():
    lock_mgr = calibration.LockManager()
    lock_mgr.set_gen(0)
    assert lock_mgr.get_gen() == 0
    assert lock_mgr.get_run() == 0
    with mp.Pool(processes=4) as pool:
        pool.map(inc_run, [lock_mgr for i in range(10)])
    assert lock_mgr.get_gen() == 0
    assert lock_mgr.get_run() == 100
