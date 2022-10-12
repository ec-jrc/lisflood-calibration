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
    lock_mgr = calibration.LockManager(num_cpus=4)
    assert lock_mgr.get_gen() == -1
    assert lock_mgr.get_run() == -1


def test_no_mp():
    lock_mgr = calibration.LockManager(num_cpus=1)

    lock_mgr.set_gen(0)
    assert lock_mgr.get_gen() == 0
    assert lock_mgr.get_run() == 0
    mapping, pool = lock_mgr.create_mapping()
    for i in range(2):
        inc_gen(lock_mgr)
    assert lock_mgr.get_gen() == 40
    assert lock_mgr.get_run() == 0
    if pool:
        assert False


def test_set_gen():
    lock_mgr = calibration.LockManager(num_cpus=4)
    lock_mgr.set_gen(10)
    assert lock_mgr.get_gen() == 10
    assert lock_mgr.get_run() == 0


def test_set_run():
    lock_mgr = calibration.LockManager(num_cpus=4)
    lock_mgr.set_run(20)
    assert lock_mgr.get_gen() == -1
    assert lock_mgr.get_run() == 20


def test_increment_gen():
    lock_mgr = calibration.LockManager(num_cpus=4)

    lock_mgr.set_gen(0)
    assert lock_mgr.get_gen() == 0
    assert lock_mgr.get_run() == 0
    mapping, pool = lock_mgr.create_mapping()
    mapping(inc_gen, [lock_mgr for i in range(4)])
    assert lock_mgr.get_gen() == 80
    assert lock_mgr.get_run() == 0
    pool.close()


def test_increment_run():
    lock_mgr = calibration.LockManager(num_cpus=4)
    lock_mgr.set_gen(0)
    assert lock_mgr.get_gen() == 0
    assert lock_mgr.get_run() == 0
    with mp.Pool(processes=4) as pool:
        pool.map(inc_run, [lock_mgr for i in range(4)])
    assert lock_mgr.get_gen() == 0
    assert lock_mgr.get_run() == 40
