import os
import contextlib
from dask import distributed
import threading
import multiprocessing as mp
from multiprocessing import pool


def get_scheduler(scheduler='Multiprocessing', num_cpus=1):
    if scheduler == 'Dask':
        return DaskScheduler(num_cpus)
    elif scheduler == 'Multiprocessing':
        return MultiprocessingScheduler(num_cpus)
    else:
        raise Exception('Scheduler {} not supported'.format(scheduler))


class Scheduler():

    def create_mapping(self):
        raise NotImplementedError

    def close(self):
        return

    def distribute(self, values):
        return values
    
    def gather(self, values):
        return values


class MultiprocessingScheduler(Scheduler):

    def __init__(self, num_cpus):

        mgr = mp.Manager()
        self.lock = mgr.Lock()
        self.num_cpus = num_cpus

    def create_mapping(self):
        if self.num_cpus > 1:
            # par_pool = pool.ThreadPool(processes=self.num_cpus, initargs=(self.lock,))
            par_pool = mp.Pool(processes=self.num_cpus, initargs=(self.lock,))
            self.pool = par_pool
            return par_pool.map
        else:
            return map

    def gather(self, values):
        return list(values)

    def close(self):
        if hasattr(self, 'pool'):
            self.pool.close()


class DaskScheduler(Scheduler):

    def __init__(self, num_cpus):
        super().__init__()
        if os.path.isfile('scheduler.json'):
            self.client = distributed.Client('scheduler.json')
        else:
            cluster = distributed.LocalCluster(n_workers=num_cpus, processes=False)
            self.client = distributed.Client(cluster)
        print(self.client)
        self.lock = distributed.Lock()
        self.num_cpus = num_cpus
        assert self.num_cpus == len(self.client.ncores())
        # check python env is the same on client and scheduler/workers
        self.client.get_versions(check=True)
        
    def create_mapping(self):
        return self.client.map

    def gather(self, futures):
        return self.client.gather(futures)

    def close(self):
        self.client.shutdown()
