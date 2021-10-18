import os
import numpy as np
import multiprocessing as mp


def get_scheduler(scheduler='Multiprocessing', num_cpus=1):
    if scheduler == 'Dask':
        return DaskScheduler(num_cpus)
    elif scheduler == 'Multiprocessing':
        return MultiprocessingScheduler(num_cpus)
    elif scheduler == 'MPI':
        return MPIScheduler(num_cpus)
    else:
        raise Exception('Scheduler {} not supported'.format(scheduler))


class Scheduler():

    def create_mapping(self):
        raise NotImplementedError

    def close(self):
        return

    def chunk(self, values):
        return values

    def broadcast(self, values):
        return values
    
    def gather(self, values):
        return values
    
    def root(self):
        return True
    
    def sequence(self, func, args):
        output = func(*args)
        return output


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

        from dask import distributed
        if os.path.isfile('scheduler.json'):
            self.client = distributed.Client(scheduler_file='scheduler.json')
        else:
            cluster = distributed.LocalCluster(n_workers=num_cpus, processes=False)
            self.client = distributed.Client(cluster)
        print(self.client)
        for core in self.client.ncores():
            print(core)
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


class MPIScheduler(Scheduler):

    def __init__(self, num_cpus):
        
        from mpi4py import MPI
        # global comm
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.name = MPI.Get_processor_name()

        # local comm
        mgr = mp.Manager()
        self.lock = mgr.Lock()
        self.num_cpus = num_cpus

    def chunk(self, values):
        chunk_size = int(np.ceil(len(values)/self.size))
        return values[self.rank*chunk_size:(self.rank+1)*chunk_size]

    def create_mapping(self):
        par_pool = mp.Pool(processes=self.num_cpus, initargs=(self.lock,))
        self.pool = par_pool
        return par_pool.map

    def gather(self, values):
        values = list(values)
        values = self.comm.allgather(values)
        values = [j for sub in values for j in sub] # flatten the array
        return values

    def broadcast(self, values):
        values = self.comm.bcast(values, root=0)
        return values

    def root(self):
        return self.rank == 0

    def sequence(self, func, *fargs):
        for i in range(self.size):
            if self.rank == i:
                output = func(*fargs)
            self.comm.Barrier()
        return output
