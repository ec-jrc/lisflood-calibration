#!/bin/ksh

# RUN LISFLOOD MODEL
time /scratch/rd/nedd/python/virtualEnv_2.7.12-01/bin/python /perm/rd/nedd/EFAS/efasCalib/input/src/lisflood/lisf1.py %prerun -t
time /scratch/rd/nedd/python/virtualEnv_2.7.12-01/bin/python /perm/rd/nedd/EFAS/efasCalib/input/src/lisflood/lisf1.py %run -t

# # Memory profiler
# /scratch/rd/nedd/python/virtualEnv_2.7.12-01/bin/python -B -m memory_profiler /scratch/rd/nedd/python/virtualEnv_2.7.12-01/bin/mprof run /perm/rd/nedd/EFAS/efasCalib/input/src/lisflood/lisf1.py %prerun -t
# 
# /scratch/rd/nedd/python/virtualEnv_2.7.12-01/bin/python -B -m memory_profiler /scratch/rd/nedd/python/virtualEnv_2.7.12-01/bin/mprof run /perm/rd/nedd/EFAS/efasCalib/input/src/lisflood/lisf1.py %run -t
