#!/bin/ksh

# RUN LISFLOOD MODEL
time /perm/rd/nedd/python/virtualEnv_2.7.12-01/bin/python /perm/rd/nedd/EFAS/efasCalib/input/src/lisflood/lisf1.py %prerun -t
time /perm/rd/nedd/python/virtualEnv_2.7.12-01/bin/python /perm/rd/nedd/EFAS/efasCalib/input/src/lisflood/lisf1.py %run -t
