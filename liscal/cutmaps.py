#!/fws5/lb/user/macw/lisflow_efas5/local/lisflow_env/bin/python3

import os
import xarray as xr
import numpy as np
import pcraster as pcr

import dask
from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
from multiprocessing.pool import ThreadPool

from liscal import pcr_utils


def clip_pcr(filein, fileout, mask):

    pcr.setclone(mask)
    
    if fileout.find("outlets") == -1 and fileout.find("res.") == -1 and fileout.find("lakes") == -1:
        # load the small mask we use to clip the input file with
        pcr.setclone(mask)

        # Use PCRasters pcrcalc with ifthen to generate a map with missing values (mv) where a condition is not met (value = 0)
        maskSmall = pcr.ifthen(mask, filein)
        pcr.report(maskSmall, fileout)

        # Resample with PCRaster, which cuts maps where it's set to mv
        # Not to be confused with the resampling as gdal warp does
        pcr_utils.pcrasterCommand('resample -c 0 F0 F1', {"F0": fileout, "F1": fileout+'.tmp'})
        os.system('mv ' + fileout + '.tmp ' + fileout)
    if fileout.find("ldd") > -1:
        ldd = pcr.readmap(fileout)
        pcr.setclone(fileout)
        lddr = pcr.lddrepair(ldd)
        pcr.report(lddr, fileout)
    # print(fileout)
    # print(pcr.pcr2numpy(pcr.readmap(fileout), np.nan))
    # print('PCRaster map {} done'.format(fileout))
    return fileout


def copy_file(filein, fileout):
    os.system("cp " + filein + " " + fileout)
    # print('File {} copied'.format(fileout))
    return


def clip_netcdf(filenc, fileout, clip_box):

    x_min = clip_box[0]    
    x_max = clip_box[1]    
    y_min = clip_box[2]    
    y_max = clip_box[3]    
    
    ds = xr.open_dataset(filenc)
    if 'time' in ds.coords:
       chunks = {coord: 'auto' for coord in ds.coords}
       ds = ds.chunk(chunks)
    
    if 'lon' in ds.coords and 'lat' in ds.coords:
        ds_out = ds.isel(lat=range(y_min, y_max + 1), lon=range(x_min, x_max + 1))
    elif 'x' in ds.coords and 'y' in ds.coords:
        ds_out = ds.isel(y=range(y_min, y_max + 1), x=range(x_min, x_max + 1))
    else:
        raise Exception('Could not find lat/lon or x/y coordinates in dataset:\n {}'.format(ds))

    ds_out.to_netcdf(fileout)
    
    ds.close()
    ds_out.close()

    # print('NetCDF file {} done'.format(filenc))


def cut_map(maskpcr, filenc, fileout, clip_box):

  ext = filenc[-4:][filenc[-4:].find("."):]

  print('creating...',fileout)
  if ext == ".map":
      clip_pcr(filenc, fileout, maskpcr)
  elif ext == ".nc":
      clip_netcdf(filenc, fileout, clip_box)
  else:
      copy_file(filenc, fileout)


def cut_maps_station(cfg, path_maps, stations_data, obsid):

    prof = Profiler()
    rprof = ResourceProfiler(dt=0.25)
    cprof = CacheProfiler() #metric=nbytes)
    prof.register()
    rprof.register()
    cprof.register()

    with dask.config.set(scheduler='threads'):  # [distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads]

        subcatchment_path = os.path.join(cfg.subcatchment_path, str(obsid))
        path_subcatch_maps = os.path.join(subcatchment_path,'maps')

        # Cut bbox from ALL static maps and forcings for subcatchment
        maskpcr = os.path.join(path_subcatch_maps, 'mask.map')

        if os.path.isfile(maskpcr):
            print('maskmap',maskpcr)
            maskmap = pcr.readmap(maskpcr)
        else:
            print('wrong input mask file')
            exit(1)

        masknp = pcr.pcr2numpy(maskmap, False)
        mask_filter = np.where(masknp)
        clip_box = []
        clip_box.append(np.min(mask_filter[1]))
        clip_box.append(np.min(mask_filter[1]))
        clip_box.append(np.min(mask_filter[0]))
        clip_box.append(np.min(mask_filter[0]))
        
        # Enter in maps dir and walk through subfolders
        for root, dirs, files in os.walk(path_maps, topdown=False, followlinks=True):
            for afile in files:
                
                fileout = os.path.join(path_subcatch_maps, afile)
                
                if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                    print("skipping already existing %s" % fileout)
                    continue
                
                else:
                    filenc = os.path.join(root, afile)
                    if filenc.find("bak") > -1:
                        continue
                    cut_map(maskpcr, filenc, fileout, clip_box)

    print('finito...')

    # visualize([prof, rprof, cprof], file_path='profile.html', show=False)
