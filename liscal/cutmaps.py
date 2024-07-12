#!/fws5/lb/user/macw/lisflow_efas5/local/lisflow_env/bin/python3

import os
import xarray as xr
import numpy as np
import pcraster as pcr

import dask
#from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
from multiprocessing.pool import ThreadPool

from liscal import pcr_utils

from datetime import datetime


def clip_pcr(filein, fileout, mask):
    """
    Clip a PCRaster map file using a mask and save the output.

    Parameters
    ----------
    filein : str
        Path to the input PCRaster map file.
    fileout : str
        Path to save the clipped PCRaster map file.
    mask : str
        Path to the PCRaster mask file used for clipping.

    Returns
    -------
    str
        Path to the output file.

    Notes
    -----
    The function applies different procedures based on the fileout name.
    It uses PCRaster commands for clipping and resampling.
    """

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

    return fileout


def copy_file(filein, fileout):
    """
    Copy a file from one location to another.

    Parameters
    ----------
    filein : str
        Path to the source file.
    fileout : str
        Path to the destination file.
    """
    os.system("cp " + filein + " " + fileout)
    return


def clip_netcdf(filenc, fileout, clip_box):
    """
    Clip a NetCDF file using a specified bounding box.

    Parameters
    ----------
    filenc : str
        Path to the input NetCDF file.
    fileout : str
        Path to save the clipped NetCDF file.
    clip_box : list of int
        Bounding box coordinates as [x_min, x_max, y_min, y_max].

    Raises
    ------
    Exception
        If lat/lon or x/y coordinates are not found in the dataset.
    """

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


def cut_map(maskpcr, filein, fileout, clip_box):
    """
    Cut a map file based on the file extension. It supports PCRaster map files (.map), NetCDF files (.nc).
    Other formats are not clipped but just copied over to the destination file.

    Parameters
    ----------
    maskpcr : str
        Path to the PCRaster mask file used for clipping .map files.
    filein : str
        Path to the input file.
    fileout : str
        Path to the output file.
    clip_box : list of int
        Bounding box coordinates used for clipping NetCDF files.

    Notes
    -----
    The function determines the file type based on its extension and applies appropriate clipping.
    """

    ext = filein[-4:][filein[-4:].find("."):]

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time, ': creating...',fileout)

    if ext == ".map":
        clip_pcr(filein, fileout, maskpcr)
    elif ext == ".nc":
        clip_netcdf(filein, fileout, clip_box)
    else:
        copy_file(filein, fileout)


def cut_maps_station(cfg, path_maps, stations_data, obsid):
    """
    Process map files for a given station by clipping them to the station's subcatchment area directory.

    Parameters
    ----------
    cfg : ConfigCalibration
        Configuration object containing paths and settings.
    path_maps : str
        Path to the directory containing map files.
    stations_data : str
        Path to the stations data file.
    obsid : int
        Observation ID of the station.

    Notes
    -----
    The function walks through the map files in the given directory, clips them based on the station's subcatchment area,
    and saves them to a specified output directory.
    """

    # prof = Profiler()
    # rprof = ResourceProfiler(dt=0.25)
    # cprof = CacheProfiler() #metric=nbytes)
    # prof.register()
    # rprof.register()
    # cprof.register()

    with dask.config.set({'scheduler': 'threads', 'array.chunk-size': '2048MiB', 'pool': ThreadPool(1)}):  # [distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads]

        subcatchment_path = os.path.join(cfg.subcatchment_path, str(obsid))
        path_subcatch_maps = os.path.join(subcatchment_path,'maps')

        # Cut bbox from ALL static maps and forcings for subcatchment
        maskpcr = os.path.join(path_subcatch_maps, 'mask.map')

        if os.path.isfile(maskpcr):
            maskmap = pcr.readmap(maskpcr)
        else:
            print('wrong input mask file')
            exit(1)

        masknp = pcr.pcr2numpy(maskmap, False)
        mask_filter = np.where(masknp)
        clip_box = []
        clip_box.append(np.min(mask_filter[1]))
        clip_box.append(np.max(mask_filter[1]))
        clip_box.append(np.min(mask_filter[0]))
        clip_box.append(np.max(mask_filter[0]))
        
        if os.path.isfile(path_maps) and os.path.getsize(path_maps) > 0:
            afile = os.path.basename(path_maps)
            fileout = os.path.join(path_subcatch_maps, afile)
            if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                print("skipping already existing %s" % fileout)
            else:
                cut_map(maskpcr, path_maps, fileout, clip_box)
        else:
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

    # visualize([prof, rprof, cprof], file_path='profile.html', show=False)
