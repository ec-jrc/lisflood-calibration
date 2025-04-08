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


def clip_netcdf(ds, fileouts, clip_boxes):
    """
    Clip a NetCDF file using a specified bounding box.

    Parameters
    ----------
    ds : str
        Xarray Dataset already in time chunck (if any)
    fileout : str
        Path to save the clipped NetCDF file.
    clip_box : list of int
        Bounding box coordinates as [x_min, x_max, y_min, y_max].

    Raises
    ------
    Exception
        If lat/lon or x/y coordinates are not found in the dataset.
    """

    if ds is None:
        raise Exception('Error, map not loaded correctly to generate output {}:\n'.format(fileout))

    ds_outs = []
    ds_outs_filenames = []
    for clip_box, fileout in zip(clip_boxes,fileouts):
        if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
            print("skipping already existing %s" % fileout)
        else:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, ': creating...',fileout)
        x_min, x_max, y_min, y_max = clip_box

        if 'lon' in ds.coords and 'lat' in ds.coords:
            ds_out = ds.isel(lat=range(y_min, y_max + 1), lon=range(x_min, x_max + 1))
        elif 'x' in ds.coords and 'y' in ds.coords:
            ds_out = ds.isel(y=range(y_min, y_max + 1), x=range(x_min, x_max + 1))
        else:
            raise Exception('Could not find lat/lon or x/y coordinates in dataset:\n {}'.format(ds))
        ds_outs.append(ds_out)
        ds_outs_filenames.append(fileout)

    for ds_out,ds_outs_filename  in zip(ds_outs,ds_outs_filenames):
        ds_out.to_netcdf(ds_outs_filename)
        ds_out.close()

def cut_map(maskpcrs, filein, fileouts, clip_boxes):
    """
    Cut a map file based on the file extension. It supports PCRaster map files (.map), NetCDF files (.nc).
    Other formats are not clipped but just copied over to the destination file.

    Parameters
    ----------
    maskpcrs : list(str)
        Paths to the PCRaster mask files used for clipping .map files.
    filein : str
        Path to the input file.
    fileouts : list(str)
        Paths to the output files.
    clip_boxes : list(array[int]) 
        Bounding box coordinates used for clipping NetCDF files.

    Notes
    -----
    The function determines the file type based on its extension and applies appropriate clipping.
    """

    ext = filein[-4:][filein[-4:].find("."):]

    if ext == ".map":
        for mask, fileout in zip(maskpcrs,fileouts):
            if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                print("skipping already existing %s" % fileout)
            else:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(current_time, ': creating...',fileout)
                clip_pcr(filein, fileout, mask)
    elif ext == ".nc":
        ds = xr.open_dataset(filein)
        if 'time' in ds.coords:
            chunks = {coord: 'auto' for coord in ds.coords}
            ds = ds.chunk(chunks)        
        clip_netcdf(ds, fileouts, clip_boxes)
        ds.close()
    else:
        for fileout in fileouts:
            if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                print("skipping already existing %s" % fileout)
            else:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(current_time, ': creating...',fileout)
                copy_file(filein, fileout)


def load_mask(mask_path):
    if os.path.isfile(mask_path):
        maskmap = pcr.readmap(mask_path)
        masknp = pcr.pcr2numpy(maskmap, False)
        mask_filter = np.where(masknp)
        clip_box = [np.min(mask_filter[1]), np.max(mask_filter[1]), np.min(mask_filter[0]), np.max(mask_filter[0])]
        return clip_box
    else:
        print('Wrong input mask file:', mask_path)
        return None


def cut_maps_stations(cfg, path_maps, obsids, useDaskConfig=False):
    if isinstance(obsids, int):
        obsids = [obsids]
    """
    Process map files for a given station or multiple stestions by clipping them to the station's subcatchment area directory.

    Parameters
    ----------
    cfg : ConfigCalibration
        Configuration object containing paths and settings.
    path_maps : str
        Path to the directory containing map files.
    obsids : int or list of ints
        Observation ID (or IDs) of the station(s).

    Notes
    -----
    The function walks through the map files in the given directory, clips them based on the station's subcatchment area,
    and saves them to a specified output directory.
    """
    if useDaskConfig:
        with dask.config.set({'scheduler': 'threads', 'array.chunk-size': '2048MiB', 'pool': ThreadPool(1)}):  # [distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads]
            _cut_maps_stations(cfg, path_maps, obsids)
    else:
        _cut_maps_stations(cfg, path_maps, obsids)

def _cut_maps_stations(cfg, path_maps, obsids):
    # prof = Profiler()
    # rprof = ResourceProfiler(dt=0.25)
    # cprof = CacheProfiler() #metric=nbytes)
    # prof.register()
    # rprof.register()
    # cprof.register()
    
    maskpcrs = []
    clip_boxes = []
    for obsid in obsids:        
        maskpcr = os.path.join(cfg.subcatchment_path, str(obsid), 'maps', 'mask.map')
        maskpcrs.append(maskpcr)
        clip_box = load_mask(maskpcr)
        if clip_box:
            clip_boxes.append(clip_box)

    if os.path.isfile(path_maps) and os.path.getsize(path_maps) > 0:
        afile = os.path.basename(path_maps)
        fileouts = []
        for obsid in obsids:
            fileout = os.path.join(cfg.subcatchment_path, str(obsid), 'maps', afile)
            fileouts.append(fileout)
        cut_map(maskpcrs, path_maps, fileouts, clip_boxes)
    else:
        # Enter in maps dir and walk through subfolders
        for root, dirs, files in os.walk(path_maps, topdown=False, followlinks=True):
            for afile in files:
                fileouts = []
                for obsid in obsids:
                    fileout = os.path.join(cfg.subcatchment_path, str(obsid),'maps', afile)
                    fileouts.append(fileout)
                filenc = os.path.join(root, afile)
                if filenc.find("bak") > -1:
                    continue
                cut_map(maskpcrs, filenc, fileouts, clip_boxes)

    # visualize([prof, rprof, cprof], file_path='profile.html', show=False)