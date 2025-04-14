#!/usr/bin/env python3
import xarray as xr
from glob import glob
import os
import pandas as pd
from lisflood.global_modules.settings import LisSettings, MaskInfo
from lisflood.global_modules.add1 import *
from pcraster import Scalar, numpy2pcr, Nominal, setclone, Boolean, pcr2numpy
from datetime import datetime, timedelta
from pcraster import *

from liscal import config
from liscal import templates, calibration, config, subcatchment, objective, hydro_model
import argparse

#########
#the script takes the station folder as input and performs calculation on files stored in maps folder
# and station_data.csv files 

var2remove=['time_bnds','crs','wgs_1984']

def ReadMe(nc_file,var=False):
    if type(nc_file) ==str:
        if os.path.basename(nc_file)[:-4]=='.map':
            nc_file=nc_file[:-4]

        try:
            ds=xr.open_mfdataset(nc_file+'.nc')
        except:
            ds=xr.open_mfdataset(nc_file)
        variables=list(ds.keys())
        for l in var2remove:
            while l in variables: variables.remove(l)
        da=ds[variables[0]]
        if var==False:
            return da
        else:
            return variables

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    settings_file = args.settings_file
    
    # Try to convert the input to an integer
    obsids = None
    try:
        obsid = int(args.station)
        obsids = [obsid]
    except ValueError:
        # If conversion fails, assume it's a file path and check if it exists
        if os.path.isfile(args.station):
            station_list = args.station
        else:
            print("Error: Input is neither a valid integer nor an existing file path.")
            exit(1)
        # read the list of station to process 
        try:
            CatchmentsToProcess = pd.read_csv(station_list,sep=",",header=None)
            obsids = CatchmentsToProcess[0]
            obsids = np.array(obsids)
        except:
            raise Exception('Error opening station txt: {} not found'.format(station_list))

    cfg = config.ConfigCalibration(settings_file, 0, 0)

    # cicle for all ids
    for obsid in obsids:
        print(f"==================== processing station {obsid} ====================")    

        subcatch = subcatchment.SubCatchment(cfg, obsid, create_links=False)

        maps_dir=os.path.join(os.path.join(subcatch.path,'maps'))
        station_data_file=os.path.join(os.path.join(subcatch.path_station,'station_data.csv'))
        station_data=pd.read_csv(station_data_file,index_col=0)

        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
        
        lock_mgr = calibration.LockManager(cfg.num_cpus)

        obj = objective.ObjectiveKGE(cfg, subcatch)

        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

        prerun_file, run_file = model.init_settings()

        settings = LisSettings(prerun_file)
        binding=settings.binding

        # open fraction file -> this should be done without hardcoded name
        for_frac=ReadMe(f"{binding['ForestFraction']}",var=False)
        oth_frac=ReadMe(f"{binding['OtherFraction']}",var=False)
        pixarea=ReadMe(f"{binding['PixelAreaUser']}",var=False)

        # for_frac=xr.open_dataset(os.path.join(maps_dir,'fracother.nc'))['Band1']
        # oth_frac=xr.open_dataset(os.path.join(maps_dir,'fracforest.nc'))['Band1']
        # pixarea=xr.open_dataset(os.path.join(maps_dir,'pixarea.nc'))['Band1']

        mask_file=os.path.join(maps_dir,'masksmall.map')


        setclone(mask_file)

        mask=pcraster.boolean(iterReadPCRasterMap(mask_file))

        mask_array=pcr2numpy(mask,np.nan)
        # fraction used to calculate Budyko
        keep_frac=for_frac+oth_frac
        # get spinup days to remove from calculation
        spinup=station_data.loc['Spinup_days'].item()
        # get lisflood timestep
        ts=float(station_data.loc['CAL_TYPE'].item())
        StepsInOneDAy=86400/(ts*3600)

        # get calibration dates
        cal_start = (datetime.strptime(station_data.loc['Obs_start'].item(),"%d/%m/%Y %H:%M") - timedelta(days=float(spinup))).strftime('%d/%m/%Y %H:%M')
        cal_end = datetime.strptime(station_data.loc['Obs_end'].item(),"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

        # open meteo precipitation and meteo maps and compute total
        pr_da=ReadMe(f"{binding['PrecipitationMaps']}",var=False)
        et_da=ReadMe(f"{binding['ET0Maps']}",var=False)
        pr=pr_da.sel(time=slice(cal_start,cal_end)).sum(dim='time')/StepsInOneDAy
        et=et_da.sel(time=slice(cal_start,cal_end)).sum(dim='time')/StepsInOneDAy
        # compute total precipitation ans PET for the calibration for the catchment
        pr_tot=((pr*keep_frac*pixarea).where(mask_array)).sum().values/((pixarea).where(mask_array)).sum().values
        et_tot=((et*keep_frac*pixarea).where(mask_array)).sum().values/((pixarea).where(mask_array)).sum().values

        # use Precipitation and EvapoTranspiration to compute aridity index ...
        # ... and get the lower value for all pixels of the mask map
        # (we can then use this index to check if it is lower than 0.5 and activate TransmissionLoss param calibration)
        pr_tot_pixels=pr_da.where(mask_array).sum(dim='time')
        et_tot_pixels=et_da.where(mask_array).sum(dim='time')
        AridityIndex_min = (pr_tot_pixels/et_tot_pixels).min().values


        # Compute minimum temperature for SnowMelt parameter enabling check
        Tavg_da=ReadMe(f"{binding['TavgMaps']}",var=False)
        Tavg=Tavg_da.sel(time=slice(cal_start,cal_end)).min(dim='time')
        ElevationStD=ReadMe(f"{binding['ElevationStD']}",var=False)
        DeltaTSnow = 0.9674 * ElevationStD * float(binding['TemperatureLapseRate'])
        TavgS = Tavg + DeltaTSnow * (- 1)
        TavgS_min=TavgS.where(mask_array).min().values

        # update station data file
        StationDataFile=pd.read_csv(station_data_file,index_col=0)
        StationDataFile.loc["precip_budyko"]=str(pr_tot)
        StationDataFile.loc["PET_budyko"]=str(et_tot)
        StationDataFile.loc["min_TAvgS"]=str(TavgS_min)
        StationDataFile.loc["min_AridIdx"]=str(AridityIndex_min)
        StationDataFile.to_csv(station_data_file)

        del mask,mask_array

    print("==================== END ====================")
