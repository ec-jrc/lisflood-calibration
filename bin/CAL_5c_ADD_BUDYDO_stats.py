
import xarray as xr
from glob import glob
import os
import pandas as pd
from lisflood.global_modules.settings import LisSettings, MaskInfo
from lisflood.global_modules.add1 import *
from pcraster import Scalar, numpy2pcr, Nominal, setclone, Boolean, pcr2numpy
from datetime import datetime, timedelta
from pcraster import *
from configparser import ConfigParser
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



input_dir = os.path.normpath(sys.argv[1])
maps_dir=os.path.join(os.path.join(input_dir,'maps'))
station_data_file=os.path.join(os.path.join(input_dir,'station','station_data.csv'))
station_data=pd.read_csv(station_data_file,index_col=0)
# settingsfile
settings_file=glob(os.path.join(input_dir,'settings','*.xml'))[0]
settings = LisSettings(settings_file)
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
# update station data file
StationDataFile=pd.read_csv(station_data_file,index_col=0)
StationDataFile.loc["precip_budyko"]=str(pr_tot)
StationDataFile.loc["PET_budyko"]=str(et_tot)
StationDataFile.to_csv(station_data_file)

del mask,mask_array
