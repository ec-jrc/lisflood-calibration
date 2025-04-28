#!/usr/bin/env python3
import os
import argparse
import pandas as pd

import rasterio.features
import xarray as xr
import pcraster as pcr
import geopandas as gpd


def pcr_setclone_aux(input_nc):
    'Get rows and cols number from the reference nc, for creating the clone needed for the pcraster'
    x_checks = ['lon', 'x', 'rlon']
    y_checks = ['lat', 'y', 'rlat']
    x_proj = set(list(input_nc.coords)) & set(x_checks)
    y_proj = set(list(input_nc.coords)) & set(y_checks)

    if len(x_proj)!=1 or len(y_proj)!=1:
        print('Input dataset coords names for lat/lon are not as expected.')
        print(f'The available coords are: {list(input_nc.coords)}')
        print(f'The checked names are {y_checks} and {x_checks} for lat and lon respectively.')
        exit(1)
    
    x_proj = list(x_proj)[0]
    y_proj = list(y_proj)[0]
    
    rows, cols = len(input_nc[y_proj]), len(input_nc[x_proj])
    return rows, cols


def vectorize(raster_mask):
    'vectorize raster masked data'
    # convert boolean xarry mask to shapefile; connectivity 8 as ldd allows, so we combine also diagonally as ldd allows (default is 4)
    maskShape = rasterio.features.shapes(raster_mask.notnull().astype('uint8'), transform=raster_mask.rio.transform(), connectivity=8)  
    mypoly=[]
    for vec in maskShape:
        if vec[1]!=0:
            mypoly.append(vec[0])

    mypoly = [{'geometry': i, 'properties': None} for i in mypoly]
    mypoly = gpd.GeoDataFrame.from_features(mypoly)
    return mypoly


def get_mask(catch_id):
    'get the shapefile mask of the intercatchment used for the calibration'

    # get ldd as auxiliary for getting the xarray info
    ldd = os.path.join(main_dir, 'catchments', str(catch_id), 'maps/ldd.nc')
    ldd = xr.open_dataset(ldd)
    projection = ldd['crs'].spatial_ref
    ldd = ldd['Band1']

    # define clone for pcraster, otherwise it gets any clone available which can be wrong
    rows, cols = pcr_setclone_aux(ldd)
    pcr.setclone(rows, cols, 1, 0, 0)

    # get the mask map and derive the mask xarray
    maskmap = os.path.join(main_dir, 'catchments', str(catch_id), 'maps/masksmall.map')
    maskmap = pcr.readmap(maskmap)
    maskmap = pcr.pcr2numpy(maskmap, 0)
    maskmap = ldd.fillna(0)*0+maskmap
    maskmap = maskmap.where(maskmap==1)

    shapefile_catchment = vectorize(maskmap) 
    shapefile_catchment.index = [catch_id]
    shapefile_catchment = shapefile_catchment.set_crs(projection)
    return shapefile_catchment


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    args = parser.parse_args()

    settings_file = args.settings_file
    
    # get the main directory from where inputs and outputs will be read/saved
    main_dir = settings_file.replace('/catchments/settings.txt', '')
    
    # read stations metadata and keep columns of interest
    metadata_stations = os.path.join(main_dir, 'data/stations/stations_data.csv')
    metadata_stations = pd.read_csv(metadata_stations)
    columns_kept = ['ObsID', 'StationName', 'River', 'Catchment', 'EC_Catchments',
                    'DrainingArea.km2.Provider', 'StationLon', 'StationLat', 'Height',
                    'DrainingArea.km2.LDD5k', 'LisfloodX5k', 'LisfloodY5k', 
                    'DrainingArea.km2.LDD', 'LisfloodX', 'LisfloodY',
                   ]
    columns_kept = list(set(columns_kept)&set(metadata_stations.columns))
    metadata_stations = metadata_stations[columns_kept]

    ################
    # NUMERIC DATA #
    ################
    # read calibration results for each catchment
    print('Reading pHistoryWRanks.csv file for each catchment...')
    calibration_data = []
    for i_catch in metadata_stations.ObsID:
        try:
            i_data = pd.read_csv(os.path.join(main_dir, 'catchments', str(i_catch), 'pHistoryWRanks.csv')).iloc[[0]]
            i_data.index = [i_catch]
            calibration_data.append(i_data)
        except:
            print(f'No pHistoryWRanks.csv file for catchment {i_catch}!')
    
    # combine all results
    calibration_data = pd.concat(calibration_data, axis=0)

    # merge with the metadata and save final sv file
    data_all = pd.merge(left=metadata_stations, right=calibration_data, left_on='ObsID', right_index=True, how='outer')
    data_all.to_csv(os.path.join(main_dir, 'summary/calibration_summary.csv'))

    ################
    # SPATIAL DATA #
    ################
    # get shapefile of intercatchment incase plotting of whole domain instead of only the outlet is preferred
    print('Generating shapefile for each catchment...')
    shapefile_data = []
    for i_catch in data_all.ObsID:
        try:
            i_data = get_mask(i_catch)
            shapefile_data.append(i_data)
        except:
            print(f'No spatial data for catchment {i_catch}!')

    if len(shapefile_data)>0:
        # combine all results
        shapefile_data = pd.concat(shapefile_data, axis=0)
        
        data_all = pd.merge(left=data_all, right=shapefile_data, left_on='ObsID', right_index=True, how='outer')
        data_all = gpd.GeoDataFrame(data_all, geometry=data_all.geometry)    
    
        data_all.to_file(os.path.join(main_dir, 'summary/calibration_summary_json.json'), driver="GeoJSON")

    print("==================== END ====================")