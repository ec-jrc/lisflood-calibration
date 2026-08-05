#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import xarray as xr

def handle_empty_rows(interstation, all_catchment_data):
    # Iterate over each row in all_catchment_data
    for idx, row in all_catchment_data.iterrows():
        if row.isnull().any():  # Check if any column in the row is NaN
            id_value = idx
            print(f"Warning: Row with ID {id_value} contains empty values. Replacing interstation points with -1.")
            # Replace interstation values corresponding to this ID with -1
            interstation[interstation == id_value] = -1

### creates parameter maps based on the interstation map and the parameter values for each catchment ID, with an option to use nearest neighbor interpolation for invalid points
### values included into catchment_data will be replaced by catchments data values, 
### values not included into catchment_data (e.g. 0 or any other ID not found in catchments) will be replaced by default values from param_ranges
### values NaN will be NaN in the output maps
### values -1 will be replaced by the value of the nearest valid point (if useNNintepolation is True) or NaN (if useNNintepolation is False)
def create_param_mapping(interstation, param_ranges, all_catchment_data, useNNintepolation=True):
    # Initialize dictionary to store parameter maps
    param_maps = {param: np.zeros(interstation.shape) for param in param_ranges.index}
    
    # Determine the maximum ID to size the parameter value arrays
    max_id = all_catchment_data.index.max()

    # Create mask for valid indices (not NaN and not -1)
    valid_mask = ~np.isnan(interstation) & (interstation != -1)

    # Create mask for -1 indices
    invalid_mask = (interstation == -1)
    
    # Get coordinates of valid and invalid points
    coords = np.indices(interstation.shape).reshape(2, -1).T

    valid_coords = coords[valid_mask.flatten()]
    if useNNintepolation:
        invalid_coords = coords[invalid_mask.flatten()]
    
        # Create a KDTree for valid points
        tree = KDTree(valid_coords)
    
        # Find nearest valid point for each invalid point
        _, nearest_valid_indices = tree.query(invalid_coords, k=1)

    for param in param_ranges.index:
        default_value = param_ranges.loc[param, 'DefaultValue']
        
        id_to_param_series = all_catchment_data[f"{param}"]
        param_values = id_to_param_series.reindex(range(max_id + 1), fill_value=default_value).values
        
        # Start with default values for all
        param_maps[param] = np.full(interstation.shape, default_value, dtype=np.float64)
                
        # Only assign values for valid indices
        valid_indices = interstation[valid_mask].astype(int)
        param_maps[param][valid_mask] = param_values[valid_indices]
        param_maps[param][~valid_mask] = np.nan
                
        if useNNintepolation:
            # Set invalid points to the value of the nearest valid point
            param_maps[param][invalid_mask] = param_maps[param][valid_mask][nearest_valid_indices].flatten()

    return param_maps

def export_netcdf(path_result, interstation_ds, param_map, name):
    # Create new xarray Dataset based on the interstation dataset structure
    ds = xr.Dataset(
        {
            name: (('lat', 'lon'), param_map)
        },
        coords={
            'lat': interstation_ds.coords['lat'],
            'lon': interstation_ds.coords['lon']
        }
    )
    ds[name].attrs['standard_name'] = name
    ds[name].attrs['long_name'] = name
    ds.to_netcdf(os.path.join(path_result, f'{name}_EFASv6.nc'),
        encoding={
            name: {
                'zlib': True
            }}
        )

def main(interstation_path, output_path, params_path, calibrated_path, regionalisation_path, useNN=False):
    """Generate parameter maps from CSV calibration results using nearest-neighbor interpolation.

    Parameters
    ----------
    interstation_path : str
        Path to interstation_regions.nc NetCDF file.
    output_path : str
        Output folder for parameter NetCDF maps.
    params_path : str
        Path to calibration parameters ranges CSV file.
    calibrated_path : str
        Path to calibrated parameters CSV file.
    regionalisation_path : str
        Path to regionalisation CSV file.
    useNN : bool, optional
        Use nearest neighbor interpolation for invalid points (-1). Default False.
    """

    print("=================== START ===================")

    path_result = output_path

    if not os.path.exists(path_result):
        os.makedirs(path_result)

    print(">> Loading parameter ranges...")
    param_ranges = pd.read_csv(params_path, sep=",", index_col=0)

    print(">> Loading calibrated parameters...")
    calibrated_data = pd.read_csv(calibrated_path, sep=",", index_col=0)

    print(">> Loading regionalized parameters...")
    regionalized_data = pd.read_csv(regionalisation_path, sep=",", index_col=0)

    # TEMP fix for EFASv6: as the intertation map contains regionalized IDs that are subtracted by 9000000, we need to subtract 9000000 to the IDs in theregionalized data to match them with the interstation map
    #regionalized_data.index = regionalized_data.index - 9000000

    all_catchment_data = pd.concat([calibrated_data, regionalized_data])

    print(">> Reading interstation regions map...")
    interstation_ds = xr.open_dataset(interstation_path)
    interstation = interstation_ds['Band1'].values

    # Handle empty rows before processing
    handle_empty_rows(interstation, all_catchment_data)

    # Replace IDs with parameter values using np.take
    param_maps = create_param_mapping(interstation, param_ranges, all_catchment_data, useNNintepolation=useNN)

    print("Exporting NetCDF files...")
    for param in param_ranges.index:
        export_netcdf(path_result, interstation_ds, param_maps[param], param)

    print("==================== END ====================")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--interstation', '-i', required=True, help='Path to interstation_regions.nc')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--params', '-p', required=True, help='Path to calibration parameters ranges csv file')
    parser.add_argument('--calibrated', '-c', required=True, help='Path to calibrated parameters csv file')
    parser.add_argument('--regionalisation', '-r', required=True, help='Path to regionalisation csv file')
    parser.add_argument('--useNN', action='store_true', help='Use nearest neighbor interpolation for invalid points (-1) instead of setting them to NaN')
    args = parser.parse_args()

    main(args.interstation, args.output, args.params, args.calibrated, args.regionalisation, args.useNN)
