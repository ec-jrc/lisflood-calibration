import os
import numpy as np
import pandas as pd

from lisflood.global_modules.add1 import loadmap, compressArray
from lisflood.global_modules.netcdf import uncompress_array, write_netcdf_header

from pcraster import boolean

from liscal import templates, calibration, config, subcatchment, objective, hydro_model


def initialise_model(cfg, station_data, obsid):

    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data=station_data)
    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
    lock_mgr = calibration.LockManager(cfg.num_cpus)
    obj = objective.ObjectiveKGE(cfg, subcatch, read_observations=False)
    model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)
    # load forcings and input maps in cache
    # required to find reservoir
    model.init_run()

    return model


def find_valid_period(model_initialized, dt, observations_filtered, valid_start, valid_end, min_years=4):
    # copy dates as string
    best_period_start, best_period_end = valid_start, valid_end

    # convert valid_start and valid_end in datetime objects
    valid_start = pd.to_datetime(valid_start, format="%d/%m/%Y %H:%M")
    valid_end = pd.to_datetime(valid_end, format="%d/%m/%Y %H:%M")

    min_steps = int(min_years*365.25*24/dt)

    # Check if reservoirs are simulated and load necessary maps
    if model_initialized.lissettings.options['simulateReservoirs']:
        reservoirs = loadmap('ReservoirSites')
        IsChannelPcr = boolean(loadmap('Channels', pcr=True))
        IsChannel = np.bool8(compressArray(IsChannelPcr))
        reservoirs[(reservoirs < 1) | (IsChannel == 0)] = 0

        # Get active reservoir sites
        ReservoirSitesCC = np.compress(reservoirs > 0, reservoirs)

        if ReservoirSitesCC.size > 0:
            # Load and filter reservoir events data
            reservoir_events_df = pd.read_csv('reservoir_events.csv')
            reservoir_events_df = reservoir_events_df[reservoir_events_df['FID'].isin(ReservoirSitesCC)]

            # Convert year columns to datetime
            reservoir_events_df['CONSTR_YEAR'] = pd.to_datetime(reservoir_events_df['CONSTR_YEAR'], format='%Y', errors='coerce')
            reservoir_events_df['DEMOL_YEAR'] = pd.to_datetime(reservoir_events_df['DEMOL_YEAR'], format='%Y', errors='coerce')

            # Gather potentially impacting events
            reservoir_events = sorted(
                [date for date in reservoir_events_df['CONSTR_YEAR'].dropna().tolist() + 
                 reservoir_events_df['DEMOL_YEAR'].dropna().tolist() 
                 if valid_start <= date <= valid_end]
            )

            # Determine the most recent valid observation period
            last_valid_end = valid_end
            for event in reversed([valid_start - pd.Timedelta(days=1)] + reservoir_events):
                period_observations = observations_filtered[event.strftime('%d/%m/%Y %H:%M'):last_valid_end.strftime('%d/%m/%Y %H:%M')]
                if len(period_observations) >= min_steps:
                    best_period_start, best_period_end = period_observations.index[0], period_observations.index[-1]
                    best_period_end_dt = pd.to_datetime(best_period_end, format="%d/%m/%Y %H:%M")
                    best_period_start_dt = pd.to_datetime(best_period_start, format="%d/%m/%Y %H:%M")
                    break
                last_valid_end = event - pd.Timedelta(days=1)

            # Identify active reservoirs for the selected period
            active_reservoirs = {
                row['FID'] for _, row in reservoir_events_df.iterrows()
                if (pd.isna(row['CONSTR_YEAR']) or row['CONSTR_YEAR'] <= best_period_end_dt) and
                   (pd.isna(row['DEMOL_YEAR']) or row['DEMOL_YEAR'] > best_period_start_dt)
            }

            # Save the reservoir filtered map for the selected period
            FilteredReservoirMap = np.full_like(reservoirs, -1, dtype=float)
            for res_id in active_reservoirs:
                # get index from reservoirs map
                FilteredReservoirMap[reservoirs==res_id]=res_id
            strFilteredReservoirMap=os.path.join(model_initialized.subcatch.path_station, 'FilteredReservoirMap.nc')
            # Save the new map LakeMultiplierMap to a NetCDF file                
            map_name = "FilteredReservoirMap"
            nf1 = write_netcdf_header(model_initialized.lissettings, map_name, strFilteredReservoirMap, None,
                                    map_name, map_name, "",
                                    None, None, None)

            map_np = uncompress_array(FilteredReservoirMap)
            nf1.variables[map_name][:, :] = map_np
            nf1.close()
            print("Generated new ReservoirSites content to:", strFilteredReservoirMap)

    return best_period_start, best_period_end
