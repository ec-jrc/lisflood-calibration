import os
import datetime
import numpy as np
import pandas as pd

from lisflood.global_modules.add1 import loadmap, compressArray
from lisflood.global_modules.netcdf import uncompress_array, write_netcdf_header

from pcraster import boolean


def time_step_from_type(station_type):
    """
    Determines the time step based on the station/calibration type.

    Parameters
    ----------
    station_type : int, float, np.int64, np.float64, or str
        Type of the station which could be an integer, float, or string
        indicating the station/calibration type.

    Returns
    -------
    int
        The time step as an integer.

    Raises
    ------
    Exception
        If the station/calibration type is not supported.
        Support formats for the station/calibration type are 6.0, 24.0, 6, 24, "*_6h", and "*_24h".
    """

    if isinstance(station_type, float) or isinstance(station_type, np.float64):
        if (station_type == 6.0 or station_type == 24.0):
            dt = int(station_type)
        else:
            raise Exception('Calibration type {} not supported'.format(station_type))
    elif isinstance(station_type, int) or isinstance(station_type, np.int64):
        if (station_type == 6 or station_type == 24):
            dt = station_type
        else:
            raise Exception('Calibration type {} not supported'.format(station_type))
    else:
        if station_type.find("_6h") > -1:
            dt = 6
        elif station_type.find("_24h") > -1:
            dt = 24
        else:
            raise Exception('Calibration type {} not supported'.format(station_type))

    return dt


def observation_period_days(station_type, observed_streamflow):
    """
    Calculates the observation period in days.

    Parameters
    ----------
    station_type : int, float, np.int64, np.float64, or str
        Type of the station which could be an integer, float, or string indicating the calibration type.
    observed_streamflow : pd.Series
        Time series of observed streamflow.

    Returns
    -------
    float
        Observation period in days.
    """

    # Extract total number of steps
    mask = observed_streamflow.notna().values
    cum_steps = np.cumsum(mask)
    max_steps = cum_steps[-1]

    # compute the observation period in years
    dt = time_step_from_type(station_type)
    freq = 24./dt
    obs_period_days = max_steps/freq

    return obs_period_days


def observation_period_years(station_type, observed_streamflow):
    """
    Calculates the observation period in years.

    Parameters
    ----------
    station_type : int, float, np.int64, np.float64, or str
        Type of the station which could be an integer, float, or string indicating the calibration type.
    observed_streamflow : pd.Series
        Time series of observed streamflow.

    Returns
    -------
    float
        Observation period in years.
    """

    # Extract total number of steps
    mask = observed_streamflow.notna().values
    cum_steps = np.cumsum(mask)
    max_steps = cum_steps[-1]

    # compute the observation period in years
    dt = time_step_from_type(station_type)
    freq = 24./dt
    obs_period_years = max_steps/365.25/freq

    return obs_period_years


def compute_split_date(obs_period_years, dt, valid_start, observations_filtered, num_max_calib_years):
    """
    Computes the split date for the dataset, which splits the dataset in
    two parts for calibration and validation.

    Parameters
    ----------
    obs_period_years : float
        Observation period in years.
    dt : int
        Time step (in hours).
    valid_start : str
        Start date of the valid observation period.
    observations_filtered : pd.Series
        Filtered observations.

    Returns
    -------
    str
        The computed split date.
    """

    # if < num_max_calib_years (usually 20 years): take all
    if obs_period_years < num_max_calib_years:
        split_date = valid_start
    # if >=num_max_calib_years, only use last num_max_calib_years years
    else:  
        steps_MAXyears = int(num_max_calib_years*365.25*24/dt)
        split_date = observations_filtered.index[-steps_MAXyears]

    return split_date

def find_valid_period_reservoir(model_initialized, reservoir_events_df, dt, observations_filtered, valid_start, valid_end, min_years=4):
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
            # filter reservoir events data
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

def extract_station_data(cfg, model_initialized, obsid, station_data, check_obs=True):
    """
    Extracts and processes station data for calibration.

    Parameters
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    obsid : str
        Observation station ID.
    station_data : pd.Series
        Series containing station data.
    check_obs : bool, optional
        Flag to enable checking if the observation period meets the minimum required period (default is True).

    Raises
    ------
    Exception
        If the observation period is shorter than the required minimum calibration days.
    """

    # A calibration requires a spinup
    # first valid observation point will be at forcing start + spinup
    start_date = (cfg.forcing_start + datetime.timedelta(days=int(float(station_data['Spinup_days'])))).strftime('%d/%m/%Y %H:%M')
    end_date = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

    # Retrieve observed streamflow and extract observation period
    observations = pd.read_csv(cfg.observed_discharges, sep=",", index_col=0)
    observed_streamflow = observations[str(obsid)]
    observed_streamflow = observed_streamflow[start_date:end_date]
    obs_period_days = observation_period_days(station_data['CAL_TYPE'], observed_streamflow)
    obs_period_years = obs_period_days/365.25

    if check_obs:
        if obs_period_days < float(station_data['Min_calib_days']):
            raise Exception('Station {} only contains {} days of data! {} required'.format(obsid, obs_period_days, station_data['Min_calib_days']))

    # Extract valid calibration period
    observations_filtered = observed_streamflow[observed_streamflow.notna()]

    dt = time_step_from_type(station_data['CAL_TYPE'])

    valid_start = observations_filtered.index[0]
    valid_end = observations_filtered.index[-1]

    if model_initialized is not None:
        if os.path.exists(cfg.reservoir_events):
            reservoir_events_df = pd.read_csv(cfg.reservoir_events)
            valid_start, valid_end = find_valid_period_reservoir(model_initialized, reservoir_events_df, dt, observations_filtered, valid_start, valid_end, min_years=4)
        else:
            print("WARNING: reservoir_events csv file not found. Observations will not be filtered by reservoir events")


    valid_observations = observed_streamflow[valid_start:valid_end]

    # Compute split date
    split_date = compute_split_date(obs_period_years, dt, valid_start, observations_filtered, cfg.num_max_calib_years)

    # Create output directory
    subcatchment_path = os.path.join(cfg.subcatchment_path, str(obsid))
    out_dir = os.path.join(subcatchment_path, 'station')
    os.makedirs(out_dir, exist_ok=True)

    # Export observation at station
    obs_df = pd.DataFrame(data=valid_observations, index=valid_observations.index)
    obs_df.columns = [str(obsid)]
    obs_df.index.name = 'Timestamp'
    print('Station observations:')
    print(obs_df)
    obs_df.to_csv(os.path.join(out_dir, 'observations.csv'))

    # Export station data at station
    station_data.loc['Obs_start'] = valid_start
    station_data.loc['Obs_end'] = valid_end
    station_data.loc['Split_date'] = split_date
    station_data.loc['N_data'] = len(observations_filtered)
    station_df = pd.DataFrame(data=station_data)
    print('Station data:')
    print(station_df)
    station_df.to_csv(os.path.join(out_dir, 'station_data.csv'))

    print('Summary for catchment {}:'.format(obsid))
    print('First observation date: {}'.format(station_data['Obs_start']))
    print('Last observation date: {}'.format(station_data['Obs_end']))
    print('Split date: {}'.format(station_data['Split_date']))
    print('Number of non-missing data: {}'.format(station_data['N_data']))
