import os
import datetime
import numpy as np
import pandas as pd


def time_step_from_type(station_type):
    """
    Determines the time step (6 or 24) based on the station/calibration type.

    Parameters
    ----------
    station_type : int, float, str, np.int64, np.float64
        Type of the station (e.g., 6, 24, 6.0, 24.0, "6", "24", "6.0", "24.0", "*_6h", "*_24h")

    Returns
    -------
    int
        Time step in hours (6 or 24)

    Raises
    ------
    Exception
        If the input type or format is unsupported.
    """
    # Handle 0D NumPy arrays (e.g., np.array(6.0))
    if isinstance(station_type, np.ndarray) and station_type.shape == ():
        station_type = station_type.item()  # Extract scalar

    # Normalize numeric types
    if isinstance(station_type, (int, float, np.integer, np.floating)):    
        station_type = int(station_type)
        if station_type in (6, 24):
            return station_type

    # Handle strings
    if isinstance(station_type, str):
        cleaned = station_type.strip()

        # Try to parse as float (e.g., "6.0")
        try:
            numeric_value = float(cleaned)
            if numeric_value in (6.0, 24.0):
                return int(numeric_value)
        except ValueError:
            pass

        # Check for suffix patterns
        if "_6h" in cleaned:
            return 6
        elif "_24h" in cleaned:
            return 24
        elif cleaned == "6":
            return 6
        elif cleaned == "24":
            return 24

    raise Exception(
        f"Calibration type {station_type} not supported. "
        "Supported formats: 6, 24, 6.0, 24.0, '6', '24', '6.0', '24.0', '*_6h', '*_24h'."
    )


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

def process_reservoir_periods(model_initialized, reservoir_events_df, dt, observations_filtered, valid_start, valid_end, Min_calib_days, isLongRun=False):
    """Moved to liscal.reservoirs. This is a compatibility shim."""
    from liscal.reservoirs import process_reservoir_periods as _prp
    return _prp(model_initialized, reservoir_events_df, dt, observations_filtered, valid_start, valid_end, Min_calib_days, isLongRun)

def create_netcdf_map(map_name, reservoirs, reservoir_events_df, model_initialized, period_start_dt, period_end_dt):
    """Moved to liscal.reservoirs. This is a compatibility shim."""
    from liscal.reservoirs import create_netcdf_map as _cnm
    return _cnm(map_name, reservoirs, reservoir_events_df, model_initialized, period_start_dt, period_end_dt)

def update_rsfil_netcdf_map(map_name, reservoir_events_df, settings, period_start_dt):
    """Moved to liscal.reservoirs. This is a compatibility shim."""
    from liscal.reservoirs import update_rsfil_netcdf_map as _urnm
    return _urnm(map_name, reservoir_events_df, settings, period_start_dt)

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

    # Create output directory
    subcatchment_path = os.path.join(cfg.subcatchment_path, str(obsid))
    out_dir = os.path.join(subcatchment_path, 'station')
    os.makedirs(out_dir, exist_ok=True)

    observations[str(obsid)].to_csv(os.path.join(out_dir, 'observations_original.csv'))
    
    # Convert the index to datetime
    observations.index = pd.to_datetime(observations.index, format='%d/%m/%Y %H:%M')

    if cfg.timestep==1440:
        full_date_range = pd.date_range(start=cfg.forcing_start.strftime('%d/%m/%Y %H:%M'), 
                                        end=cfg.forcing_end.strftime('%d/%m/%Y %H:%M'), 
                                        freq='D')
    else:
        assert(cfg.timestep==360)
        full_date_range = pd.date_range(start=cfg.forcing_start.strftime('%d/%m/%Y %H:%M'), 
                                        end=cfg.forcing_end.strftime('%d/%m/%Y %H:%M'), 
                                        freq='6H')

    # Reindex the DataFrame to include the full date range
    observations = observations.reindex(full_date_range)
    observations.index = observations.index.strftime('%d/%m/%Y %H:%M')

    observed_streamflow = observations[str(obsid)]
    observed_streamflow = observed_streamflow[start_date:end_date]
    observed_streamflow.to_csv(os.path.join(out_dir, 'observations_complete.csv'))

    obs_period_days = observation_period_days(station_data['CAL_TYPE'], observed_streamflow)
    obs_period_years = obs_period_days/365.25

    if check_obs:
        if obs_period_days < float(station_data['Min_calib_days']):
            raise Exception('Station {} only contains {} days of data! {} required'.format(obsid, obs_period_days, station_data['Min_calib_days']))

    # Extract valid calibration period
    observations_filtered = observed_streamflow[observed_streamflow.notna()]

    dt = time_step_from_type(station_data['CAL_TYPE'])  # here we use dt to calculate the observation period in process_reservoir_periods
    Min_calib_days = float(station_data['Min_calib_days']) # used also in process_reservoir_periods
    valid_start = observations_filtered.index[0]
    valid_end = observations_filtered.index[-1]

    if model_initialized is not None:
        if cfg.reservoir_events is not None:
            if os.path.exists(cfg.reservoir_events):
                reservoir_events_df = pd.read_csv(cfg.reservoir_events)
                valid_start, valid_end = process_reservoir_periods(model_initialized, reservoir_events_df, dt, observations_filtered, valid_start, valid_end, Min_calib_days, isLongRun=False)
                # update observed_streamflow and obs_period_years after reservoir_events to compute correct split date
                observed_streamflow = observed_streamflow[valid_start:valid_end]
                obs_period_days = observation_period_days(station_data['CAL_TYPE'], observed_streamflow)
                obs_period_years = obs_period_days/365.25
                if check_obs:
                    if obs_period_days < Min_calib_days:
                        raise Exception('ERROR after process_reservoir_periods: Station {} only contains {} days of data! {} required'.format(obsid, obs_period_days, Min_calib_days))
                observations_filtered = observed_streamflow[observed_streamflow.notna()]
            else:
                print("WARNING: reservoir_events csv file not found. Observations will not be filtered by reservoir events")


    valid_observations = observed_streamflow[valid_start:valid_end]

    # Compute split date
    split_date = compute_split_date(obs_period_years, dt, valid_start, observations_filtered, cfg.num_max_calib_years)

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
