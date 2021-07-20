import os
import datetime
import numpy as np
import pandas as pd


def time_step_from_type(station_type):

    if str(station_type).find("_6h") > -1:
        dt = 6
    elif str(station_type).find("_24h") > -1:
        dt = 24
    else:
        raise Exception('Calibration type {} not supported'.format(station_type))
    return dt
    # return int(station_type)


def observation_period_days(station_type, observed_streamflow):

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

    # Extract total number of steps
    mask = observed_streamflow.notna().values
    cum_steps = np.cumsum(mask)
    max_steps = cum_steps[-1]

    # compute the observation period in years
    dt = time_step_from_type(station_type)
    freq = 24./dt
    obs_period_years = max_steps/365.25/freq

    return obs_period_years


def compute_split_date(obs_period_years, dt, valid_start, observations_filtered):
    
    # if < 8 years: take all
    if obs_period_years <= 8:
        split_date = valid_start
    # if > 8 and < 16 years, only use last 8 years
    elif obs_period_years > 8 and obs_period_years < 16:    
        steps_8years = 8*365.25*24/dt
        split_date = observations_filtered.index[-steps_8years]
    # if >= 16, split in two
    elif obs_period_years >= 16:
        split_date = observations_filtered.index[int(len(observations_filtered.index)/2)]

    return split_date


def extract_station_data(cfg, obsid, station_data):

    # A calibration requires a spinup
    # first valid observation point will be at forcing start + spinup
    start_date = (cfg.forcing_start + datetime.timedelta(days=int(station_data['Spinup_days']))).strftime('%d/%m/%Y %H:%M')
    end_date = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

    # Retrieve observed streamflow and extract observation period
    observations = pd.read_csv(cfg.observed_discharges, sep=",", index_col=0)
    observed_streamflow = observations[str(obsid)]
    observed_streamflow = observed_streamflow[start_date:end_date]
    obs_period_years = observation_period_years(station_data['CAL_TYPE'], observed_streamflow)
    # min 3.5 years data to calibrate
    if obs_period_years < cfg.min_obs_years:
        raise Exception('Station {} only contains {} years of data!'.format(obsid, obs_period_years))

    # Extract valid calibration period
    observations_filtered = observed_streamflow[observed_streamflow.notna()]

    valid_start = observations_filtered.index[0]
    valid_end = observations_filtered.index[-1]
    valid_observations = observed_streamflow[valid_start:valid_end]

    # Compute split date
    dt = time_step_from_type(station_data['CAL_TYPE'])
    split_date = compute_split_date(obs_period_years, dt, valid_start, observations_filtered)

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
    station_data['Obs_start'] = valid_start
    station_data['Obs_end'] = valid_end
    station_data['Split_date'] = split_date
    station_data['N_data'] = len(observations_filtered)
    station_df = pd.DataFrame(data=station_data)
    print('Station data:')
    print(station_df)
    station_df.to_csv(os.path.join(out_dir, 'station_data.csv'))

    print('Summary for catchment {}:'.format(obsid))
    print('First observation date: {}'.format(station_data['Obs_start']))
    print('Last observation date: {}'.format(station_data['Obs_end']))
    print('Split date: {}'.format(station_data['Split_date']))
    print('Number of non-missing data: {}'.format(station_data['N_data']))