#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
import pandas as pd

from liscal import config, stations


class ConfigFilter(config.Config):

    def __init__(self, settings_file):
        super().__init__(settings_file)
        
        # Date parameters
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")
        self.timestep = int(self.parser.get('Main', 'timestep'))  # in minutes
        if self.timestep != 360 and self.timestep != 1440:
            raise Exception('Calibration timestep {} not supported'.format(self.timestep))
        
        # observations
        self.observed_discharges = self.parser.get('Stations', 'observed_discharges')
        self.stations_data = self.parser.get('Stations', 'stations_data')


if __name__ == '__main__':

    print("=================== START ===================")
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration pre-processing settings file')
    parser.add_argument('stations_csv', help='Input station csv file')
    parser.add_argument('stations_type', help='Station type (EC_calib value)')
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = ConfigFilter(settings_file)

    # Read stations data and selection stations of interest
    print('Reading input stations data {}'.format(args.stations_csv))
    stations_meta = pd.read_csv(args.stations_csv, sep=",", index_col='ObsID')
    print(stations_meta)
    stations_meta = stations_meta[stations_meta['EC_calib'] >= int(args.stations_type)]
    print('Found {} calibration stations to check'.format(len(stations_meta)))

    observed_data = pd.read_csv(cfg.observed_discharges, sep=",", index_col=0)

    valid_stations = []
    unvalid_stations = []

    for index, row in stations_meta.iterrows():

        # A calibration requires a spinup
        # first valid observation point will be at forcing start + spinup
        start_date = (cfg.forcing_start + timedelta(days=int(row['Spinup_days']))).strftime('%d/%m/%Y %H:%M')
        end_date = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')

        # Retrieve observed streamflow and extract observation period
        observed_streamflow = observed_data[str(index)]
        observed_streamflow = observed_streamflow[start_date:end_date]
        obs_period_days = stations.observation_period_days(row['CAL_TYPE'], observed_streamflow)

        if obs_period_days >= float(row['Min_calib_days']):
            valid_stations.append(index)
        else:
            tmp = observed_streamflow[observed_streamflow.notna()]
            unvalid_stations.append(index)
            print('Station {} only contains {:.2f} days of data ({} required), discarding... {} {}'.format(index, obs_period_days, row['Min_calib_days'], tmp.index[0], tmp.index[-1]))

    print('Found {} valid calibration stations'.format(len(valid_stations)))
    if len(valid_stations) == 0:
        raise Exception("ERROR: Not a single station has enough data for calibration.")

    # Save filtered
    valid_stations = stations_meta.loc[valid_stations]
    unvalid_stations = stations_meta.loc[unvalid_stations]
    print(valid_stations)
    valid_stations.to_csv(cfg.stations_data)
    unvalid_stations.to_csv(cfg.stations_data[:-4]+'_invalid.csv')
    print("==================== END ====================")
