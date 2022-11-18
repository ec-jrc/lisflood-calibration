import os
import numpy as np
import xarray as xr
import pandas as pd
import calendar
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import transforms
from matplotlib import ticker

from plotflood import evaluation
from liscal import hydro_stats, thresholds


def create_products(cfg, subcatch, obj, observations_file):

    # create output directory
    os.makedirs(cfg.summary_path, exist_ok=True)

    # set validation period
    validation_start = datetime.strptime(subcatch.data['Validation_start'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
    validation_end = datetime.strptime(subcatch.data['Validation_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

    # read observation
    observed_streamflow = obj.read_observed_streamflow(observations_file, validation_start, validation_end)

    # read long term run
    simulated_streamflow = obj.read_simulated_streamflow_best()

    # compute statistics (KGE, NSE, etc.)
    Q, stats = obj.compute_statistics(validation_start, validation_end, observed_streamflow, simulated_streamflow)
    print(stats)

    # compute monthly discharge data
    sim_monthly, obs_monthly = hydro_stats.split_monthly(Q.index, Q['Sim'].values, Q['Obs'].values)

    # get return periods at station coordinates
    return_periods = thresholds.compute_thresholds(simulated_streamflow)
    # thresholds = xr.open_dataset(cfg.return_periods).sel(x=subcatch.data['LisfloodX'], y=subcatch.data['LisfloodY'])
    print(return_periods)

    # create asci output of stats
    with open(os.path.join(subcatch.path_out, 'stats.txt'), 'w') as f:
        f.write(str(stats))
        f.close()

    # create speedometer plots
    speedo = evaluation.SpeedometerPlot(cfg.plot_params)
    speedo.plot(os.path.join(subcatch.path_out, 'speedo'), stats)
    os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'speedo')))

    # create box plot
    box = evaluation.MonthlyBoxPlot(cfg.plot_params)
    box.plot(os.path.join(subcatch.path_out, 'boxy'), sim_monthly, obs_monthly)
    os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'boxy')))

    # create time series plot
    ts = evaluation.TimeSeriesPlot(cfg.plot_params)
    ts.plot(os.path.join(subcatch.path_out, 'timmy'), Q.index, Q['Sim'].values, Q['Obs'].values, return_periods)
    os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'timmy')))

    # compute contingency table and export
    # contingency_values = binary_scores.contingency_table(thresholds, Q)
    # contingency_df = pd.DataFrame(data=contingency_values, index=subcatch.obsid)
    # print(contingency_df)
    # contingency_df.to_csv(path.join(cfg.summary_path, 'contingency_table_{}.csv'.format(subcatch.obsid)))
