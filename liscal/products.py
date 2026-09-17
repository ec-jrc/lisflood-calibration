import os
from datetime import datetime

from liscal import hydro_stats, thresholds, evaluation


def create_products(cfg, subcatch, obj, save=True):
    """
    Creates various hydrological evaluation products including statistical summaries, plots, and tables.

    Parameters
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    subcatch : Subcatchment
        Subcatchment information and data.
    obj : ObjectiveKGE
        ObjectiveKGE instance containing methods for reading and computing streamflow statistics.

    Notes
    -----
    This function performs several operations including:
    - Reading and computing statistics for simulated streamflow.
    - Generating monthly discharge data.
    - Computing return periods.
    - Creating ASCII output of statistics.
    - Producing various plots (speedometer, box, and time series plots).
    - Converting plots from SVG to PDF format.
    - (Commented out) Computing contingency tables.
    """

    obs_start = datetime.strptime(subcatch.data['Obs_start'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
    obs_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

    if save:
        # create output directory
        os.makedirs(cfg.summary_path, exist_ok=True)

    # Long term run has run_id X
    simulated_streamflow = obj.read_simulated_streamflow_best()

    # compute statistics (KGE, NSE, etc.)
    Q, stats = obj.compute_statistics(obs_start, obs_end, simulated_streamflow)

    # compute monthly discharge data
    sim_monthly, obs_monthly = hydro_stats.split_monthly(Q.index, Q['Sim'].values, Q['Obs'].values)

    # get return periods at station coordinates
    return_periods = thresholds.compute_thresholds(simulated_streamflow)
    # thresholds = xr.open_dataset(cfg.return_periods).sel(x=subcatch.data['LisfloodX'], y=subcatch.data['LisfloodY'])
    if save:
        print(return_periods)

        # create asci output of stats
        with open(os.path.join(subcatch.path_out, 'stats.txt'), 'w') as f:
            f.write(str(stats))
            f.close()

    # create speedometer plots
    speedo = evaluation.SpeedometerPlot(cfg.plot_params)
    speedo_fig = speedo.plot(os.path.join(subcatch.path_out, 'speedo')  if save else None, stats)
    if save:
        os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'speedo')))

    # create box plot
    box = evaluation.MonthlyBoxPlot(cfg.plot_params)
    box_fig = box.plot(os.path.join(subcatch.path_out, 'boxy') if save else None, sim_monthly, obs_monthly)
    if save:
        os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'boxy')))

    # create time series plot
    ts = evaluation.TimeSeriesPlot(cfg.plot_params)
    ts_fig = ts.plot(os.path.join(subcatch.path_out, 'timmy') if save else None, Q.index, Q['Sim'].values, Q['Obs'].values, return_periods)
    if save:
        os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'timmy')))

    # create Q-Q plot
    qq = evaluation.QQPlot(cfg.plot_params)
    qq_fig = qq.plot(os.path.join(subcatch.path_out, 'qqy') if save else None, Q.index, Q['Sim'].values, Q['Obs'].values)
    if save:
        os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'qqy')))
    
    # create best parameters table plot
    bestparmtrs = evaluation.BestParamPlot(cfg.plot_params)
    bestparmtrs_fig = bestparmtrs.plot(os.path.join(subcatch.path_out, 'bestparmtrs') if save else None, os.path.join(subcatch.path, 'pareto_front.csv'), subcatch.obsid)
    if save:
        os.system('convert {0}.png {0}.pdf'.format(os.path.join(subcatch.path_out, 'bestparmtrs')))

    # create (inter)catchment plot
    spatialplot = evaluation.SpatialPlot(cfg.plot_params)
    spatialplot_fig = spatialplot.plot(os.path.join(subcatch.path_out, 'spatial') if save else None, subcatch.path)
    if save:
        os.system('convert {0}.png {0}.pdf'.format(os.path.join(subcatch.path_out, 'spatial')))
    
    # compute contingency table and export
    # contingency_values = binary_scores.contingency_table(thresholds, Q)
    # contingency_df = pd.DataFrame(data=contingency_values, index=subcatch.obsid)
    # print(contingency_df)
    # contingency_df.to_csv(path.join(cfg.summary_path, 'contingency_table_{}.csv'.format(subcatch.obsid)))

    return speedo_fig, box_fig, ts_fig, qq_fig, bestparmtrs_fig, spatialplot_fig
