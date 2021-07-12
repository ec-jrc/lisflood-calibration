import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from liscal import products, hydro_stats, config


def test_speedometers():

    speedo = products.SpeedometerPlot(config.PlotParameters())

    stats = {}
    stats['kge'] = 0.55
    stats['corr'] = 0.56
    stats['bias'] = 1.15
    stats['spread'] = 0.55

    speedo.plot('speedo', stats)
    # os.remove('speedo.svg')


def test_monthly_box_plot():

    index = pd.date_range(datetime(1985,7,1), datetime(2015,7,1))
    sim = (5+np.random.randn(len(index)))*30
    obs = (5+np.random.randn(len(index)))*30

    # compute monthly discharge data
    sim_monthly, obs_monthly = hydro_stats.split_monthly(index, sim, obs, spinup=0)

    box = products.MonthlyBoxPlot(config.PlotParameters())
    box.plot('boxy', sim_monthly, obs_monthly)
    # os.remove('boxy.svg')


# @pytest.mark.parametrize('catch', [2823, 380])
def test_time_series_plot():

    thresholds = {
        'rl1.5': 15,
        'rl2': 30,
        'rl5': 50,
        'rl20': 100,
    }

    index = pd.to_datetime(pd.date_range(datetime(2015,1,25), datetime(2015,10,10)))
    sim = (5+np.random.randn(len(index)))*30
    obs = (5+np.random.randn(len(index)))*30

    ts = products.TimeSeriesPlot(config.PlotParameters())
    ts.plot('timmy', index, sim, obs, thresholds)
    # os.remove('timmy.svg')