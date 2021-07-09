import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from liscal import products, hydro_stats


class PlotParametersSpeedo():
    
    title_size_big = 36
    title_size_small = 18
    label_size = 30
    axes_size = 24
    legend_size_small = 12

    file_format = 'svg'

    text = {
        'figure': {'autolayout': True},
        'font': {
            'family':'sans-serif',
            'sans-serif':['Arial']
        },
        'text': {'usetex': True},
    }


def test_speedometers():

    speedo = products.SpeedometerPlot(PlotParametersSpeedo())

    stats = {}
    stats['kge'] = 0.55
    stats['corr'] = 0.56
    stats['bias'] = 1.15
    stats['spread'] = 0.55

    speedo.plot('speedo', stats)
    # os.remove('speedo.svg')


class PlotParametersBox():
    
    title_size_big = 36
    title_size_small = 18
    label_size = 30
    axes_size = 24
    legend_size_small = 16

    file_format = 'svg'

    text = {
        'figure': {'autolayout': True},
        'font': {
            'size': 14,
            'family':'sans-serif',
            'sans-serif':['Arial'],
            'weight': 'bold'
        },
        'text': {'usetex': True},
        'axes': {'labelweight': 'bold'},
    }


def test_monthly_box_plot():

    index = pd.date_range(datetime(1985,7,1), datetime(2015,7,1))
    # index = np.arange(datetime(1985,7,1), datetime(2015,7,1), timedelta(days=1)).astype(datetime)

    sim = (5+np.random.randn(len(index)))*30
    obs = (5+np.random.randn(len(index)))*30

    # compute monthly discharge data
    sim_monthly, obs_monthly = hydro_stats.split_monthly(index, sim, obs, spinup=0)

    box = products.MonthlyBoxPlot(PlotParametersBox())
    box.plot('boxy', sim_monthly, obs_monthly)
    # os.remove('boxy.svg')
