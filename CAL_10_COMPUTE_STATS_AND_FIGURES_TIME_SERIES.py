# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import datetime
import time
import glob
import string
from ConfigParser import SafeConfigParser

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import rcParams
from matplotlib import rc
import pandas

import HydroStats


rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=False)
rcParams.update({'figure.autolayout': True})

t = time.time()

########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])

(drive, path) = os.path.splitdrive(iniFile)
(path, fil) = os.path.split(path)
print ">> Reading settings file (" + fil + ")..."

file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

parser = SafeConfigParser()
parser.read(iniFile)

ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT', 'ForcingStart'), "%d/%m/%Y %H:%M")  # Start of forcing
ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT', 'ForcingEnd'), "%d/%m/%Y %H:%M")  # Start of forcing

WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))

CatchmentDataPath = parser.get('Path', 'CatchmentDataPath')
SubCatchmentPath = parser.get('Path', 'SubCatchmentPath')

path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'), "maps")
path_result = parser.get('Path', 'Result')

Qtss_csv = parser.get('CSV', 'Qtss')
Qgis_csv = parser.get('CSV', 'Qgis')

########################################################################
#   Loop through catchments and perform calibration
########################################################################

print ">> Reading Qgis2.csv file..."

stationdata = pandas.read_csv(os.path.join(path_result, "Qgis2.csv"), sep=",", index_col=0)
stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'], ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess, sep=",", header=None)

stationdata_sorted['KGE_cal'] = np.nan
stationdata_sorted['KGE_val'] = np.nan
stationdata_sorted['NSE_cal'] = np.nan
stationdata_sorted['NSE_val'] = np.nan

Series = CatchmentsToProcess.ix[:, 0]
for index, row in stationdata_sorted.iterrows():
    if len(Series[Series == str(row["ID"])]) == 0:  # Only process catchments whose ID is in the CatchmentsToProcess.txt file
        continue
    path_subcatch = os.path.join(SubCatchmentPath, row['ID'])
    Val_Start = datetime.datetime.strptime(row['Val_Start'], "%d/%m/%Y %H:%M")
    print row['ID'] + " " + str(Val_Start)  # For debugging

    # Make figures directory
    figures_path = os.path.join(path_subcatch, "FIGURES")
    if not os.path.exists(figures_path):
        os.mkdir(os.path.join(path_subcatch, "FIGURES"))

    # Delete contents of figures directory
    for filename in glob.glob(os.path.join(path_subcatch, "FIGURES", '*.*')):
        os.remove(filename)

    # Compute the time steps at which the calibration should start and end
    if row['Val_Start'][:10] != "Streamflow":  # Check if Q record is long enough for validation
        Val_Start = datetime.datetime.strptime(row['Val_Start'], "%d/%m/%Y %H:%M")
        Val_End = datetime.datetime.strptime(row['Val_End'], "%d/%m/%Y %H:%M")
        Val_Start_Step = (Val_Start - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
        Val_End_Step = (Val_End - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
    else:
        Val_Start = []
        Val_End = []
        Val_Start_Step = []
        Val_End_Step = []

    Cal_Start = datetime.datetime.strptime(row['Cal_Start'], "%d/%m/%Y %H:%M")
    Cal_End = datetime.datetime.strptime(row['Cal_End'], "%d/%m/%Y %H:%M")
    Cal_Start_Step = (Cal_Start - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
    Cal_End_Step = (Cal_End - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
    Forcing_End_Step = (ForcingEnd - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!

    # Load observed streamflow
    streamflow_data = pandas.read_csv(Qtss_csv, sep=",", index_col=0, parse_dates=True, dayfirst=True)
    Qobs = streamflow_data[row['ID']]
    Qobs[Qobs < 0] = np.NaN

    # Load streamflow of best run
    if os.path.isfile(os.path.join(path_subcatch, "streamflow_simulated_best.csv")):
        print "Making figures for catchment " + row['ID'] + ", size " + str(row['CatchmentArea']) + " pixels..."
        Qsim = pandas.read_csv(os.path.join(path_subcatch, "streamflow_simulated_best.csv"), sep=",", parse_dates=True, index_col=0, header=None)
    else:
        print "streamflow_simulated_best.csv missing for catchment" + row['ID'] + "..."
        continue

    Qsim = Qsim.ix[:, 1]  # from dataframe to series

    # Make dataframe with aligned Qsim and Qobs columns
    Q = pandas.concat([Qsim, Qobs], axis=1)  # .reset_index()

    ########################################################################
    #   FIGURES  FIGURES  FIGURES   FIGURES  FIGURES  FIGURES  FIGURES     #
    ########################################################################

    fig = plt.figure()
    gs = plt.GridSpec(13, 6)
    plots_grids = {
        'title': (0, slice(None, None)),  # first row - Title
        'calibration': (slice(1, 5), slice(None, None)),  # second row - plot (a) calibration period
        'validation': (slice(5, 9), slice(None, None)),  # third row - plot (b) validation period
        'climatology_cal': (slice(9, 13), slice(0, 3)),  # fourth row - plot (c) monthly discharge (calibration)
        'climatology_val': (slice(9, 13), slice(3, 6)),  # fourth row - plot (d) monthly discharge (validation)
    }

    # ticks config
    months = np.array([9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # water year

    # TEXT OF CALIBRATION RESULTS
    ax0 = plt.subplot(gs[plots_grids['title']])  # first row
    ax0.set_axis_off()
    texts = '{}: {} at {} ({})'.format(row['ID'], row['RiverName'], row['Stationnam'], row['CountryNam'])
    texts_filtered = filter(lambda x: x in string.printable, texts)
    ax0.text(0.5, 0.0, texts_filtered, verticalalignment='top', horizontalalignment='center', transform=ax0.transAxes, fontsize=15)

    # FIGURE OF CALIBRATION PERIOD TIME SERIES
    Dates_Cal = Q.loc[Cal_Start:Cal_End].index
    Q_sim_Cal = Q.loc[Cal_Start:Cal_End].ix[:, 0].values
    Q_obs_Cal = Q.loc[Cal_Start:Cal_End].ix[:, 1].values
    ax1 = plt.subplot(gs[plots_grids['calibration']])  # second row
    max_y_cal = max(Q_sim_Cal.max(), Q_obs_Cal.max()) * 1.3
    ax1.set_ybound(upper=max_y_cal)
    ax1.axis('auto')
    ax1.set_adjustable('datalim')
    # format the ticks
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())

    ax1.plot(Dates_Cal.to_pydatetime(), Q_sim_Cal, 'r', Dates_Cal.to_pydatetime(), Q_obs_Cal, 'b')
    ax1.set_title('(a) Daily discharge - calibration period')
    ax1.set_ylabel(r'$Discharge [m^3 / s]$')
    statsum = r" " \
              + "KGE$=" + "{0:.2f}".format(HydroStats.KGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
              + "$, NSE$=" + "{0:.2f}".format(HydroStats.NS(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
              + "$, $R=" + "{0:.2f}".format(HydroStats.correlation(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
              + "$, $B=" + "{0:.2f}".format(HydroStats.pc_bias2(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
              + "$%"
    ax1.text(0.015, 0.93, statsum, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes)
    leg = ax1.legend(['Simulated', 'Observed'], fancybox=True, framealpha=0.8, prop={'size': 10}, labelspacing=0.1, loc=1)
    leg.get_frame().set_edgecolor('white')

    stationdata_sorted.loc[index, 'KGE_cal'] = HydroStats.KGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)
    stationdata_sorted.loc[index, 'NSE_cal'] = HydroStats.NS(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)
    # FIGURE OF VALIDATION PERIOD TIME SERIES
    if row['Val_Start'][:10] != "Streamflow":  # Check if Q record is long enough for validation
        Dates_Val = Q.loc[Val_Start:Val_End].index
        Q_sim_Val = Q.loc[Val_Start:Val_End].ix[:, 0].values
        Q_obs_Val = Q.loc[Val_Start:Val_End].ix[:, 1].values
        if len(Q_obs_Val[~np.isnan(Q_obs_Val)]) > (365 * 2):
            ax2 = plt.subplot(gs[plots_grids['validation']])
            ax2.axis('auto')
            ax2.set_adjustable('datalim')
            # format the ticks
            ax2.xaxis.set_major_locator(mdates.YearLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax2.xaxis.set_minor_locator(mdates.MonthLocator())
            ax2.set_title('(b) Daily discharge - validation period')
            ax2.set_ylabel(r'$Discharge [m^3 / s]$')
            max_y_val = max(Q_sim_Val.max(), Q_obs_Val.max()) * 1.3
            ax2.set_ybound(upper=max_y_val)

            statsum = r" " \
                      + "KGE$=" + "{0:.2f}".format(HydroStats.KGE(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)) \
                      + "$, NSE$=" + "{0:.2f}".format(HydroStats.NS(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)) \
                      + "$, $R=" + "{0:.2f}".format(HydroStats.correlation(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)) \
                      + "$, $B=" + "{0:.2f}".format(HydroStats.pc_bias2(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)) \
                      + "$%"
            ax2.text(0.015, 0.93, statsum, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes)
            ax2.plot(Dates_Val.to_pydatetime(), Q_sim_Val, 'r', Dates_Val.to_pydatetime(), Q_obs_Val, 'b')
            stationdata_sorted.loc[index, 'KGE_val'] = HydroStats.KGE(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)
            stationdata_sorted.loc[index, 'NSE_val'] = HydroStats.NS(s=Q_sim_Val, o=Q_obs_Val, warmup=WarmupDays)

    # FIGURE OF MONTHLY CLIMATOLOGY FOR CALIBRATION PERIOD
    Q_obs_clim_Cal = np.zeros(shape=(12, 1)) * np.NaN
    Q_sim_clim_Cal = np.zeros(shape=(12, 1)) * np.NaN
    Q_obs_clim_Cal_stddev = np.zeros(shape=(12, 1)) * np.NaN
    Q_sim_clim_Cal_stddev = np.zeros(shape=(12, 1)) * np.NaN
    for month in range(1, 13):
        mask = ~np.isnan(Q_obs_Cal) & ~np.isnan(Q_sim_Cal)
        Q_obs_clim_Cal[month - 1] = np.mean(Q_obs_Cal[(Dates_Cal.month == month) & mask])
        Q_sim_clim_Cal[month - 1] = np.mean(Q_sim_Cal[(Dates_Cal.month == month) & mask])
        Q_obs_clim_Cal_stddev[month - 1] = np.std(Q_obs_Cal[(Dates_Cal.month == month) & mask])
        Q_sim_clim_Cal_stddev[month - 1] = np.std(Q_sim_Cal[(Dates_Cal.month == month) & mask])

    ax3 = plt.subplot(gs[plots_grids['climatology_cal']])
    fill_sim_upper = Q_sim_clim_Cal[months - 1] + 0.5 * Q_sim_clim_Cal_stddev[months - 1]
    fill_sim_lower = Q_sim_clim_Cal[months - 1] - 0.5 * Q_sim_clim_Cal_stddev[months - 1]
    fill_obs_upper = Q_obs_clim_Cal[months - 1] + 0.5 * Q_obs_clim_Cal_stddev[months - 1]
    fill_obs_lower = Q_obs_clim_Cal[months - 1] - 0.5 * Q_obs_clim_Cal_stddev[months - 1]
    fill_sim_upper[fill_sim_upper < 0] = 0
    fill_sim_lower[fill_sim_lower < 0] = 0
    fill_obs_upper[fill_obs_upper < 0] = 0
    fill_obs_lower[fill_obs_lower < 0] = 0
    ax3.fill_between(np.arange(0, 14),
                     fill_sim_upper.reshape(-1),
                     fill_sim_lower.reshape(-1),
                     facecolor='red', alpha=0.1, edgecolor='none')
    ax3.fill_between(np.arange(0, 14),
                     fill_obs_upper.reshape(-1),
                     fill_obs_lower.reshape(-1),
                     facecolor='blue', alpha=0.1, edgecolor='none')
    ax3.plot(range(0, 14), Q_sim_clim_Cal[months - 1], 'r', range(0, 14), Q_obs_clim_Cal[months - 1], 'b')
    ax3.set_title('(c) Monthly discharge - calibration')
    ax3.set_xticks(months)
    ax3.set_xlim([0.5, 12.5])
    ax3.set_ylabel(r'$Discharge [m^3 / s]$')
    ax3.set_xlabel(r'Month')
    leg.get_frame().set_edgecolor('white')

    # FIGURE OF MONTHLY CLIMATOLOGY FOR VALIDATION PERIOD
    if row['Val_Start'][:10] != "Streamflow":  # Check if Q record is long enough for validation
        if len(Q_obs_Val[~np.isnan(Q_obs_Val)]) > 365:
            Q_obs_clim_Val = np.zeros(shape=(12, 1)) * np.NaN
            Q_sim_clim_Val = np.zeros(shape=(12, 1)) * np.NaN
            Q_obs_clim_Val_stddev = np.zeros(shape=(12, 1)) * np.NaN
            Q_sim_clim_Val_stddev = np.zeros(shape=(12, 1)) * np.NaN
            for month in range(1, 13):
                mask = ~np.isnan(Q_obs_Val) & ~np.isnan(Q_sim_Val)
                Q_obs_clim_Val[month - 1] = np.mean(Q_obs_Val[(Dates_Val.month == month) & mask])
                Q_sim_clim_Val[month - 1] = np.mean(Q_sim_Val[(Dates_Val.month == month) & mask])
                Q_obs_clim_Val_stddev[month - 1] = np.std(Q_obs_Val[(Dates_Val.month == month) & mask])
                Q_sim_clim_Val_stddev[month - 1] = np.std(Q_sim_Val[(Dates_Val.month == month) & mask])

            ax4 = plt.subplot(gs[plots_grids['climatology_val']])
            fill_sim_upper = Q_sim_clim_Val[months - 1] + 0.5 * Q_sim_clim_Val_stddev[months - 1]
            fill_sim_lower = Q_sim_clim_Val[months - 1] - 0.5 * Q_sim_clim_Val_stddev[months - 1]
            fill_obs_upper = Q_obs_clim_Val[months - 1] + 0.5 * Q_obs_clim_Val_stddev[months - 1]
            fill_obs_lower = Q_obs_clim_Val[months - 1] - 0.5 * Q_obs_clim_Val_stddev[months - 1]
            fill_sim_upper[fill_sim_upper < 0] = 0
            fill_sim_lower[fill_sim_lower < 0] = 0
            fill_obs_upper[fill_obs_upper < 0] = 0
            fill_obs_lower[fill_obs_lower < 0] = 0
            ax4.fill_between(np.arange(0, 14),
                             fill_sim_upper.reshape(-1),
                             fill_sim_lower.reshape(-1),
                             facecolor='red', alpha=0.1, edgecolor='none')
            ax4.fill_between(np.arange(0, 14),
                             fill_obs_upper.reshape(-1),
                             fill_obs_lower.reshape(-1),
                             facecolor='blue', alpha=0.1, edgecolor='none')
            ax4.plot(range(0, 14), Q_sim_clim_Val[months - 1], 'r', range(0, 14), Q_obs_clim_Val[months - 1], 'b')
            ax4.set_title('(d) Monthly discharge - validation')
            ax4.set_xticks(months)
            ax4.set_xlim([0.5, 12.5])
            ax4.set_ylabel(r'$Discharge [m^3 / s]$')
            ax4.set_xlabel(r'Month')

    # FIGURES OF CALIBRATION EVOLUTION
    front_history = pandas.read_csv(os.path.join(path_subcatch, "front_history.csv"), sep=",", parse_dates=True, index_col=0)
    adjustprops = dict(left=0.1, bottom=0, right=1, top=0.5, wspace=-0.2, hspace=0.0)
    fig.subplots_adjust(**adjustprops)

    plt.draw()
    fig.set_size_inches(22 / 2.54, 30 / 2.54)
    fig.savefig(os.path.join(path_subcatch, "FIGURES", row['ID'] + '_summary.png'), dpi=300, format='PNG', bbox_inches='tight')
    plt.close("all")

stationdata_sorted.to_csv(os.path.join(path_result, "Qgis3.csv"), ',')
