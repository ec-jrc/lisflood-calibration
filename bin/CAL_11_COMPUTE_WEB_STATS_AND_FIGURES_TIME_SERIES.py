# -*- coding: utf-8 -*-
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm, ticker, rcParams, rc, image, get_backend, transforms
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage,AnchoredOffsetbox
from matplotlib.gridspec import GridSpec
import shapely
from shapely.geometry import LineString, Point
import HydroStats
import numpy as np
import datetime as dt
import calendar
import time
import pandas as pd
import math
import glob
import string
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15
import xarray as xr
import json
from collections import OrderedDict
import binaryForecastsSkill as bfs
import rpy2
from scipy.optimize import curve_fit
from decimal import Decimal
import scipy.stats as ss
import scipy.special as sp
from random import random
import statsmodels.api as smapi
from statsmodels.formula.api import ols
import statsmodels.graphics as smgraphics
import multiprocessing as mp
from multiprocessing import Pool
from pandarallel import pandarallel
import inspect
# import ray
# ray.init(num_cpus=6)

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
# rc('text', usetex=True)
rcParams.update()

# Climatological wingplots
pink = [i / 255. for i in (236, 118, 218)]
lightpurple = [i / 255. for i in (114, 119, 223)]
purple = [i / 255. for i in (91, 63, 159)]
darkestblue = [i / 255. for i in (23, 125, 245)]
darkerblue = [i / 255. for i in (91, 155, 213)]
lighterblue = [i / 255. for i in (131, 179, 223)]
lightestblue = [i / 255. for i in (189, 215, 238)]
grey = [i / 255. for i in (196, 198, 201)]

# Return periods
magenta = [i / 255. for i in (191, 81, 225)]
red = [i / 255. for i in (255, 29, 29)]
orange = [i / 255. for i in (250, 167, 63)]
green = [i / 255. for i in (112, 173, 71)]

# KGE speedometers
KGEpink = [i / 255. for i in (236, 118, 218)]
KGEpurple = [i / 255. for i in (186, 172, 214)]
KGEdarkestblue = [i / 255. for i in (57, 16, 139)]
KGEdarkerblue = [i / 255. for i in (58, 68, 214)]
KGElighterblue = [i / 255. for i in (86, 148, 254)]
KGElighestblue = [i / 255. for i in (160, 201, 254)]
KGEgrey = [i / 255. for i in (196, 198, 201)]

dpi = 300

axesFontSize = 24
legendFontSize = 27
tableFontSize = 30
labelFontSize = 30
contFontSize = 24
titleFontSize = 36

nfit = 1000


def fexp(number):
  (sign, digits, exponent) = Decimal(number).as_tuple()
  return len(digits) + exponent - 1



def fman(number):
  return Decimal(number).scaleb(-fexp(number)).normalize()


# How to use:
# (m, e) = nsplit(np.float(np.max(x)))
# x = np.linspace(np.min(x), np.round(m / 10) * 10 ** (e + 1), 1e3)
def nsplit(number):
  return np.float(fman(np.float(np.max(number)))), fexp(np.float(np.max(number)))



# Set of function for plotting speedometer gauges
# Original code from https://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/
def degree_range(n):
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points



# Set of function for plotting speedometer gauges
# Original code from https://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/
def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation



# Set of function for plotting speedometer gauges, modified to match Louise Arnal's EFAS4.0 designs
# Original code from https://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/
def gauge(ax, labels=['', '', '', '', ''], colors='jet_r', arrow=1, title='', score=None, fontSize=None):
  # some sanity checks first
  if arrow > 1:
    raise Exception("\n\nThe arrow position can't be larger than 100%.\n\n")
  # if colors is a string, we assume it's a matplotlib colormap and we discretize in N discrete colors
  N = len(labels)
  if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1, :].tolist()
  if isinstance(colors, list):
    if len(colors) == N:
      colors = colors[::-1]
    else:
      raise Exception("\n\nnumber of colors {} not equal to number of categories{}\n".format(len(colors), N))
  # begins the plotting
  ang_range, mid_points = degree_range(N)
  labels = labels[::-1]
  # plots the sectors and the arcs
  patches = []
  for ang, c in zip(ang_range, colors):
    # sectors
    patches.append(Wedge((0., 0.), .4, *ang, facecolor='white', lw=2))
    # arcs
    patches.append(Wedge((0., 0.), .4, *ang, width=0.20, edgecolor='white', facecolor=c, lw=2, alpha=1.0))
  [ax.add_patch(p) for p in patches]
  # set the labels
  for i, (mid, lab) in enumerate(zip(mid_points, labels)):
    if colors[i] == KGEdarkerblue or colors[i] == KGEdarkestblue:
      ax.text(0.3 * np.cos(np.radians(mid)), 0.3 * np.sin(np.radians(mid)), lab, horizontalalignment='center', verticalalignment='center', fontsize=fontSize, fontweight='bold', rotation=rot_text(mid), color='w')
    else:
      ax.text(0.3 * np.cos(np.radians(mid)), 0.3 * np.sin(np.radians(mid)), lab, horizontalalignment='center', verticalalignment='center', fontsize=fontSize, fontweight='bold', rotation=rot_text(mid), color='k')
  # set the title
  ax.text(0, 0.01, title, horizontalalignment='center', verticalalignment='center', fontsize=fontSize, fontweight='bold')
  # Calculate arrow angle
  pos = 180 - arrow * 180
  # normal arrow
  # ax.arrow(
  # 	0, 0, 0.3 * np.cos(np.radians(pos)), 0.3 * np.sin(np.radians(pos)),
  # 	width=0.01, head_width=0.01, head_length=0.2, facecolor='black', edgecolor='white',
  # 	head_starts_at_zero=True, length_includes_head=True
  # )
  # inverted arrow
  ax.arrow(
    0.499 * np.cos(np.radians(pos)), 0.499 * np.sin(np.radians(pos)), -0.3 * np.cos(np.radians(pos)), -0.3 * np.sin(np.radians(pos)),
    width=0.01, head_width=0.01, head_length=0.2, facecolor='black', edgecolor='white',
    head_starts_at_zero=True, length_includes_head=True
  )
  # Value label
  if score is not None:
    ax.text(0.45 * np.cos(np.radians(pos)), 0.45 * np.sin(np.radians(pos)), "{0:.2f}".format(score),
            horizontalalignment='center', verticalalignment='center', fontsize=2.5*fontSize, fontweight='bold',
            rotation=rot_text(pos), color='k')
  # removes frame and ticks, and makes axis equal and tight
  ax.set_frame_on(False)
  ax.axes.set_xticks([])
  ax.axes.set_yticks([])
  ax.axis('equal')



def maximizePlot():
  # # Maximize the plotting window
  # backend = get_backend()
  # mng = plt.get_current_fig_manager()
  # try:
  # 	if backend == "QT":
  # 		mng.window.showMaximized()
  # 	elif backend == "TkAgg":
  # 		mng.resize(*mng.window.maxsize())
  # 	elif backend == "WX":
  # 		mng.frame.Maximize(True)
  # except AttributeError:
  # 	mng.full_screen_toggle()
  plt.draw()
  plt.pause(1.0e-10)
  fig = plt.gcf()
  fig.set_size_inches(16.5, 11.7) # A3 size
  adjustprops = dict(left=0.1, bottom=0, right=1, top=1, wspace=-0.2, hspace=0.0)
  fig.subplots_adjust(**adjustprops)



def place_image(im, loc=3, ax=None, zoom=1, **kw):
  if ax==None: ax=plt.gca()
  imagebox = OffsetImage(im, zoom=zoom)
  ab = AnchoredOffsetbox(loc=loc, child=imagebox, frameon=False, **kw)
  ax.add_artist(ab)



def genLegend():
  prettydata = [
    0,
    1, 1, 1, 1, 1,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7,
    5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7,
    8, 8 ,8, 8, 8,
    9, 9, 9, 9,
    10, 10,
    14,
    16
  ]
  # add it to the figure
  fig = plt.gcf()
  fontsize = fig.get_axes()[0].get_xticklabels()[0].get_fontsize() / 2.0 * 0.8
  # temprorary axes to get boxplot values
  axd = fig.add_subplot(566)
  boxplot = axd.boxplot([prettydata, prettydata], notch=True, bootstrap=10000)
  plt.delaxes(axd)
  # new axes to plot legend
  axl = fig.add_subplot(555)
  # boxplot
  legendbox = axl.boxplot(prettydata, notch=True, sym='.', bootstrap=100, showmeans=False, meanline=True, patch_artist=True, widths=0.5)
  applyBoxplotTheme(legendbox)
  # corresponding wingplot
  (p1, p5, p25, median, p75, p95, p99) = wingplot(boxplot, axl, filler=np.nan)
  axl.set_xlim(-0.5, 8)
  axl.set_ylim(-3, 20)
  plt.axis('off')
  # annotations
  axl.text(x=3.5, y=(p99[0]+p95[0])/2, s='outliers', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
  axl.text(x=3.5, y=p95[0], s='Q3 + 1.5 IQR', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
  axl.text(x=3.5, y=p75[0], s='Q3: 75th perc.', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
  axl.text(x=3.5, y=median[0], s='median', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
  axl.text(x=3.5, y=p25[0], s='Q1: 25th perc.', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
  axl.text(x=3.5, y=p5[0], s='Q1 - 1.5 IQR', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
  axl.text(x=1, y=-1.5, s='Qsim', color='black', verticalalignment='center', horizontalalignment='center', fontweight='bold', family='fantasy', fontsize=fontsize+2, weight='bold')
  axl.text(x=2.5, y=-1.5, s='Qobs', color='black', verticalalignment='center', horizontalalignment='center', fontweight='bold', family='fantasy', fontsize=fontsize+2, weight='bold')
  # Pretty box around it
  bb = mtransforms.Bbox([[0.5, -2.5], [6.5, 17]])
  p_bbox = mpatches.FancyBboxPatch((bb.xmin, bb.ymin), abs(bb.width), abs(bb.height), boxstyle="round", edgecolor="k", facecolor='white', zorder=0)
  axl.add_patch(p_bbox)



def wingplot(boxplot, ax, filler=None):
  n = len(boxplot['medians'])
  # Calculate median
  median = []
  for i in boxplot['medians']:
    median += [i.get_ydata()[0]]
  # Wings of 3 stdevs (+/- Q3+1.5*IQR)
  p5 = []
  p95 = []
  for ii, i in enumerate(boxplot['whiskers']):
    ydata = i.get_ydata()
    if len(ydata) == 0:
      p5 += [np.nan]
      p95 += [np.nan]
    else:
      if ydata[0] <= median[int(ii/2)]:
        p5 += [ydata[1]]
      elif ydata[0] >= median[int(ii/2)]:
        p95 += [ydata[1]]
      else:
        print("Invalid value for data: " + str(ydata[0]))
  # Wings of 1 stdev (+/- Q1 and Q3)
  p25 = []
  p75 = []
  for i in boxplot['boxes']:
    ydata = i.get_ydata()
    if len(ydata) == 0:
      p25 += [np.nan]
      p75 += [np.nan]
    else:
      p25 += [min(ydata)]
      p75 += [max(ydata)]
  # Wings covering outliers
  p1 = []
  p99 = []
  for ii, i in enumerate(boxplot['fliers']):
    p1 += [p75[ii]]
    ydata = i.get_ydata()
    if len(ydata) == 0:
      p99 += [1.01*p1[ii]]
    else:
      p99 += [max(ydata)]
  # Layer it all on the plot in the right order
  if filler is None:
    try:
      ax.fill_between(np.arange(1,n+3), [p1[-3]] + p1 + [p1[2]], [p99[-3]] + p99 + [p99[2]], facecolor=lightestblue, alpha=1.0, edgecolor=None)
    except ValueError:
      print("OUCH1")
    try:
      ax.fill_between(np.arange(1,n+3), [p5[-3]] + p5 + [p5[2]], [p95[-3]] + p95 + [p95[2]], facecolor=lighterblue, alpha=1.0, edgecolor=None)
    except ValueError:
      print("OUCH2")
    try:
      ax.fill_between(np.arange(1,n+3), [p25[-3]] + p25 + [p25[2]], [p75[-3]] + p75 + [p75[2]], facecolor=darkerblue, alpha=1.0, edgecolor=None)
    except ValueError:
      print("OUCH3")
    # median line
    try:
      plt.plot(np.arange(1, n + 3), [median[-3]] + median + [median[2]], color=darkestblue, linewidth=3)
    except ValueError:
      print("OUCH4")
  else:
    ax.fill_between(np.arange(1, n + 3), [filler] + p1 + [filler], [filler] + p99 + [filler], facecolor=lightestblue, alpha=1.0, edgecolor=None)
    ax.fill_between(np.arange(1, n + 3), [filler] + p5 + [filler], [filler] + p95 + [filler], facecolor=lighterblue, alpha=1.0, edgecolor=None)
    ax.fill_between(np.arange(1, n + 3), [filler] + p25 + [filler], [filler] + p75 + [filler], facecolor=darkerblue, alpha=1.0, edgecolor=None)
    plt.plot(np.arange(1, n + 3), [np.nan] + median + [np.nan], color=darkestblue, linewidth=3)
  return (p1, p5, p25, median, p75, p95, p99)



def applyBoxplotTheme(boxplot):
  for i in boxplot['boxes']:
    i.set_color(purple)
    i.set_facecolor(lightpurple)
    i.set_edgecolor(purple)
    i.set_linewidth(1.5)
  for i in boxplot['whiskers'] + boxplot['caps'] + boxplot['fliers']:
    i.set_color(purple)
    i.set_fillstyle('full')
    i.set_markerfacecolor(lightpurple)
    i.set_markeredgecolor(purple)
    i.set_linewidth(2)
  for i in boxplot['medians']:
    i.set_color(purple)
    i.set_fillstyle('full')
    i.set_linewidth(2)
  for i in boxplot['means']:
    i.set_color(pink)
    i.set_fillstyle('full')
    i.set_linewidth(1.5)



def createContingencyTable(threshold, Q):
  rpMask = Q >= threshold
  n = np.float(len(Q))
  a = np.float(sum((rpMask['Sim']==True) & (rpMask['Obs']==True)))
  b = np.float(sum((rpMask['Sim']==True) & (rpMask['Obs']==False)))
  c = np.float(sum((rpMask['Sim']==False) & (rpMask['Obs']==True)))
  d = np.float(sum((rpMask['Sim']==False) & (rpMask['Obs']==False)))
  return (n, a, b, c, d)



def parallelize_dataframe(df, func, n_cores=6):
  df_split = np.array_split(df, n_cores)
  pool = Pool(n_cores)
  df = pd.concat(pool.map(func, df_split))
  pool.close()
  pool.join()
  return df



def pandarallelize_dataframe(df, func, n_cores=6):
  # Initialization
  pandarallel.initialize(nb_workers=n_cores, progress_bar=True)
  # # Standard pandas apply
  # df.groupby(column1).column2.rolling(4).apply(func)
  # # Parallel apply
  # df.groupby(column1).column2.rolling(4).parallel_apply(func)
  # # Standard pandas apply
  # df.apply(func, axis=1)
  # Parallel apply
  df.parallel_apply(func)



def processdFrame(df, func):
  df.apply(func, axis=1)


#
# class QFitter():
#
#
#   def __init__(self, dFrame, X, Y=None, ax=plt.gca(), plotMode='T', fitData='Grinorten', fitT=True, tstep='Y', fittersList=None):
#     self.dFrame = dFrame
#     self.X = dFrame[X].sort_values(ascending=True)
#     if Y is None:
#       self.Y = 0
#     else:
#       self.Y = dFrame[Y]
#     self.len = len(self.dFrame)
#     self.nfit = 100
#     self.fits = []
#     self.tstep = tstep
#     self.fitData = fitData
#     self.fitT = fitT
#     self.ax = ax
#     self.plotMode = plotMode
#     if fittersList is None:
#       self.fittersList = ['norm', 'lognorm', 'pearson3', 'gumbel_r', 'linear']
#     else:
#       self.fittersList = fittersList
#
#
#   def fitDischarge(self):
#     fits = []
#     for f in self.fittersList:
#       if self.fitT:
#         Qfit = self.Fit(self, self.X, f, Y=self.T, popt=None)
#         Qfit.plot()
#
#         if f == 'gumbel_r':
#           fit = self.gumbel(self.X, Y=self.Y)
#           popt, perr = fit.fitGumble()
#           fits += [('gumbel_r', popt, perr, Tpgumb)]
#         # elif f == 'longorm':
#         # 	popt, perr = fitLognorm(X, Y)
#         # 	fits += [('lognorm', popt, perr, Tplognorm)]
#       else:
#         if f == 'linear':
#           (slope, intercept), r_value, ppf = fitLinearScipy(X, popt=None)
#         # fitLinearStatsmodels()
#         elif f == 'lognorm':
#           popt = eval('ss.lognorm.fit(X, floc=0)')
#         else:
#           popt = eval('ss.' + f + '.fit(X)')
#         try:
#           dist = eval('ss.' + f + '(*popt)')
#           # RMSE of extreme part of distribution
#           # mcdf = cdf[cdf > 0.8]
#           # my = dist.cdf(X)[cdf > 0.8]
#           perr = rmse(cdf, dist.cdf(X))
#         except AttributeError:
#           pass
#         if f == 'linear':
#           fits += [('linear', (slope, intercept), 1 - r_value, fitLinearScipy)]
#         else:
#           fits += [(f, popt, perr, dist)]
#     return fits
#
#
#   def hydrocdf(self, x, excfitter='Grinorten'):
#     # empirical CDF option
#     if excfitter == 'cdf':
#       n = len(x)
#       # From 1/n to 1, correct for discrete distributions
#       return np.arange(1, n + 1) / n
#       # From 0 to 1-1/n, to match Grinorten's distribution better
#       # return np.arange(0, n) / n
#       # From 0 to 1, is actually a continuous approximation
#       # return = np.arange(0, n) / (n-1)
#     # or Hydrological CDFs
#     elif excfitter == 'Weibull':
#       b = 0.0
#     elif excfitter == 'Hazen':
#       b = 0.5
#     elif excfitter == 'Grinorten':
#       b = 0.44
#     else:
#       raise Exception("Unknown hydrocdf exceedance fitter. Valid options are Weibull, Hazen and Grinorten (default)")
#     return 1.0 - ((x-b) / (self.len+1-2*b))
#
#
#   def calcReturnPeriods(self):
#     retPeriods = []
#     dFrames = []
#     for field in self.dFrame.columns:
#       # Sort Descending to assign rank
#       self.dFrame = self.dFrame.sort_values(field, ascending=False)
#       # Construct the rank m
#       self.dFrame[field + ' rank'] = np.arange(1, self.len + 1)
#       # Sort ascending to make cdf
#       self.dFrame = self.dFrame.sort_values(field + ' rank', ascending=False)
#       # Exceedance probability given by F(x) = (m-b) / (n+1-2*b) (similar to cdf)
#       self.cdf = self.hydrocdf(self.dFrame[field + ' rank'], self.fitData)
#       self.T = invp(self.cdf)
#       # Fit all candidate distributions to the cdf
#       fits = self.fitDischarge()  # to fit on the cdfs
#       # Fit on the return periods
#       # fits = fitDischarge(df[field + ' TGr'])
#       rps = []
#       for f in fits:
#         if f[0] == 'linear':
#           yrp = np.array((ps - f[1][1]) / f[1][0])
#         else:
#           if Tfit:
#             yrp = np.array([f[3](i, *f[1]) for i in T])
#           else:
#             yrp = np.array([f[3].ppf(i) for i in ps])
#         rps += [(f[0], yrp)]
#       retPeriods += [(field, rps)]
#       dFrames += [(field, df)]
#     return retPeriods, dFrames, fits
#
#
#   def plotReturnPeriods(self):
#     # Prepare comon arrays
#     xes = np.linspace(1e-3, 1 - 1e-3, nfit)
#     xinv = invp(xes)
#     yes = df[field]
#     # Grinorten fit
#     x = invp(df[field + ' piGr'])
#     if self.plotMode == 'T':
#       plt.plot(x, yes, 'k', label='Grinorten Empirical')
#     elif self.plotMode == 'cdf':
#       plt.plot(yes, df[field + ' piGr'], 'k', label='Grinorten Empirical')
#     # Plot return periods and cdfs
#     for f in fits:
#       if f[0] == 'linear':
#         y = (xes - f[1][1]) / f[1][0]
#         if self.plotMode == 'T':
#           plt.plot(xinv, y, 'k', label=None, LineWidth=0.5, LineStyle='--')
#         elif self.plotMode == 'cdf':
#           plt.plot(y, xes, 'k', label=None, LineWidth=0.5, LineStyle='--')
#         elif self.plotMode == 'pdf':
#           ax.hist(yes, density=False, stacked=False, histtype='stepfilled', alpha=0.2)
#         # 	yfit = f[1][0]*np.linspace(np.min(yes), np.max(yes), nfit) + f[1][1]
#         # 	yfiti = invp(yfit)
#         # 	dy = yfiti[1]-yfiti[0]
#         # 	pdf = np.diff(yfiti)/dy
#         # 	plt.plot(yfiti[1:], pdf, 'k', label=None, LineWidth=0.5, LineStyle='--')
#         yrp = np.array((ps - f[1][1]) / f[1][0])
#       else:
#         if Tfit:
#           yrp = f[3](T, *f[1])
#           y = f[3](xinv, *f[1])
#         else:
#           yrp = np.array([f[3].ppf(i) for i in ps])
#           y = f[3].ppf(xes)
#         yf = y[np.isfinite(y)]
#         yrpf = yrp[np.isfinite(yrp)]
#         if self.plotMode == 'T':
#           if Tfit:
#             if f[0] == 'gumbel_r':
#               popt, perr = f[1], f[2]
#               xfit = [invTpgumb(i, *popt) for i in xinv]
#               plt.plot(xinv, xfit, 'g', label='Grinorten-T-fitted Gumb', linewidth=5, LineStyle='-', alpha=0.3)
#             elif f[0] == 'lognorm':
#               popt, perr = f[1], f[2]  # fitLognorm(yes, df[field + ' TGr'])
#               xfit = [invTplognorm(i, *popt) for i in xinv]
#               plt.plot(xinv, xfit, 'r', label='Grinorten-T-fitted Gumb', linewidth=5, LineStyle='-', alpha=0.3)
#           else:
#             plt.plot(xinv, yf, 'k', label=None, LineWidth=0.5, LineStyle='--')
#           # DD For debugging and testing, only here because it runs once. has nothing to do with linear fit
#           plt.plot(Tpgumb(yes), yes, 'g', label='theor Gumb', LineWidth=1, LineStyle='--')
#         elif self.plotMode == 'cdf':
#           plt.plot(yf, xes, 'k', label=None, LineWidth=0.5, LineStyle='--')
#         elif self.plotMode == 'pdf':
#           yp = f[3].pdf(y)
#           ypi = yp[np.isfinite(yp)]
#           yfit = np.linspace(np.min(yes), np.max(yes), len(yp))
#           plt.plot(yfit, ypi, 'k', label=None, LineWidth=0.5, LineStyle='--')
#       if self.plotMode == 'T':
#         if Tfit:
#           if f[0] == 'gumbel_r':
#             popt, perr = f[1], f[2]
#             yrp = [invTpgumb(i, *popt) for i in T]
#             plt.plot(T[0:len(yrp)], yrp, label=str(f[0][0:6]) + ': ' + str(invp(f[2])), LineStyle='', Marker='*',
#                      MarkerSize=5)
#           elif f[0] == 'lognorm':
#             popt, perr = f[1], f[2]  # fitLognorm(yes, df[field + ' TGr'])
#             yrp = [invTplognorm(i, *popt) for i in T]
#             plt.plot(T[0:len(yrp)], yrp, label=str(f[0][0:6]) + ': ' + str(invp(f[2])), LineStyle='', Marker='*',
#                      MarkerSize=5)
#           else:
#             plt.plot(T[0:len(yrpf)], yrpf, label=str(f[0][0:6]) + ': ' + str(invp(f[2])), LineStyle='', Marker='*',
#                      MarkerSize=5)
#         else:
#           plt.plot(T[0:len(yrpf)], yrpf, label=str(f[0][0:6]) + ': ' + str(invp(f[2])), LineStyle='', Marker='*',
#                    MarkerSize=5)
#       elif self.plotMode == 'cdf':
#         plt.plot(yrp, ps, label=str(f[0][0:6]) + ': ' + str(f[2]), LineStyle='', Marker='*', MarkerSize=5)
#       elif self.plotMode == 'pdf':
#         try:
#           ax.hist(f[3].rvs(size=10000), density=True, histtype='stepfilled', alpha=0.2)
#         except AttributeError:
#           pass
#     # Esthetics
#     ax.legend(fancybox=True, framealpha=0.8, prop={'size': contFontSize}, labelspacing=0.1, loc='best')
#     if self.plotMode == 'T':
#       # plt.xscale(r'log')
#       ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
#       ax.set_xlim(1, 1.1 * np.max(df[field + ' TGr']))
#       ax.set_xlabel('Return period [' + str(tstep) + ']', fontsize=labelFontSize)
#       ax.set_ylim(0, 1.1 * yes[-1])
#       ax.set_ylabel('Discharge Q [m3/s]', fontsize=labelFontSize)
#       ax.set_xticks(T)
#     elif self.plotMode == 'cdf':
#       ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
#       ax.set_xlim(0.95 * yes[0], 1.05 * yes[-1])
#       ax.set_xlabel('Discharge Q [m3/s]', fontsize=labelFontSize)
#       ax.set_ylim(0, 1)
#       ax.set_ylabel('Probability', fontsize=labelFontSize)
#       ax.set_yticks(np.linspace(0, 1, 11))
#     elif self.plotMode == 'pdf':
#       ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
#       ax.set_xlabel('Discharge Q [m3/s]', fontsize=labelFontSize)
#       ax.set_xlim(0.95 * yes[0], 1.05 * yes[-1])
#       ax.set_ylabel('Probability', fontsize=labelFontSize)
#



#   class Fit():
#     def __init__(self, parent, X, fitter, Y=None, popt=None):
#       print(fitter)
#       self.parent = parent
#       self.X = X
#       if not Y is None:
#         self.Y = Y
#       self.len = parent.len
#       self.nfit = parent.nfit
#       self.setFit(fitter, ax=parent.ax, plotMode=parent.plotMode, fitT=parent.fitT, tstep=parent.tstep, popt=popt)
#     def setFit(self, fitter, ax=plt.gca(), plotMode='T', fitT=True, tstep='Y', popt=None):
#       Tyr = [1.5, 2, 5, 10, 20, 50, 100, 200, 500]
#       self.tstep = tstep
#       if self.tstep == 'D':
#         self.T = [i * 365.25 for i in Tyr]
#       elif self.tstep == 'M':
#         self.T = [i * 12 for i in Tyr]
#       elif self.tstep == 'Y':
#         self.T = Tyr
#       self.ps = [(i - 1.0) / i for i in self.T]
#       self.Tfit = np.linspace(self.T[0], self.T[-1], self.nfit)
#       self.pfit = np.linspace(1e-3, 1 - 1e-3, self.nfit)
#       self.fitT = fitT
#       self.tstep = tstep
#       self.ax = ax
#       self.plotMode = plotMode
#       self.fitter = fitter
#       self.popt = popt
#     def cdf(self, popt=None):
#       return self.parent.cdf #CALC IT! BLAAT
#     def pdf(self, popt=None):
#       return self.parent.pdf
#     def Tdf(self, popt=None):
#       return self.parent.T
#     def iTdf(self, popt=None):
#       return self.X
#     def fit(self):
#       self.popt, pcov = curve_fit(self.Tdf, self.X, self.Y)
#       self.perr = np.sqrt(np.diag(pcov))
#     def plot(self):
#       # Plot the cdf
#       if self.plotMode == 'T':
#         x = self.Tdf()
#         y = self.X
#       elif self.plotMode == 'cdf':
#         x = self.X
#         y = self.cdf()
#       elif self.plotMode == 'pdf':
#         x = self.X
#         y = self.pdf()
#       plt.plot(x, y[0:len(x)], 'k', label=self.fitter)
#       # Plot the fitted curve
#       if self.plotMode == 'T':
#         xfit = self.Tfit
#         yfit = self.Tdf(xfit, self.popt)
#       elif self.plotMode == 'cdf':
#         yfit = self.pfit
#         xfit = self.iTdf(invp(yfit), self.popt)
#       elif self.plotMode == 'pdf':
#         xfit = self.iTdf(invp(self.pfit), self.popt)
#         yfit = self.pdf(xfit)
#         try:
#           self.ax.hist(self.X, density=True, histtype='stepfilled', alpha=0.2)
#         except AttributeError:
#           pass
#       plt.plot(xfit, yfit[0:len(xfit)], 'k', label=None, LineWidth=0.5, LineStyle='--')
#       # Plot the return period markers
#       if self.plotMode == 'T':
#         xrp = self.T
#         yrp = self.Tdf(xrp, self.popt)
#       elif self.plotMode == 'cdf':
#         yrp = self.ps
#         xrp = self.iTdf(invp(yrp), self.popt)
#       elif self.plotMode == 'pdf':
#         xrp = self.iTdf(invp(self.ps), self.popt)
#         yrp = self.pdf(xrp)
#       plt.plot(xrp, yrp[0:len(xrp)], label = self.fitter + ': ' + str(self.perr), LineStyle = '', Marker = '*')
#
#
#
#   class Gumbel(Fit):
#     def __init__(self, X, fitter, Y=None, popt=None):
#       print("Gumbel")
#       super().__init__(X, fitter, Y, popt)
#       # Gumbel-transformed return periods
#       self.TGumbelTheoretical = self.Tpgumb(self.X)
#     def cdf(self, a=None, u=None):
#       # Gumbel fit parameters
#       if a is None:
#         a = np.sqrt(6) * np.nanstd(self.X, ddof=1) / np.pi
#       if u is None:
#         u = np.nanmean(self.X) - 0.5772 * a
#       return np.exp(-np.exp(-1.0 * ((self.X - u) / a)))
#     def Tdf(self, a=None, u=None):
#       # Gumbel fit parameters
#       if a is None:
#         a = np.sqrt(6) * np.nanstd(self.X, ddof=1) / np.pi
#       if u is None:
#         u = np.nanmean(self.X) - 0.5772 * a
#       return self.X.invp(pgumb(self.X, a, u))
#     def iTdf(self, a=None, u=None):
#       # Gumbel fit parameters
#       if a is None:
#         a = np.sqrt(6) * np.nanstd(self.Y, ddof=1) / np.pi
#       if u is None:
#         u = np.nanmean(self.Y) - 0.5772 * a
#       return -a * np.log(np.log(self.Y / (self.Y - 1))) + u
#
#
#
# def invT(T):
#   return (T - 1) / T
#
#
#
# def invp(p):
#   if isinstance(p, float):
#     return 1.0 / (1.0 - p)
#   else:
#     return 1.0 / (1.0 - p[p < 1])
#   pl = p[p < 1]
#   ph = p[p > 1]
#   pp = 1.0 / (1.0 - pl)
#   if len(ph) > 0:
#     if isinstance(ph, np.ndarray):
#       pp = np.concatenate([pp, 1.0 / ph])
#     else:  # if isinstance(ph, list):
#       pp += 1.0 / ph
#   return pp
#
#
#
# def pgumb(x, a=None, u=None):
#   if a is None:
#     a = np.sqrt(6) * np.nanstd(x, ddof=1) / np.pi
#   if u is None:
#     u = np.nanmean(x) - 0.5772 * a
#   return np.exp(-np.exp(-1.0 * ((x - u) / a)))
#
# def Tgumb(y, a=None, u=None):
#   if a is None:
#     a = np.sqrt(6) * np.nanstd(y, ddof=1) / np.pi
#   if u is None:
#     u = np.nanmean(y) - 0.5772 * a
#   return -a * np.log(np.log(y / (y - 1))) + u
#
# def fitgumb(x, y):
#   a = np.sqrt(6) * np.nanstd(y, ddof=1) / np.pi
#   u = np.nanmean(y) - 0.5772 * a
#   popt, pcov = curve_fit(Tgumb, x, y, p0=[a, u])
#   # Finds the standard deviations of given parameters alpha and u
#   perr = np.sqrt(np.diag(pcov))
#   return popt, perr
#
#
#
# def plognorm(x, s=None, u=None):
#   if s is None:
#     s = np.nanstd(x)
#   if u is None:
#     u = np.nanmean(x)
#   return 0.5 + 0.5 * np.array([math.erf((np.log(ix) - u) / np.sqrt(2) / s) for ix in x])
#
# def Tlognorm(y, s=None, u=None):
#   if s is None:
#     s = np.nanstd(y)
#   if u is None:
#     u = np.nanmean(y)
#   return np.exp(np.sqrt(2)*s*math.erf((y-2)/y)**-1 + u)
#
# def fitlognorm(x, y):
#   s = np.nanstd(y)
#   u = np.nanmean(y)
#   popt, pcov = curve_fit(Tlognorm, x, y, p0=[s, u])
#   # Finds the standard deviations of given parameters alpha and u
#   perr = np.sqrt(np.diag(pcov))
#   return popt, perr
#
#
#
# def ppearson3(x, k=None):
#   if k is None:
#     k = len(x)-1
#   return sp.gammainc(k/2, x/2) / sp.gamma(k/2)
#
#
#
# def plinear(x, a=None, b=None):
#   if a is None:
#     a = 1
#   if b is None:
#     b = 0
#   return a*x + b
#
# def Tlinear(y, a=None, b=None):
#   if a is None:
#     a = 1
#   if b is None:
#     b = 0
#   return (y-b)/a
#
# def fitlinear(x, y):
#   popt, pcov = curve_fit(Tlinear, x, y)
#   # Finds the standard deviations of given parameters alpha and u
#   perr = np.sqrt(np.diag(pcov))
#   return popt, perr
#
#
#
# def fitLinearScipy(x, popt=None):
#   if popt is None:
#     n = len(x)
#     cdf = np.arange(1, n+1) / n
#     slope, intercept, r_value, p_value, std_err = ss.linregress(x, cdf)
#   else:
#     slope, intercept = popt
#     r_value = -1
#   return (slope, intercept), r_value, intercept + slope * x
#
#
#
# def fitLinearStatsmodels():
#   # Make data #
#   x = range(30)
#   y = [y * (10 + random()) + 200 for y in x]
#   # Add outlier #
#   x.insert(6, 15)
#   y.insert(6, 220)
#   # Make fit #
#   regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
#   # Find outliers #
#   test = regression.outlier_test()
#   outliers = ((x[i], y[i]) for i, t in enumerate(test.icol(2)) if t < 0.5)
#   print('Outliers: ', list(outliers))
#   # Figure #
#   figure = smgraphics.regressionplots.plot_fit(regression, 1)
#   # Add line #
#   smgraphics.regressionplots.abline_plot(model_results=regression, ax=figure.axes[0])
#


def rmse(x, x0):
  return np.sqrt(np.nanmean((x - x0) ** 2))



def month2string(m):
    if m==1:
      return 'Jan'
    elif m==2:
      return 'Feb'
    elif m==3:
      return 'Mar'
    elif m==4:
      return 'Apr'
    elif m==5:
      return 'May'
    elif m==6:
      return 'Jun'
    elif m==7:
      return 'Jul'
    elif m==8:
      return 'Aug'
    elif m==9:
      return 'Sep'
    elif m==10:
      return 'Oct'
    elif m==11:
      return 'Nov'
    elif m==12:
      return 'Dec'
    else:
      raise Exception('Invalid month digit given. Should be 1 ~ 12')






def main(*args, **kwargs):

  rc('font',**{'family':'DejaVu Sans','serif':['Palatino']})
  rc('text', usetex=True) # DD need to add these latex directives to interpret things like /huge and /bfseries
  rcParams.update({'figure.autolayout': True})

  iniFile = os.path.normpath(sys.argv[1])
  (drive, path) = os.path.splitdrive(iniFile)
  (path, fil)  = os.path.split(path)
  print(">> Reading settings file ("+fil+")...")

  file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

  if ver.find('3.') > -1:
    parser = ConfigParser()  # python 3.6.10
  else:
    parser = SafeConfigParser()  # python 2.7.15
  parser.read(iniFile)

  ObservationsStart = dt.datetime.strptime(parser.get('DEFAULT', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
  ObservationsEnd = dt.datetime.strptime(parser.get('DEFAULT', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
  ForcingStart = dt.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
  ForcingEnd = dt.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing

  WarmupDays = 0 #int(parser.get('DEFAULT', 'WarmupDays'))

  CatchmentDataPath = parser.get('Path','CatchmentDataPath')
  SubCatchmentPath = parser.get('Path','SubCatchmentPath')
  srcPath = parser.get('Path', 'Templates').replace('templates/', '')

  path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
  path_result = parser.get('Path', 'Result')

  Qtss_csv = parser.get('CSV', 'Qtss')
  Qmeta_csv = parser.get('CSV', 'Qmeta')
  calibrationFreq = parser.get('DEFAULT', 'calibrationFreq')

  ########################################################################
  #   Loop through catchments
  ########################################################################

  # Reading longterm run data
  print(">> Reading longtermrun output...")
  # longtermdataFile = "disWin.csv"
  # if os.path.exists(longtermdataFile.replace(".csv", ".npy")) and os.path.getsize(
  # 	longtermdataFile.replace(".csv", ".npy")) > 0:
  # 	longtermdata = pd.DataFrame(np.load(longtermdataFile.replace(".csv", ".npy"), allow_pickle=True))
  # 	longtermdata.index = np.load(longtermdataFile.replace(".csv", "_index.npy"), allow_pickle=True)
  # 	longtermdata.columns = np.load(longtermdataFile.replace(".csv", "_columns.npy"), allow_pickle=True)
  # else:
  # 	longtermdata = pd.read_csv(os.path.join(path_result, longtermdataFile), sep=',', index_col=0)
  # 	np.save(longtermdataFile.replace(".csv", ".npy"), longtermdata)
  # 	np.save(longtermdataFile.replace(".csv", "_index.npy"), longtermdata.index)
  # 	np.save(longtermdataFile.replace(".csv", "_columns.npy"), longtermdata.columns.values)
  # discharges.nc
  ds = xr.open_dataset('discharges_efas41.nc')
  longtermdata = 	ds['dis'].to_dataframe()
  newSim = pd.read_csv(os.path.join(SubCatchmentPath, 'streamflow_simulated_best.csv'))
  # Reading station data
  print(">> Reading Qmeta2.csv file...")
  stationdataFile = "Qmeta2.csv"
  if os.path.exists(stationdataFile.replace(".csv", ".npy")) and os.path.getsize(
    stationdataFile.replace(".csv", ".npy")) > 0:
    stationdata = pd.DataFrame(np.load(stationdataFile.replace(".csv", ".npy"), allow_pickle=True, encoding='latin1'))
    stationdata.index = np.load(stationdataFile.replace(".csv", "_index.npy"), allow_pickle=True, encoding='latin1')
    stationdata.columns = np.load(stationdataFile.replace(".csv", "_columns.npy"), allow_pickle=True, encoding='latin1')
  else:
    stationdata = pd.read_csv(os.path.join(path_result, stationdataFile), sep=",", index_col=0)
    np.save(stationdataFile.replace(".csv", ".npy"), stationdata)
    np.save(stationdataFile.replace(".csv", "_index.npy"), stationdata.index)
    np.save(stationdataFile.replace(".csv", "_columns.npy"), stationdata.columns.values)
  stationdata['ObsID'] = stationdata.index
  catchments = stationdata['ObsID'].to_xarray()

  # Reading station observed discharge
  print(">> Reading ecQts.csv file...")
  if os.path.exists(Qtss_csv.replace(".csv", ".npy")) and os.path.getsize(Qtss_csv.replace(".csv", ".npy")) > 0:
    streamflow_data = pd.DataFrame(np.load(Qtss_csv.replace(".csv", ".npy"), allow_pickle=True))
    streamflow_datetimes = np.load(Qtss_csv.replace(".csv", "_dates.npy"), allow_pickle=True).astype('string_')
    # streamflow_data.index = [dt.datetime.strptime(i.decode('utf-8'), "%Y-%m-%dT%H:%M:%S.000000000") for i in streamflow_datetimes]
    streamflow_data.index = [dt.datetime.strptime(i.decode('utf-8'), "%d/%m/%Y %H:%M") for i in streamflow_datetimes]
    streamflow_data.columns = np.load(Qtss_csv.replace(".csv", "_catchments.npy"), allow_pickle=True)
  else:
    streamflow_data = pd.read_csv(Qtss_csv, sep=",", index_col=0)
    streamflow_data.index = pd.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
    # streamflow_data = pandas.read_csv(Qtss_csv, sep=",", index_col=0, parse_dates=True) # DD WARNING buggy unreliable parse_dates! Don't use it!
    np.save(Qtss_csv.replace(".csv", ".npy"), streamflow_data)
    np.save(Qtss_csv.replace(".csv", "_dates.npy"), streamflow_data.index)
    np.save(Qtss_csv.replace(".csv", "_catchments.npy"), streamflow_data.columns.values)

  print(">> Reading Return Periods...")
  returnPeriods = xr.open_dataset(os.path.join(srcPath, "return_levels.nc"))

  CatchmentsToProcess = pd.read_csv(file_CatchmentsToProcess,sep=",",header=None)

  stationdata['KGE_cal'] = np.nan
  stationdata['KGE_val'] = np.nan
  stationdata['NSE_cal'] = np.nan
  stationdata['NSE_val'] = np.nan


  newCsvDict = {}

  # @ray.remote
  # def processStations(stationdata):
  #   row = stationdata
  for index, row in stationdata.iterrows():
    #if row['ObsID'] == 2541:
    #	return
    #try:
    #	row
    Series = CatchmentsToProcess
    if len(Series[Series == index]) == 0:  # Only process catchments whose ID is in the CatchmentsToProcess.txt file
      continue
    path_subcatch = path_result  # os.path.join(SubCatchmentPath,str(row['ObsID']))
    # Skip already processed plots
    jsonFile = os.path.join(path_subcatch, 'WEB', str(index) + '.json')
    if os.path.exists(jsonFile) and os.path.getsize(jsonFile) > 0:
      print("Skipping " + jsonFile)
      continue
    # Make figures directory
    try:
      os.stat(os.path.join(path_subcatch, "WEB"))
    except:
      os.mkdir(os.path.join(path_subcatch, "WEB"))

    # # Delete contents of figures directory
    # for filename in glob.glob(os.path.join(path_subcatch,"WEB",'*.*')):
    # 	os.remove(filename)

    # Compute the time steps at which the calibration should start and end
    if row['Val_Start'][:10] != "Streamflow":  # Check if Q record is long enough for validation
      Val_Start = dt.datetime.strptime(row['Val_Start'], "%d/%m/%Y %H:%M")
      Val_End = dt.datetime.strptime(row['Val_End'], "%d/%m/%Y %H:%M")
      Val_Start_Step = (Val_Start - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
      Val_End_Step = (Val_End - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
    else:
      Val_Start = []
      Val_End = []
      Val_Start_Step = []
      Val_End_Step = []
    Cal_Start = dt.datetime.strptime(row['Cal_Start'], "%d/%m/%Y %H:%M")
    Cal_End = dt.datetime.strptime(row['Cal_End'], "%d/%m/%Y %H:%M")
    Cal_Start_Step = (Cal_Start - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
    Cal_End_Step = (Cal_End - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!
    Forcing_End_Step = (ForcingEnd - ForcingStart).days + 1  # For LISFLOOD, not for indexing in Python!!!

    # Load observed streamflow
    Qobs = streamflow_data[str(index)]
    Qobs[Qobs < 0] = np.NaN

    # Load streamflow of longterm run
    # Used when extracting from disWin.csv
    # Qsim = longtermdata[str(row['ObsID'])]
    # Qsim.index = [dt.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in Qsim.index]
    # Used when extracting from discharges.nc
    try:
      idx = pd.IndexSlice
      Qsim = longtermdata.loc[idx[:, int(index)], :]
      Qsim.index = [i[0] for i in Qsim.index]
    except KeyError:
      print("MISSING " + str(index))
      os.system("echo " + str(index) + " >> missingCatchments.txt")
      continue
    # Handle 24-hourly catchments
    if calibrationFreq == r"6-hourly":
      if row["CAL_TYPE"].find("_24h") > -1:
        Qsim = Qsim.resample('24H', label="right", closed="right").mean()
        Qobs = Qobs.resample('24H', label="right", closed="right").mean()
    elif calibrationFreq == r"daily":
      # DD Untested code! DEBUG TODO
      Qobs = Qobs.resample('24H', label="right", closed="right").mean()

    # Make dataframe with aligned Qsim and Qobs columns
    Q = pd.concat({"Sim": Qsim['dis'], "Obs": Qobs}, axis=1)  # .reset_index()
    # Spread nans in obs to Q['Sim']
    Q[np.isnan(Q['Obs'])] = np.nan  # better as it leaves the gaps in the plot.
    # Q = Q.dropna() # not great as it results in connecting lines between periods of missing data
    # Drop the nans at the front of the dataset
    firstStep = np.min(Q[Q['Obs'].notna() == 1].index)
    lastStep = np.max(Q[Q['Obs'].notna() == 1].index)
    Q = Q.drop(index=Q.index[Q.index < firstStep])
    Q = Q.drop(index=Q.index[Q.index > lastStep])

    # Get the specific sim and obs BLAAT

    Dates_Cal = Q.loc[Cal_Start:Cal_End].index
    Q_sim_Cal = Q.loc[Cal_Start:Cal_End]['Sim'].values
    Q_obs_Cal = Q.loc[Cal_Start:Cal_End]['Obs'].values
    # Dates_Cal2 = Q.index
    # Q_sim_Cal2 = Q['Sim'].values
    # Q_obs_Cal2 = Q['Obs'].values

    # Extract the return periods from Lorenzo's script
    # Return period
    pixelData = returnPeriods.sel(x=row['LisfloodX'], y=row['LisfloodY'])
    # 1.5-year
    rp1_5 = pixelData['rl1.5'].values
    # 2-year
    rp2 = pixelData['rl2'].values
    # 5-year
    rp5 = pixelData['rl5'].values
    # 10-year
    rp10 = pixelData['rl10'].values
    # 20-year
    rp20 = pixelData['rl20'].values
    # 50-year
    rp50 = pixelData['rl50'].values
    # 100-year
    rp100 = pixelData['rl100'].values
    # 200-year
    rp200 = pixelData['rl200'].values
    # 500-year
    rp500 = pixelData['rl500'].values

    # Manually calculate return periods
    # Extract a clean dataframe with a single index
    Qsimf = longtermdata.loc[idx[:, int(index)], :]
    Qsimf.index = [i[0] for i in Qsimf.index]
    Qsimf = Qsimf.resample('1M', label="right", closed="right").max().dropna()
    # Qpyr = Qs.dropna().resample('1Y', label="right", closed="right").max()
    # Qpday = Qs.dropna().resample('1D', label="right", closed="right").max() #.apply(nanmax))
    # Load example
    # ex = pd.read_csv('/perm/rd/nedd/EFAS/returnPeriods/annual_peak_streamflow_data.csv', sep=",", index_col=0)
    # rpsex, dFramesex, fitsex = calcReturnPeriods(ex, tstep='Y')
    #
    # def calcThresholds(pp, ps):
    #   n = len(ps)
    #   cdf = np.arange(1, n + 1) / n
    #   return [np.nanmin(np.nanmax(ps[ps>p])) for p in pp]
    #
    # calcThresholds([0.1, 0.9, 0.95], Qsimf)
    #
    #
    # Qf = QFitter(Qsimf, 'dis', Y=None, ax=plt.gca(), plotMode='T', fitData='Grinorten', fitT=True, tstep='M', fittersList=['gumbel_r'])
    # Qf.calcReturnPeriods()
    # Qf.saveReturnPeriods(fileout)
    # # Calculate return periods ourselves
    # Qs = Qsim
    # def nanmax(x):
    #   y = np.nanmax(x)
    #   return y
    #
    # Create the contingency tables for the return periods
    rp1_5ct = createContingencyTable(rp1_5, Q)
    rp2ct = createContingencyTable(rp2, Q)
    rp5ct = createContingencyTable(rp5, Q)
    rp10ct = createContingencyTable(rp10, Q)
    rp20ct = createContingencyTable(rp20, Q)
    rp50ct = createContingencyTable(rp50, Q)
    rp100ct = createContingencyTable(rp100, Q)
    rp200ct = createContingencyTable(rp200, Q)
    rp500ct = createContingencyTable(rp500, Q)

    # Extract other stats to plot
    # Extract values for Kling-Gupta efficiency
    KGE = HydroStats.fKGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)[0]
    # Extract values for correlation
    r = HydroStats.fKGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)[1]
    # Extract values for bias
    B = HydroStats.fKGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)[2]
    # Extract values for spread
    S = HydroStats.fKGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)[3]
    # Nash-Sutcliffe efficiency
    NSE = HydroStats.NS(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)

    newCsvDict[index] = {'ObsID': index, 'KGE': KGE, 'corr': r, 'LisfloodX': row['LisfloodX'], 'LisfloodY': row['LisfloodY']}




    # Climatology of Q
    Q_obs_clim_Cal = np.zeros(shape=(12, 1)) * np.NaN
    Q_sim_clim_Cal = np.zeros(shape=(12, 1)) * np.NaN
    Q_obs_clim_Cal_stddev = np.zeros(shape=(12, 1)) * np.NaN
    Q_sim_clim_Cal_stddev = np.zeros(shape=(12, 1)) * np.NaN
    Q_obs_monthData = []
    Q_sim_monthData = []
    for month in range(1, 13):
      mask = ~np.isnan(Q_obs_Cal) & ~np.isnan(Q_sim_Cal)
      # Obs
      Q_obs_month = Q_obs_Cal[(Dates_Cal.month == month) & mask]
      Q_obs_clim_Cal[month - 1] = np.mean(Q_obs_month)
      Q_obs_clim_Cal_stddev[month - 1] = np.std(Q_obs_month)
      Q_obs_monthData.append(Q_obs_month)
      # Sim
      Q_sim_month = Q_sim_Cal[(Dates_Cal.month == month) & mask]
      Q_sim_clim_Cal[month - 1] = np.mean(Q_sim_month)
      Q_sim_clim_Cal_stddev[month - 1] = np.std(Q_sim_month)
      Q_sim_monthData.append(Q_sim_month)

    Qp = Q.dropna()
    if row["CAL_TYPE"].find("_24h") > -1:
      period = len(Qp) / 365.
    else:
      period = len(Qp) / 365. / 4.

    def drawReturnPeriods(ax, mindate=None, maxdate=None):
      # Return period
      if mindate is None or maxdate is None:
        mindate = Dates_Cal[0].value * 1e-9
        maxdate = Dates_Cal[-1].value * 1e-9
      # 1.5-year
      plt.hlines(rp1_5, color=green, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
      ax.text(x=maxdate + 1, y=rp1_5, s='1.5-year', color=green, verticalalignment='center', fontsize=contFontSize)
      # 2-year
      plt.hlines(rp2, color=orange, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
      ax.text(x=maxdate + 1, y=rp2, s='2-year', color=orange, verticalalignment='center', fontsize=contFontSize)
      # 5-year
      plt.hlines(rp5, color=red, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
      ax.text(x=maxdate + 1, y=rp5, s='5-year', color=red, verticalalignment='center', fontsize=contFontSize)
      # # 10-year
      # plt.hlines(rp10, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
      # ax.text(x=maxdate + 1, y=rp10, s='10-year', color='black', verticalalignment='center', fontsize=contFontSize)
      # 20-year
      if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp20:
        plt.hlines(rp20, color=magenta, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
        ax.text(x=maxdate + 10, y=rp20, s='20-year', color=magenta, verticalalignment='center', fontsize=contFontSize)

    # # 50-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp50:
    # 	plt.hlines(rp50, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp50, s='50-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # # 100-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp100:
    # 	plt.hlines(rp100, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp100, s='100-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # # 200-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp200:
    # 	plt.hlines(rp200, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp200, s='200-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # # 500-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp500:
    # 	plt.hlines(rp500, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp500, s='500-year', color='black', verticalalignment='center', fontsize=contFontSize)

    #### CREATE PLOTS FROM HERE ###

    # FIGURE OF return periods

    fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig)
    # create axes
    ax1 = fig.add_subplot(gs[0,0])
    Qf.plotReturnPeriods()
    # plotReturnPeriods(ax1, dFramesmonTfit[0][1], 'dis', fitsmonTfit, mode='T', tstep='M', Tfit=True)
    # drawReturnPeriods(ax1, mindate=0.95*Tmon[0], maxdate=Tmon[-1]*1.05)
    # ax2 = fig.add_subplot(gs[0,1])
    # plotReturnPeriods(ax2, dFramesmonpfit[0][1], 'dis', fitsmonpfit, mode='T', tstep='M', Tfit=False)
    # drawReturnPeriods(ax2, mindate=0.95*Tmon[0], maxdate=Tmon[-1]*1.05)
    # # ax2 = fig.add_subplot(gs[0,1])
    # # plotReturnPeriods(ax2, dFrames[1][1], 'Obs', fits, mode='T', tstep='M')
    # ax3 = fig.add_subplot(gs[1,0])
    # plotReturnPeriods(ax3, dFrames[0][1], 'dis', fits, mode='cdf', tstep='M')
    # # ax4 = fig.add_subplot(gs[1,1])
    # # plotReturnPeriods(ax4, dFrames[1][1], 'Obs', fits, mode='cdf', tstep='Y')
    # ax5 = fig.add_subplot(gs[2, 0])
    # plotReturnPeriods(ax5, dFrames[0][1], 'dis', fits, mode='pdf', tstep='M')
    # # ax6 = fig.add_subplot(gs[2, 1])
    # # plotReturnPeriods(ax6, dFrames[1][1], 'Obs', fits, mode='pdf', tstep='M')
    maximizePlot()
    print("WAIT")
    # plt.scatter(ex['T'], ex[None], color=purple, linewidth=1)
    # # Esthetics
    # ax.set_title('Monthly discharge climatology in calibration period', fontsize=titleFontSize)
    # ax.grid(b=True, axis='y')
    # plt.rcParams["font.size"] = 14
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # # horizontal axis
    # plt.xlabel(r'Month')
    # plt.xticks(range(1, 16), [""] + list(months))
    # plt.xlim([1.5, 15.5])
    # # vertical axis
    # plt.ylabel(r'Discharge [m3/s]')
    # # Add manually-made legend
    # genLegend()
    # # Restore the correct active axes
    # ax = fig.get_axes()[0]
    # plt.sca(ax)
    # # Maximize the window for optimal view
    # maximizePlot()
    # # linear scale
    # logscale = False
    # if logscale:
    # 	plt.yscale(r'log')
    # 	ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
    # 	ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    # 	plt.ylim([1, 2 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # else:
    # 	plt.yscale(r'linear')
    # 	plt.ylim([1, 1.05 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # # Save the linear scale figure
    # plt.savefig(os.path.join(path_subcatch, "WEB", str(index) + "_Q_clim_linear.svg"), format='svg')
    # # Log scale for clearer skill cover
    # logscale = True
    # if logscale:
    # 	plt.yscale(r'log')
    # 	ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
    # 	ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    # 	plt.ylim([1, 2 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # else:
    # 	plt.ylim([1, 1.05 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # # Save the logarithmic scale figure
    # plt.savefig(os.path.join(path_subcatch,"WEB", str(index) + "_Q_clim_log.svg"), format='svg')
    # plt.close("all")



    # ROEBBER DIAGRAM
    # Update the font before creating any plot objects
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams.update()
    # Start figure
    fig = plt.figure()
    # Define a subplot positioning grid
    gs = GridSpec(3, 1, figure=fig)
    # ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[:2, :])
    # Threat Score (Critical Success Index)
    SRpoints = np.arange(0, 1.0001, 0.01)
    Hpoints = np.arange(0, 1.0001, 0.01)
    X, Y = np.meshgrid(SRpoints, Hpoints)
    Z = 1 / (1 / X + 1 / Y - 1)
    Z[np.logical_not(np.isfinite(Z))] = 0
    TS = ax.contour(X, Y, Z, levels=[0.1 * i for i in range(10)], colors='k', linewidths=0.5, zorder=0)
    cb = ax.clabel(TS, TS.levels, inline=False, fmt='TS=%.1f', fontsize=contFontSize, inline_spacing=0)
    [txt.set_bbox(dict(boxstyle='square,pad=0', fc='w', ec='w')) for txt in cb]
    # Bias
    # DD Using the contour, we get tightly-packed lines near (0, 0)
    Z = Y / X
    Z[np.logical_not(np.isfinite(Z))] = 0
    Bfit = ax.contour(X, Y, Z, levels=[i / 10. for i in range(1, 10)] + [10. / i for i in range(10, 0, -1)],
                      colors='k', linewidths=0.5, linestyles='dashed')
    for c in Bfit.collections:
      c.set_dashes([(0, (2.0, 2.0))])
    # cb = ax.clabel(B, B.levels, inline=False, fmt='B=%.1f', fontsize=contFontSize, inline_spacing=0)
    # [txt.set_bbox(dict(boxstyle='square,pad=0', fc='w', ec='w')) for txt in cb]
    # DD Manually draw the lines instead, but don't plot beyond the plot axes
    for b in [i / 10. for i in range(1, 10)] + [10. / i for i in range(10, 0, -1)]:
      if b < 1:
        # plt.plot([0, 1.0], [0, b], color='k', lineWidth=0.5, lineStyle='-.', zorder=0)
        ax.text(x=1.01, y=b, s='B=%.1f' % b, color='k', verticalalignment='center', fontsize=axesFontSize,
                fontfamily='Arial')
      else:
        # line1 = LineString([[0, 0], [1, b]])
        # line2 = LineString([[0, 1], [1, 1]])
        # int_pt = line1.intersection(line2)
        # plt.plot([0, int_pt.x], [0, int_pt.y], color='k', lineWidth=0.5, lineStyle='-.', zorder=0)
        ax.text(x=1 / b, y=1.025, s='B=%.1f' % (np.round(b * 10) / 10.0), color='k', horizontalalignment='center',
                verticalalignment='center', fontsize=axesFontSize, fontfamily='Arial')
    # Plot the data points for the various return periods
    plt.plot(1 - bfs.FAR(*rp1_5ct), bfs.POD(*rp1_5ct), marker='X', color=green, markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp1_5ct))+0.05, bfs.POD(*rp1_5ct), s='1.5-year', color=green, horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    plt.plot(1 - bfs.FAR(*rp2ct), bfs.POD(*rp2ct), marker='X', color=orange, markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp2ct))+0.05, bfs.POD(*rp2ct), s='2-year', color=orange, horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    plt.plot(1 - bfs.FAR(*rp5ct), bfs.POD(*rp5ct), marker='X', color=red, markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp5ct))+0.05, bfs.POD(*rp5ct), s='5-year', color=red, horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    # plt.plot(1 - bfs.FAR(*rp10ct), bfs.POD(*rp10ct), marker='X', color='k', markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp10ct))+0.05, bfs.POD(*rp10ct), s='10-year', color='k', horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    plt.plot(1 - bfs.FAR(*rp20ct), bfs.POD(*rp20ct), marker='X', color=magenta, markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp20ct))+0.05, bfs.POD(*rp20ct), s='20-year', color=magenta, horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    # plt.plot(1 - bfs.FAR(*rp50ct), bfs.POD(*rp50ct), marker='X', color='k', markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp50ct))+0.05, bfs.POD(*rp50ct), s='50-year', color='k', horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    # plt.plot(1 - bfs.FAR(*rp100ct), bfs.POD(*rp100ct), marker='X', color='k', markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp100ct))+0.05, bfs.POD(*rp100ct), s='100-year', color='k', horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    # plt.plot(1 - bfs.FAR(*rp200ct), bfs.POD(*rp200ct), marker='X', color='k', markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp200ct))+0.05, bfs.POD(*rp200ct), s='200-year', color='k', horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    # plt.plot(1 - bfs.FAR(*rp500ct), bfs.POD(*rp500ct), marker='X', color='k', markerSize=15, zorder=10)
    # ax.text((1-bfs.FAR(*rp500ct))+0.05, bfs.POD(*rp500ct), s='500-year', color='k', horizontalalignment='left', verticalalignment='center', fontSize=12, fontWeight='bold', zorder=10)
    # Add a table at the bottom of the axes
    tbl = plt.table(
      cellText=[
        ["{0:.2f}".format(1 - bfs.FAR(*rp1_5ct)), "{0:.2f}".format(1 - bfs.FAR(*rp2ct)),
         "{0:.2f}".format(1 - bfs.FAR(*rp5ct)), "{0:.2f}".format(1 - bfs.FAR(*rp20ct))],
        ["{0:.2f}".format(bfs.HR(*rp1_5ct)), "{0:.2f}".format(bfs.HR(*rp2ct)), "{0:.2f}".format(bfs.HR(*rp5ct)),
         "{0:.2f}".format(bfs.HR(*rp20ct))],
        ["{0:.2f}".format(bfs.B(*rp1_5ct)), "{0:.2f}".format(bfs.B(*rp2ct)), "{0:.2f}".format(bfs.B(*rp5ct)),
         "{0:.2f}".format(bfs.B(*rp20ct))],
        ["{0:.2f}".format(bfs.TS(*rp1_5ct)), "{0:.2f}".format(bfs.TS(*rp2ct)), "{0:.2f}".format(bfs.TS(*rp5ct)),
         "{0:.2f}".format(bfs.TS(*rp20ct))],
        ["{0:.4f}".format(bfs.PCR(*rp1_5ct)), "{0:.4f}".format(bfs.PCR(*rp2ct)), "{0:.4f}".format(bfs.PCR(*rp5ct)),
         "{0:.4f}".format(bfs.PCR(*rp20ct))]
      ],
      rowLabels=['Succces Rate (SR)', 'Hit Rate (HR)', 'Bias (B)', 'Threat Score (TS)',
                 'Percent Correct Rejections (PCR)'],
      rowColours=[KGEgrey for i in range(5)],
      rowLoc='center',
      colColours=[green, orange, red, magenta],
      colLabels=['1.5-year', '2-year', '5-year', '20-year'],
      colWidths=[0.1, 0.1, 0.1, 0.1, 0.1],
      cellColours=[[green, orange, red, magenta] for i in range(5)],
      cellLoc='center',
      bbox=(0.33, -0.7, 0.6, 0.6)
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(tableFontSize)
    # tbl.set_text_props({'fontfamily': 'Arial'})
    tbl.scale(1, 1)
    # esthetics
    # ax.set_title('Roebber Diagram', fontsize=titleFontSize)
    ax.axis('equal', adjustable='box')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_position(('data', 1))
    ax.spines['right'].set_position(('data', 1))
    ax.spines["bottom"].set_bounds(0, 1)
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["top"].set_bounds(0, 1)
    ax.spines["right"].set_bounds(0, 1)
    ax.set_xticks(np.arange(0, 1.00001, 0.1))
    ax.set_yticks(np.arange(0, 1.00001, 0.1))
    ax.tick_params(labelsize=axesFontSize, size=8, width=2, which='major')
    ax.tick_params(size=4, width=1.5, which='minor')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    plt.xlabel('Success Rate (SR)', fontsize=labelFontSize)
    plt.ylabel('Hit Rate (H)', fontsize=labelFontSize)
    plt.subplots_adjust(bottom=0.35)
    maximizePlot()
    # Respace in between tick labesl
    N = 10
    plt.gca().margins(x=0, y=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([2 * t.get_window_extent().width for t in tl])
    m = 0.9  # inch margin
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(1.2 * plt.gcf().get_size_inches()[0], 1.2 * plt.gcf().get_size_inches()[1])
    # Save the figure
    plt.savefig(os.path.join(path_subcatch, "WEB", str(index) + "_Q_roebber.svg"), format='svg')
    plt.close("all")

    # KGE SPEEDOMETERS
    # Update the font before creating any plot objects
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams.update()
    # Calculate relative position in %
    mini = 0.3;
    maxi = 1.7
    Bp = max(min((B - mini) / (maxi - mini), 1), 0)
    # Extract values for spread
    Sp = max(min((S - mini) / (maxi - mini), 1), 0)
    # Start figure
    fig = plt.figure()
    # Define a subplot positioning grid
    gs = GridSpec(3, 3, figure=fig)
    # KGE
    ax1 = fig.add_subplot(gs[:2, :])
    gauge(ax1, labels=['$<$ 0.2', '0.2\n-\n0.4', '0.4\n-\n0.6', '0.6\n-\n0.8', '0.8\n-\n1.0'],
          colors=[KGEgrey, KGElighestblue, KGElighterblue, KGEdarkerblue, KGEdarkestblue], arrow=KGE, title='KGE',
          score=KGE, fontSize=titleFontSize)
    # Correlation
    ax2 = fig.add_subplot(gs[2, 0])
    gauge(ax2, labels=['$<$ 0.2', '0.2\n-\n0.4', '0.4\n-\n0.6', '0.6\n-\n0.8', '0.8\n-\n1.0'],
          colors=[KGEgrey, KGElighestblue, KGElighterblue, KGEdarkerblue, KGEdarkestblue], arrow=r,
          title='Correlation', score=r, fontSize=2+labelFontSize/2)
    # Bias ratio
    ax3 = fig.add_subplot(gs[2, 1])
    gauge(ax3,
          labels=['$<$ 0.5', '0.5\n-\n0.7', '0.7\n-\n0.9', '0.9\n-\n1.1', '1.1\n-\n1.3', '1.3\n-\n1.5', '$>$ 1.5'],
          colors=[KGEgrey, KGElighterblue, KGEdarkerblue, KGEdarkestblue, KGEdarkerblue, KGElighterblue, KGEgrey],
          arrow=Bp, title='Bias ratio', score=B, fontSize=2 + labelFontSize/2)
    # Spread ratio
    ax4 = fig.add_subplot(gs[2, 2])
    gauge(ax4,
          labels=['$<$ 0.5', '0.5\n-\n0.7', '0.7\n-\n0.9', '0.9\n-\n1.1', '1.1\n-\n1.3', '1.3\n-\n1.5', '$>$ 1.5'],
          colors=[KGEgrey, KGElighterblue, KGEdarkerblue, KGEdarkestblue, KGEdarkerblue, KGElighterblue, KGEgrey],
          arrow=Sp, title='Variability ratio', score=S, fontSize=2 + labelFontSize/2)
    maximizePlot()
    # Save the logarithmic scale figure
    plt.savefig(os.path.join(path_subcatch, "WEB", str(index) + "_Q_speedometers.svg"), format='svg')
    plt.close("all")



    # FIGURE OF CALIBRATION PERIOD TIME SERIES
    # Update the font before creating any plot objects
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams.update()
    fig = plt.figure()
    ax = plt.axes()
    # Qsim
    plt.plot([i.value * 1e-9 for i in Dates_Cal], Q_sim_Cal, color=purple, linewidth=1)
    # Qobs
    ax.fill_between([i.value * 1e-9 for i in Dates_Cal], np.zeros(len(Q_obs_Cal)), Q_obs_Cal, facecolor=lighterblue,
                    alpha=1.0, edgecolor=darkestblue, linewidth=0.5)
    # Return period
    mindate = Dates_Cal[0].value * 1e-9
    maxdate = Dates_Cal[-1].value * 1e-9
    # 1.5-year
    plt.hlines(rp1_5, color=green, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    ax.text(x=maxdate + 40, y=rp1_5, s='1.5-year', color=green, verticalalignment='center', fontsize=contFontSize)
    # 2-year
    plt.hlines(rp2, color=orange, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    ax.text(x=maxdate + 40, y=rp2, s='2-year', color=orange, verticalalignment='center', fontsize=contFontSize)
    # 5-year
    plt.hlines(rp5, color=red, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    ax.text(x=maxdate + 40, y=rp5, s='5-year', color=red, verticalalignment='center', fontsize=contFontSize)
    # # 10-year
    # plt.hlines(rp10, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # ax.text(x=maxdate + 1, y=rp10, s='10-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # 20-year
    if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp20:
      plt.hlines(rp20, color=magenta, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
      ax.text(x=maxdate + 40, y=rp20, s='20-year', color=magenta, verticalalignment='center', fontsize=contFontSize)
    # # 50-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp50:
    # 	plt.hlines(rp50, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp50, s='50-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # # 100-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp100:
    # 	plt.hlines(rp100, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp100, s='100-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # # 200-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp200:
    # 	plt.hlines(rp200, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp200, s='200-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # # 500-year
    # if max(max(Q_sim_Cal), max(Q_obs_Cal)) > rp500:
    # 	plt.hlines(rp500, color='black', linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
    # 	ax.text(x=maxdate + 10, y=rp500, s='500-year', color='black', verticalalignment='center', fontsize=contFontSize)
    # Title
    # ax.set_title('Discharge time series for calibration period', fontsize=titleFontSize)
    plt.ylabel(r'Discharge [m3/s]', fontsize=labelFontSize)  # ³
    # Activate major ticks at the beginning of each boreal season
    if period > 10:
      majorticks = [
        calendar.timegm(dt.datetime(k, j, 1).timetuple())
        for k in np.arange(Dates_Cal[0].year, Dates_Cal[-1].year + 1, 1)
        for j in [9]
        if dt.datetime(k, j, 1) >= Dates_Cal[0] and dt.datetime(k, j, 1) <= Dates_Cal[-1]
      ]
    else:
      majorticks = [
        calendar.timegm(dt.datetime(k, j, 1).timetuple())
        for k in np.arange(Dates_Cal[0].year, Dates_Cal[-1].year + 1, 1)
        for j in [3, 6, 9, 12]
        if dt.datetime(k, j, 1) >= Dates_Cal[0] and dt.datetime(k, j, 1) <= Dates_Cal[-1]
      ]
    ax.set_xticks(majorticks)
    # Rewrite labels
    locs, intlabels = plt.xticks()
    plt.setp(intlabels, rotation=70)
    labels = [dt.datetime.strftime(dt.datetime.fromtimestamp(i), "%b %Y") for i in majorticks]
    ax.set_xticklabels(labels)
    # Activate minor ticks every month
    minorticks = [
      calendar.timegm(dt.datetime(k, j + 1, 1).timetuple())
      for k in np.arange(Dates_Cal[0].year, Dates_Cal[-1].year + 1, 1)
      for j in range(12)
      if dt.datetime(k, j + 1, 1) >= Dates_Cal[0] and dt.datetime(k, j + 1, 1) <= Dates_Cal[-1]
    ]
    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(ticker.FixedLocator(minorticks))
    ax.tick_params(labelsize=axesFontSize, size=8, width=2, which='major')
    ax.tick_params(size=4, width=1.5, which='minor')
    # Maximize the window for optimal view
    maximizePlot()
    # DD better to always place the legend box to avoid random placement and risking not being able to read KGE NSE etc.
    leg = ax.legend(['Qsim', 'Qobs'], fancybox=True, framealpha=0.8, prop={'size': contFontSize*.75}, labelspacing=0.1,
                    loc='center', bbox_to_anchor=(0.5, -0.3))
    leg.get_frame().set_edgecolor('white')
    # linear scale
    logscale = False
    if logscale:
      plt.yscale(r'log')
      ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
      ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
      plt.ylim([1, 2 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    else:
      plt.yscale(r'linear')
      plt.ylim([1, 1.05 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # Save the linear scale figure
    plt.savefig(os.path.join(path_subcatch, "WEB", str(index) + "_Q_tseries_linear.svg"), format='svg')
    # # Log scale for clearer skill cover
    # logscale = True
    # if logscale:
    #   plt.yscale(r'log')
    #   ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
    #   ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    #   plt.ylim([1, 2 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # else:
    #   plt.ylim([1, 1.05 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # # Save the logarithmic scale figure
    # plt.savefig(os.path.join(path_subcatch, "WEB", str(index) + "_Q_tseries_log.svg"), format='svg')
    plt.close("all")



    # FIGURE OF MONTHLY CLIMATOLOGY FOR CALIBRATION PERIOD
    # Update the font before creating any plot objects
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams.update()
    months = np.array([9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # water year
    fig = plt.figure()
    # Get obs data from boxplot and delete it
    axd = plt.axes()
    Q_obs_box = axd.boxplot([Q_obs_monthData[i - 1] for i in months])
    plt.delaxes(axd)
    # create axes
    ax = plt.axes()
    # Make the obs wings plot
    (p1, p5, p25, median, p75, p95, p99) = wingplot(Q_obs_box, ax)
    # Plot the sim as boxplots
    Q_sim_box = ax.boxplot([np.ones((len(months))) * np.nan] + [Q_sim_monthData[i - 1] for i in months],
                           notch=True, sym='.', bootstrap=10000, showmeans=False, meanline=True, patch_artist=True,
                           widths=0.5)
    applyBoxplotTheme(Q_sim_box)
    # Esthetics
    # ax.set_title('Monthly discharge climatology in calibration period', fontsize=titleFontSize)
    ax.grid(b=True, axis='y')
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # horizontal axis
    plt.xlabel(r'Month', fontsize=labelFontSize)
    plt.xticks(range(1, 16), [""] + [month2string(m) for m in months])
    plt.xlim([1.5, 15.5])
    # vertical axis
    plt.ylabel(r'Discharge [m3/s]', fontsize=labelFontSize)
    ax.tick_params(labelsize=axesFontSize, size=8, width=2, which='major')
    ax.tick_params(size=4, width=1.5, which='minor')
    # Add manually-made legend
    genLegend()
    # Restore the correct active axes
    ax = fig.get_axes()[0]
    plt.sca(ax)
    # Maximize the window for optimal view
    maximizePlot()
    # linear scale
    logscale = False
    if logscale:
      plt.yscale(r'log')
      ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
      ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
      plt.ylim([1, 2 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    else:
      plt.yscale(r'linear')
      plt.ylim([1, 1.05 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # Save the linear scale figure
    plt.savefig(os.path.join(path_subcatch, "WEB", str(index) + "_Q_clim_linear.svg"), format='svg')
    # # Log scale for clearer skill cover
    # logscale = True
    # if logscale:
    #   plt.yscale(r'log')
    #   ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
    #   ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    #   plt.ylim([1, 2 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # else:
    #   plt.ylim([1, 1.05 * np.nanmax([Q_obs_Cal, Q_sim_Cal])])
    # # Save the logarithmic scale figure
    # plt.savefig(os.path.join(path_subcatch, "WEB", str(index) + "_Q_clim_log.svg"), format='svg')
    plt.close("all")



    # JSON file
    table_attributes = []
    # Add standard attributes from csv input file from HYDRO
    # for ii, i in enumerate(stationdata.columns):
    # 	table_attributes += [{i: row[ii]}]
    # First the crucial fields:
    table_attributes += [{'StationName': row['StationName']}]
    table_attributes += [{'Country': row['Country code']}]
    table_attributes += [{'Catchment': row['Catchment']}]
    if row["CAL_TYPE"].find("_24h") > -1:
      table_attributes += [{'Calibration Type': 'daily'}]
    else:
      table_attributes += [{'Calibration Type': '6-hourly'}]
    table_attributes += [{'Calibration Period': row['Cal_Start'] + " ~ " + row['Cal_End']}]
    table_attributes += [{'Calibration KGE': "{0:.2f}".format(KGE)}]
    table_attributes += [{'Calibration Correlation': "{0:.2f}".format(r)}]
    table_attributes += [{'Calibration Bias': "{0: .2f}".format(B)}]
    table_attributes += [{'Calibration Spread': "{0:.2f}".format(S)}]
    table_attributes += [{'Calibration NSE': "{0:.2f}".format(NSE)}]
    # Then the rest
    Qp = Q.dropna()
    if row["CAL_TYPE"].find("_24h") > -1:
      table_attributes += [{'Time steps available for calibration': "%i (%.2f years)" % (len(Qp), len(Qp) / 365.)}]
    else:
      table_attributes += [{'Time steps available for calibration': "%i (%.2f years)" % (len(Qp), len(Qp) / 4.0 / 365.)}]
    table_attributes += [{'Factors affecting runoff (Dam / Lake)': row['Dam/Lake']}]
    table_attributes += [{u'Qobs / Qsim mean [m³/s]': "{0:.0f}".format(np.nanmean(Q_obs_Cal)) + " / " + "{0:.0f}".format(np.nanmean(Q_sim_Cal))}]
    table_attributes += [{'Qsim return periods [m³/s]': "1.5-year: %i <br/> 2-year: %i <br/> 5-year: %i <br/> 20-year: %i" % (rp1_5, rp2, rp5, rp20)}]
    # \u00b3[m³/s]
    # \u000d
    # Add images
    images = {}
    # for i in range(1):
    # 	exec("images['image"+str(i+1)+"'] = {'title': 'image" + str(i+1) + "', 'order': "+str(i+1)+", 'name': 'image" + str(i+1) + ".svg'}")
    images['Q_speedometer'] = {'title': 'KGE decomposition - calibration period', 'order': 1, 'name': str(index) + '_Q_speedometers.svg', 'width': 1}
    # images['Q_roebber'] = {'title': 'Binary forecast scores', 'order': 2, 'name': str(index) + '_Q_roebber.svg', 'width': 1}
    images['Q_clim_linear'] = {'title': 'Monthly discharge climatology - calibration period', 'order': 2, 'name': str(index) + '_Q_clim_linear.svg', 'width': 1}
    # images['Q_clim_log'] = {'title': 'Monthly discharge climatology - calibration period - logarithmic scale', 'order': 4, 'name': str(index) + '_Q_clim_log.svg', 'width': 1}
    images['Q_tseries_linear'] = {'title': 'Discharge - calibration period', 'order': 3, 'name': str(index) + '_Q_tseries_linear.svg', 'width': 2}
    # images['Q_tseries_log'] = {'title': 'Discharge - calibration period - logarithmic scale', 'order': 6, 'name': str(index) + '_Q_tseries_log.svg', 'width': 2}
    with open(jsonFile, 'w') as f:
      f.write(json.dumps({'table_attributes': table_attributes, 'images': images}, indent=4))
      f.close()



    # # # TEXT OF CALIBRATION RESULTS
    # # texts = r"\huge \bfseries "+str(row["ObsID"])+": "+str(row["River"])+" at "+str(row["StationName"])
    # # texts = texts.replace("_","\_")
    # # texts_filtered = filter(lambda x: x in string.printable, texts)
    # # ax.text(0.5, 0.0, texts_filtered, verticalalignment='top',horizontalalignment='center', transform=ax.transAxes)
    # # plt.axis('on')
    # # statsum = r" " \
    # # 					+ "KGE$=" + "{0:.2f}".format(HydroStats.KGE(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
    # # 					+ "$, NSE$=" + "{0:.2f}".format(HydroStats.NS(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
    # # 					+ "$, $R=" + "{0:.2f}".format(HydroStats.correlation(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
    # # 					+ "$, $B=" + "{0:.2f}".format(HydroStats.pc_bias2(s=Q_sim_Cal, o=Q_obs_Cal, warmup=WarmupDays)) \
    # # 					+ "$ \%"
    # # ax.text(0.025, 0.93, statsum, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    #
    #
    print("finito il " + str(index))
    # sys.exit()

  newCsv = pd.DataFrame(data=newCsvDict)
  newCsv.to_csv('KGEs.csv', )


  # DD fastIO helper functions to either retrieve global variable from caller function (for multiple fastIO lisflodd runs)
  #    or create the queue and slaver writer process in case we run lisflood stand-alone once only
  def getParentGlobal(text):
    for i, frame in enumerate(inspect.stack()):
      print(str(i) + ": " + str(frame))
      if frame[1].find("add1") > -1 and frame[3].find("writenet") > -1:
        caller_frame = inspect.stack()[i + 1][0]
        break
    try:
      return caller_frame.f_locals.get(text)
    except NameError:
      for i, frame in enumerate(inspect.stack()):
        if frame[1].find("lisf1") > -1 and frame[3].find("main") > -1:
          caller_frame = inspect.stack()[i + 2][0]
          break
      try:
        return caller_frame.f_locals.get(text)
      except NameError:
        return inspect.stack()[1][0].f_locals.get(text)

  #
  #
  # def getWriterQueue():
  #   # DD fastIO slave process to process the netCDF writer queue
  #   q = mp.Queue()
  #
  #   def processWriterQueue():
  #     task = True
  #     print("START")
  #     while task is not None:
  #       try:
  #         task = q.get(block=True)
  #         print("reading task")
  #         print(task)
  #         processStations(*task)
  #       except:
  #         continue
  #     print("END")
  #   p = mp.Process(target=processWriterQueue)
  #   p.daemon = True
  #   p.start()
  #   return (p, q)



  # vprocessStations = np.vectorize(processStations)
  # dfOut = vprocessStations(stationdata)
  # dfOut = pandarallelize_dataframe(stationdata, vprocessStations, n_cores=6)

  # n_cores = 1
  # chunk = len(stationdata) // n_cores
  # pos = 0
  # ps = list(np.zeros((n_cores,)))
  # qs = list(np.zeros((n_cores,)))
  # results = []
  # for cpu in range(n_cores):
  # 	dfp = stationdata.iloc[pos:pos+chunk]
  # 	# # With ray
  # 	# results.append(processStations.remote(dfp))
  # 	# With a dedicated process, requires doubling virtuam mem every time
  # 	if ps[cpu] == 0  or qs[cpu] == 0:
  # 		ps[cpu], qs[cpu] = getWriterQueue()
  # 		qs[cpu].put(dfp)
  # 	pos += chunk
  #
  # # Wait for the tasks to complete and retrieve the results.
  # # With at least 4 cores, this will take 1 second.
  # results = ray.get(results)  # [0, 1, 2, 3]


  # stationdata.apply(processStations, axis=1)



  return



if __name__=="__main__":
    main(*sys.argv)


