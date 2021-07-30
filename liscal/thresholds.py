import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from datetime import datetime

def hydrocdf(ranks, excfitter='Grinorten'):
  n = len(ranks)
  # empirical CDF option
  if excfitter == 'cdf':
    # From 1/n to 1, correct for discrete distributions
    # cdf = np.arange(1, n + 1) / n
    # From 0 to 1-1/n, to match Grinorten's distribution better
    cdf = np.arange(0, n) / n
    # From 0 to 1, is actually a continuous approximation
    # cdf = np.arange(0, n) / (n-1)
    return np.array(list(reversed(cdf)))
  # or Hydrological CDFs
  elif excfitter == 'Weibull':
    b = 0.0
  elif excfitter == 'Hazen':
    b = 0.5
  elif excfitter == 'Grinorten':
    b = 0.44
  else:
    raise Exception("Unknown hydrocdf exceedance fitter. Valid options are Weibull, Hazen and Grinorten (default)")
  return 1.0 - ((ranks - b) / (n + 1 - 2 * b))


def invp(p):
  if isinstance(p, float):
    return 1.0 / (1.0 - p)
  else:
    return 1.0 / (1.0 - p[p < 1])


def Tgumb(y, a=None, u=None):
  if a is None:
    a = np.sqrt(6) * np.nanstd(y, ddof=1) / np.pi
  if u is None:
    u = np.nanmean(y) - 0.5772 * a
  return -a * np.log(np.log(y / (y - 1))) + u


def fitgumb(x, y):
  a = np.sqrt(6) * np.nanstd(y, ddof=1) / np.pi
  u = np.nanmean(y) - 0.5772 * a
  popt, pcov = curve_fit(Tgumb, x, y, p0=[a, u])
  # Finds the standard deviations of given parameters alpha and u
  perr = np.sqrt(np.diag(pcov))
  return popt, perr


def get_periods(return_periods, tstep):
  if tstep == 'D':
    T = [i * 365.25 for i in return_periods]
  elif tstep == 'M':
    T = [i * 12 for i in return_periods]
  elif tstep == 'Y':
    T = return_periods
  return np.array(T)


def curve_gumbel_fit(return_periods, ranked_peaks, T):
  popt, perr = fitgumb(return_periods, ranked_peaks)
  thresholds = Tgumb(T, *popt)
  return thresholds


def compute_thresholds(qsim, tstep='D'):

    return_periods = [1.5, 2, 5, 20]
    rp_index = {'return_period': ['rl1.5', 'rl2', 'rl5', 'rl20']}

    qsim.index = [datetime.strptime(i, "%d/%m/%Y %H:%M") for i in qsim.index]

    # Agreggation of simulated streamflow
    qsim_peaks = qsim.resample(tstep, label="right", closed="right").max()

    # extract annual peak streamflows
    qsim_peaks_ranked = qsim_peaks.sort_values(ascending=False)
    qsim_peaks_ranked.index = [i+1 for i in range(len(qsim_peaks_ranked))]

    # construct cdf and transform into return period
    nonexceedence_probs = hydrocdf(qsim_peaks_ranked.index.values)
    return_periods_estimated  = invp(nonexceedence_probs)

    # add curve fit
    if len(return_periods_estimated) > 30:
      x = return_periods_estimated[:30]
      y = qsim_peaks_ranked.values[:30]
    else:
      x = return_periods_estimated
      y = qsim_peaks_ranked.values

    # fit only the extremest peaks
    T = get_periods(return_periods, tstep)
    thresholds_bestfit = curve_gumbel_fit(x, y, T)

    ds = xr.Dataset()
    ds['rl1.5'] = thresholds_bestfit[0]
    ds['rl2'] = thresholds_bestfit[1]
    ds['rl5'] = thresholds_bestfit[2]
    ds['rl20'] = thresholds_bestfit[3]

    return ds
