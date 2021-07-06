#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:36:37 2011
@ author:                  Sat Kumar Tomer (modified by Hylke Beck)
@ author's webpage:        http://civil.iisc.ernet.in/~satkumar/
@ author's email id:       satkumartomer@gmail.com
@ author's website:        www.ambhas.com

A library with Python functions for calculating several objective functions commonly used in hydrological sciences.
Inputs consist of two equal sized arrays representing modeled and observed time series, and an integer specifying the
number of days to ignore in the beginning of the time series.

Example usage:
correlation = HydroStats.correlation(s=Qsim,o=Qobs,365)

Functions:
    RSR :     RMSE-observations standard deviation ratio
    br :      bias ratio
    pc_bias : percentage bias
    pc_bias2: percentage bias 2
    apb :     absolute percent bias
    apb2 :    absolute percent bias 2
    rmse :    root mean square error
    mae :     mean absolute error
    bias :    bias
    NS :      Nash Sutcliffe Coefficient
    NSlog :   Nash Sutcliffe Coefficient from log-transformed data
    correlation: correlation
    KGE:      Kling Gupta Efficiency
    vr :      variability ratio
    
"""

# import required modules
import numpy as np
from random import randrange


def filter_nan(s,o):
    """
    this functions removed the data  from simulated and observed data
    whereever the observed data contains nan
    
    this is used by all other functions, otherwise they will produce nan as 
    output
    """
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]

    #mask = ~np.isnan(s) & ~np.isnan(o)
    #o_nonan = o[mask]
    #s_nonan = s[mask]

    #return o_nonan,s_nonan
    return data[:,0],data[:,1]



def RSR(s,o,spinup):
    """
    RMSE-observations standard deviation ratio
    input:
        s: simulated
        o: observed
    output:
        RSR: RMSE-observations standard deviation ratio
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    RMSE = np.sqrt(np.sum((s-o) ** 2))
    STDEV_obs = np.sqrt(np.sum((o-np.mean(o)) ** 2))
    return RMSE/STDEV_obs

def br(s,o,spinup):
    """
    Bias ratio
    input:
        s: simulated
        o: observed
    output:
        br: bias ratio
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return 1 - abs(np.mean(s)/np.mean(o) - 1)

def pc_bias(s,o,spinup):
    """
    Percent Bias
    input:
        s: simulated
        o: observed
    output:
        pc_bias: percent bias
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return 100.0*sum(s-o)/sum(o)

def pc_bias2(s,o,spinup):
    """
    Percent Bias 2
    input:
        s: simulated
        o: observed
    output:
        apb2: absolute percent bias 2
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return 100*(np.mean(s)-np.mean(o))/np.mean(o)

def apb(s,o,spinup):
    """
    Absolute Percent Bias
    input:
        s: simulated
        o: observed
    output:
        apb: absolute percent bias
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return 100.0*sum(abs(s-o))/sum(o)

def apb2(s,o,spinup):
    """
    Absolute Percent Bias 2
    input:
        s: simulated
        o: observed
    output:
        apb2: absolute percent bias 2
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return 100*abs(np.mean(s)-np.mean(o))/np.mean(o)

def rmse(s,o,spinup):
    """
    Root Mean Squared Error
    input:
        s: simulated
        o: observed
    output:
        rmses: root mean squared error
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return np.sqrt(np.mean((s-o)**2))

# DD Total sum absolute error
def sae(s, o, spinup):
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return np.sum(abs(s-o))

def mae(s,o,spinup):
    """
    Mean Absolute Error
    input:
        s: simulated
        o: observed
    output:
        maes: mean absolute error
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return np.mean(abs(s-o))

# DD try using the MAE as objective function
def maeSkill(s, o, spinup, lowFlowPercentileThreshold=0.0, usePeaksOnly=False):
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    if lowFlowPercentileThreshold > 0:
        # DD Construct CDF of observation
        olen = len(o)
        osorted = np.sort(o)
        ocdf = np.array(xrange(olen)) / float(olen)
        # Interpolate to the requested percentile
        othr = np.interp(lowFlowPercentileThreshold,ocdf,osorted)
        if usePeaksOnly:
            # Filter out the low flow completely
            s = s[o > othr]
            o = o[o > othr]
    return 1.0 - mae(s, o, spinup=-1)

def bias(s,o,spinup):
    """
    Bias
    input:
        s: simulated
        o: observed
    output:
        bias: bias
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return np.mean(s-o)

def NS(s,o,spinup):
    """
    Nash-Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        NS: Nash-Sutcliffe efficient coefficient
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/(sum((o-np.mean(o))**2)+1e-20)

def NSlog(s,o,spinup):
    """
    Nash-Sutcliffe efficiency coefficient from log-transformed data
    input:
        s: simulated
        o: observed
    output:
        NSlog: Nash-Sutcliffe efficient coefficient from log-transformed data
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    s = np.log(s)
    o = np.log(o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)

def correlation(s,o,spinup):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o, s)[0,1]
        
    return corr


def index_agreement(s,o,spinup):
    """
	index of agreement
	input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    ia = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    return ia

def KGE(s,o,spinup):
    """
	Modified Kling Gupta Efficiency (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    B = np.mean(s) / np.mean(o)
    y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    r = np.corrcoef(o, s)[0,1]

    KGE = 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)

    return KGE

def fKGE(s, o, spinup, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False):
    """
	Filtered Kling Gupta Efficiency (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
	input:
        s: simulated
        o: observed
        weightedLogWeight: weight to give to the KGE of log(time series) as this potentially improves calibration of low flows
        lowFlowPercentileThreshold: percentile defining low flow threshold, ranged from 0~1. Inactive when 0.
        usePeaksOnly: Requires lowFlowPercentileThreshold > 0
                      True = only include the peaks in the KGE, completely ignoring the low flow
                      False = calculate signal ratio on low flow only (flat part of discharge) and noise ratio on high flow only (discharge peaks)
    output:
        KGE: Kling Gupta Efficiency
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o) # This will also flatten the arrays

    r = np.corrcoef(o, s)[0,1]
    if lowFlowPercentileThreshold > 0:
        # Don't use the logKGE together with the filtered flow version
        weightedLogWeight = 0.0
        # DD Construct CDF of observation
        olen = len(o)
        osorted = np.sort(o)
        ocdf = np.array(range(olen)) / float(olen)
        # Interpolate to the requested percentile
        othr = np.interp(lowFlowPercentileThreshold,ocdf,osorted)
        if usePeaksOnly:
            # Filter out the low flow completely
            B = np.mean(s[o > othr]) / np.mean(o[o > othr])
            y = (np.std(s[o > othr]) / np.mean(s[o > othr])) / (np.std(o[o > othr]) / np.mean(o[o > othr]))
        else:
            # Calculate signal ratio on low flow (flat part) and noise ratio on high flow (peaks)
            B = np.mean(s[o <= othr]) / np.mean(o[o <= othr])
            y = (np.std(s[o > othr]) / np.mean(s[o > othr])) / (np.std(o[o > othr]) / np.mean(o[o > othr]))
    else:
        B = np.mean(s) / np.mean(o)
        y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
        se = sae(s, o, spinup=0)
    aKGE = 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)
    if aKGE < -100:
        aKGE = -100
    if np.isnan(aKGE):
        print("WARNING: nan KGE found")
        print("r = " + str(r))
        print("B = " + str(B))
        print("y = " + str(y))
        print("sqrt = " + str((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2))
        aKGE = -100.0

    # DD little trick to give more sensitivity to low flows
    if weightedLogWeight > 0:
        sl = np.log(s)
        ol = np.log(o)
        Bl = np.mean(sl) / np.mean(ol)
        yl = (np.std(sl) / np.mean(sl)) / (np.std(ol) / np.mean(ol))
        rl = np.corrcoef(ol, sl)[0,1]
        KGEl = 1 - np.sqrt((rl - 1) ** 2 + (Bl - 1) ** 2 + (yl - 1) ** 2)
        aKGE = (((1 - weightedLogWeight) * aKGE)**6.0 + (weightedLogWeight * KGEl)**6.0)**(1.0/6.0)
        r = (((1 - weightedLogWeight) * r)**6.0 + (weightedLogWeight * rl)**6.0)**(1.0/6.0)
        B = (((1 - weightedLogWeight) * B)**6.0 + (weightedLogWeight * Bl)**6.0)**(1.0/6.0)
        y = (((1 - weightedLogWeight) * y)**6.0 + (weightedLogWeight * yl)**6.0)**(1.0/6.0)

    return (aKGE, r, B, y, se)

def vr(s,o,spinup):
    """
	Variability ratio
	input:
        s: simulated
        o: observed
    output:
        vr: variability ratio
    """
    s = s[spinup+1:]
    o = o[spinup+1:]
    s,o = filter_nan(s,o)
    return 1 - abs((np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o)) - 1)