#!/bin/python

import math
from matplotlib import pyplot as plt


def contingency_table(threshold, Q):

    Qdf = Q.dropna(inplace=True)
    mask = Qdf >= threshold
    contingency_values = {}
    contingency_values['n'] = np.float(len(Qdf))
    contingency_values['a'] = np.float(sum(mask['Qsim'] * mask['Obs']))
    contingency_values['b'] = np.float(sum(mask['Qsim'] * ~mask['Obs']))
    contingency_values['c'] = np.float(sum(~mask['Qsim'] * mask['Obs']))
    contingency_values['d'] = np.float(sum(~mask['Qsim'] * ~mask['Obs']))
    return contingency_values

# Accuracy: Proportion Correct (Finley, 1884)
def PC(n, a, b, c, d):
    return (a+d) / n

# Accuracy: Threat Score (Gilbert, 1884)
def TS(n, a, b, c, d):
    try:
        return a / (a+b+c)
    except ZeroDivisionError:
        return float("nan")

# Accuracy: Critical Success Index (Gilbert, 1884)
def CSI(n, a, b, c, d):
    return TS(n, a, b, c, d)

# Accuracy: Odds Ratio (Stephenson, 2000)
def OR(n, a, b, c, d):
    try:
        return (a*d) / (b*c)
    except ZeroDivisionError:
        return float("nan")

# Bias
def B(n, a, b, c, d):
    try:
        return (a+b) / (a+c)
    except ZeroDivisionError:
        return float("nan")

# Reliability and Resolution: False Alarm Ratio (Barnes et al., 2009 explains history)
def FAR(n, a, b, c, d):
    try:
        return b / (a+b)
    except ZeroDivisionError:
        return float("nan")

# Reliability and Resolution: calibration-refinement counterpart of False Alarm Ratio
def FARCR(n, a, b, c, d):
    try:
        return d / (c+d)
    except ZeroDivisionError:
        return float("nan")

# Discrimination: Hit Rate
def HR(n, a, b, c, d):
    try:
        return a / (a+c)
    except ZeroDivisionError:
        return float("nan")

# Discrimination: Probability Of Detection, true-positive fraction, sensitivity
def POD(n, a, b, c, d):
    return HR(n, a, b, c, d)

# Discrimination: False Alarm Rate, false-positive fraction, specificity
def F(n, a, b, c, d):
    try:
        return b / (b+d)
    except ZeroDivisionError:
        return nan

# Discrimination: Percentage Correct Rejections
def PCR(n, a, b, c, d):
    try:
        return d / (b+d)
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Heidke Skill Score (Doolittle, 1888)
def HSS(n, a, b, c, d):
    try:
        return 2 * (a*d - b*c) / ((a+c)*(c+d) + (a+b)*(b+d))
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Peirce Skill Score (Peirce, 1884)
def PSS(n, a, b, c, d):
    try:
        return ((a*d)-(b*c)) / ((a+c)*(b+d))
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Hanssen-Kuipers Discriminant (Hanssen & Kuipers, 1965)
def HKD(n, a, b, c, d):
    return PSS(n, a, b, c, d)

# Skill Score: Kuipers' Performance Index (Murphy and Daan, 1985)
def KPI(n, a, b, c, d):
    return PSS(n, a, b, c, d)

# Skill Score: True Skill Statistic (Fleuck, 1987)
def TSS(n, a, b, c, d):
    return PSS(n, a, b, c, d)

# Skill Score: Clayton Skill Score (Clayton, 1927, 1934)
def CSS(n, a, b, c, d):
    try:
        return ((a*d)-(b*c)) / ((a+b)*(c+d))
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Gilbert Skill Score (Gilbert, 1884)
def GSS(n, a, b, c, d):
    try:
        a_ref = (a+b)*(a+c)/n
        return (a-a_ref) / (a-a_ref+b+c)
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Ratio Of Success (Gilbert, 1884)
def ROS(n, a, b, c, d):
    return GSS(n, a, b, c, d)

# Skill Score: Equitable Threat Score
def ETS(n, a, b, c, d):
    return GSS(n, a, b, c, d)

# Skill score: Extreme Dependency Score (Coles et al., 1999, Stephenson et al., 2008a)
def EDS(n, a, b, c, d):
    try:
        return 2*math.log((a+c)/n) / math.log(a/n) - 1
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Symmetric Extreme Dependency Score (Hogan et al, 2009)
def SEDS(n, a, b, c, d):
    try:
        return (math.log((a+b)/n) + math.log((a+c)/n)) / math.log(a/n) - 1
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Yule's Q (Yule, 1900; Woodcock, 1976)
def Q(n, a, b, c, d):
    try:
        return (a*d - b*c) / (a*d + b*c)
    except ZeroDivisionError:
        return float("nan")

# Skill Score: Odds Ratio Skill Score (Stephenson, 2000)
def ORSS(n, a, b, c, d):
    return Q(n, a, b, c, d)

# Base Rate or sample climatological relative frequency
def BR(n, a, b, c, d):
    return (a+c) / n

# Calibration Refinement
def CR(n, a, b, c, d):
    rateeturn (a+b) / n

# Summary: likelihood-base rate factorization (Pepe, 2003)
def BRHFSummary(n, a, b, c, d):
    print("BRHF Summary:")
    print("  BR: " + str(BR(n, a, b, c, d)))
    print("  HR: " + str(HR(n, a, b, c, d)))
    print("  F: " + str(F(n, a, b, c, d)))

# Summary: calibration-refinement factorization
def CRFSummary(n, a, b, c, d):
    print("CRF Summary:")
    print("  FAR: " + str(FAR(n, a, b, c, d)))
    print("  FARCR: " + str(FARCR(n, a, b, c, d)))
    print("  CR: " + str(CR(n, a, b, c, d)))

# Summary: BHF (Stephenson, 2000)
def BHFSummary(n, a, b, c, d):
    print("BHF Summary:")
    print("  B: " + str(B(n, a, b, c, d)))
    print("  HR: " + str(HR(n, a, b, c, d)))
    print("  F: " + str(F(n, a, b, c, d)))

# Summary: BOP (Stephenson, 2000)
def BOPSummary(n, a, b, c, d):
    print("BOP Summary:")
    print("  B: " + str(B(n, a, b, c, d)))
    print("  OR: " + str(OR(n, a, b, c, d)))
    print("  PSS: " + str(PSS(n, a, b, c, d)))

# Summary: BROP (Stephenson, 2000)
def BROPSummary(n, a, b, c, d):
    print("BROP Summary:")
    print("  BR: " + str(BR(n, a, b, c, d)))
    print("  OR: " + str(OR(n, a, b, c, d)))
    print("  PSS: " + str(PSS(n, a, b, c, d)))

# Summary: HBBR (Stephenson et al., 2008a; Brill, 2009)
def HBBRSummary(n, a, b, c, d):
    print("HBBR Summary:")
    print("  HR: " + str(HR(n, a, b, c, d)))
    print("  B: " + str(B(n, a, b, c, d)))
    print("  BR: " + str(BR(n, a, b, c, d)))

# Diagram: FARH (Roebber, 2009)
def FARHDiagram(n, a, b, c, d):
     print("FARH Diagram:")
     plt.figure()
     plt.plot(1-FAR(n, a, b, c, d), HR(n, a, b, c, d), 'go--', linewidth=2, markersize=12)
     plt.plot(B(n, a, b, c, d), TS(n, a, b, c, d), 'ro--', linewidth=2, markersize=12)
     # contours of TS
     # contours of B
     plt.show()

# Summary: Equitable Score (Gandin and Murphy, 1992; Hogan et al., 2010)
def ESSummary(n, a, b, c, d):
    print("ES Summary:")
    print("  PSS: " + str(PSS(n, a, b, c, d)))
    print("  HSS: " + str(HSS(n, a, b, c, d)))

# Summary: Asymptotically Equitable Score (Hogan et al., 2010)
def AESSummary(n, a, b, c, d):
    print("AES Summary:")
    print("  GSS: " + str(GSS(n, a, b, c, d)))
    print("  SEDS: " + str(SEDS(n, a, b, c, d)))
    print("  Q: " + str(Q(n, a, b, c, d)))
