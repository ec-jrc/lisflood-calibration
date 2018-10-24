# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import HydroStats
import numpy as np
import datetime
import pdb
import time
import pandas
import math
import glob
import string
from ConfigParser import SafeConfigParser
from mpl_toolkits.basemap import Basemap

from matplotlib import rcParams
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams.update({'figure.autolayout': True})

t = time.time()


########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])
(drive, path) = os.path.splitdrive(iniFile)
(path, fil)  = os.path.split(path)
print ">> Reading settings file ("+fil+")..."

#file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

parser = SafeConfigParser()
parser.read(iniFile)

ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%m/%d/%Y")  # Start of forcing
ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%m/%d/%Y")  # Start of forcing

WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')

path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
path_result = parser.get('Path', 'Result')

Qtss_csv = parser.get('CSV', 'Qtss')
Qgis_csv = parser.get('CSV', 'Qgis')


########################################################################
#   Loop through catchments and perform calibration
########################################################################

print ">> Reading Qgis2.csv file..."
stationdata = pandas.read_csv(os.path.join(path_result,"Qgis3.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=True)

#CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)








X = stationdata_sorted['StationLon'].values
Y = stationdata_sorted['StationLat'].values
VALS = stationdata_sorted['KGE_val'].values
X = X[~np.isnan(VALS)]
Y = Y[~np.isnan(VALS)]
VALS = VALS[~np.isnan(VALS)]



fig = plt.figure(figsize=(11.7,8.3),dpi=100)
#plot1
m = fig.add_subplot(111)  
#m.set_position([0.15,0.30,0.35,0.46]) 
x1 = -20.
x2 = 50.
y1 = 10.
y2 = 70.
m = Basemap(resolution='i',projection='merc', llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,lat_ts=(x1+x2)/2) 
m.drawcountries(linewidth=0.5) 
m.drawcoastlines(linewidth=0.5)
m.drawrivers(linewidth=0.5) 


x,y = m(X,Y) 
m.scatter(x,y,s=60,c=VALS,edgecolors='k',marker='D')

plt.show()

pdb.set_trace()

#stationdata_sorted['KGE_cal'] = np.nan
#stationdata_sorted['KGE_val'] = np.nan
#stationdata_sorted['NSE_cal'] = np.nan
#stationdata_sorted['NSE_val'] = np.nan












# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:03:19 2012

@author: bissebe
"""
from pylab import *
from numpy import zeros, float32, loadtxt
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np 

stat=genfromtxt('e:/bissebe/calibration_Africa/validation/out.txt',skiprows=1,missing_values='NA',usecols=(1,0))
coord=genfromtxt('e:/bissebe/calibration_Africa/obs/coord_afr_stations.txt',usecols=(0,1))

stat1 = zeros([65,1], float32)
stat2 = zeros([65,1], float32)
stat3 = zeros([65,1], float32)
stat4 = zeros([65,1], float32)
coord1 = zeros([65,2], float32)
coord2 = zeros([65,2], float32)
coord3 = zeros([65,2], float32)
coord4 = zeros([65,2], float32)
for i in range(0,65):
    if stat[i,0] >= 0.75:
        stat1[i,0] = stat[i,0]
        coord1[i,0] = coord[i,0]
        coord1[i,1] = coord[i,1]
    elif stat[i,0] >= 0.5 and stat[i,0] < 0.75:
        stat2[i,0] = stat[i,0]
        coord2[i,0] = coord[i,0]
        coord2[i,1] = coord[i,1]
    elif stat[i,0] > 0.0 and stat[i,0] < 0.5:
        stat3[i,0] = stat[i,0]
        coord3[i,0] = coord[i,0]
        coord3[i,1] = coord[i,1]
    elif stat[i,0] <= 0.0:
        stat4[i,0] = stat[i,0]
        coord4[i,0] = coord[i,0]
        coord4[i,1] = coord[i,1]
            
stat1[stat1==0]=NaN
stat2[stat2==0]=NaN
stat3[stat3==0]=NaN
stat4[stat4==0]=NaN
coord1[coord1==0]=NaN
coord2[coord2==0]=NaN
coord3[coord3==0]=NaN
coord4[coord4==0]=NaN

fig = plt.figure(figsize=(11.7,8.3),dpi=100)
#plot1
m = fig.add_subplot(111)  
#m.set_position([0.15,0.30,0.35,0.46]) 
x1 = -20.
x2 = 50.
y1 = -35.
y2 = 20.
m = Basemap(resolution='i',projection='merc', llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,lat_ts=(x1+x2)/2) 
m.drawcountries(linewidth=0.5) 
m.drawcoastlines(linewidth=0.5)
m.drawrivers(linewidth=0.5) 
#m.bluemarble(scale=0.2) 
#m.drawparallels(np.arange(49.,53.,1.),labels=[1,0,0,0],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) 
# draw parallels 
#m.drawmeridians(np.arange(1.,9.,1.),labels=[0,0,0,1],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) 
# draw meridians   
lon1 = coord1[:,0]
lat1 = coord1[:,1]
lon2 = coord2[:,0]
lat2 = coord2[:,1]
lon3 = coord3[:,0]
lat3 = coord3[:,1]
lon4 = coord4[:,0]
lat4 = coord4[:,1]
# with x,y=m(lon,lat) we calculate the x,y positions 
x,y = m(lon1,lat1) 
m.scatter(x,y,facecolors='green',edgecolors='k',marker='D', s=60)
x,y = m(lon2,lat2)
m.scatter(x,y,facecolors='yellow',edgecolors='k',marker='D', s=60)
x,y = m(lon3,lat3)
m.scatter(x,y,facecolors='orange',edgecolors='k',marker='D', s=60)
x,y = m(lon4,lat4)
m.scatter(x,y,facecolors='red',edgecolors='k',marker='D', s=60)
#plot2
#m = fig.add_subplot(122)  
#m.set_position([0.52,0.30,0.38,0.46]) 
#x1 = 12.
#x2 = 36.
#y1 = -35.
#y2 = -20.
#m = Basemap(resolution='i',projection='merc', llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,lat_ts=(x1+x2)/2) 
#m.drawcountries(linewidth=0.5) 
#m.drawcoastlines(linewidth=0.5)
#m.drawrivers(linewidth=0.5)  
#m.drawparallels(np.arange(49.,53.,1.),labels=[1,0,0,0],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) 
# draw parallels 
#m.drawmeridians(np.arange(1.,9.,1.),labels=[0,0,0,1],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) 
# draw meridians   
# with x,y=m(lon,lat) we calculate the x,y positions 
#x,y = m(lon1,lat1) 
p1 = m.scatter(x,y,facecolors='green',edgecolors='k',marker='D', s=60)
#x,y = m(lon2,lat2)
p2 = m.scatter(x,y,facecolors='yellow',edgecolors='k',marker='D', s=60)
#x,y = m(lon3,lat3)
p3 = m.scatter(x,y,facecolors='orange',edgecolors='k',marker='D', s=60)
#x,y = m(lon4,lat4)
p4 = m.scatter(x,y,facecolors='red',edgecolors='k',marker='D', s=60)
#p1 = Rectangle((0, 0), 1, 1, fc="b")
#p2 = Rectangle((0, 0), 1, 1, fc="g")
#p3 = Rectangle((0, 0), 1, 1, fc="r")
#fig.legend((p1,p2,p3,p4),('> 0.75','0.5 - 0.75','0.0 - 0.5','< 0'), scatterpoints=1, bbox_to_anchor=(.41, .28), loc='center left')
fig.legend((p1,p2,p3,p4),('> 0.75','0.5 - 0.75','0.0 - 0.5','< 0'), scatterpoints=1, loc='center left')
plt.show() 
#title('validation 1998-2004')
#savefig('e:/bissebe/calibration_Africa/validation/validation_1998-2004_daily_4jutta.tiff',dpi=300)
