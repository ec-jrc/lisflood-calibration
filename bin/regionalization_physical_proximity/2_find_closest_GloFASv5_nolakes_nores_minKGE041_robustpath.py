from netCDF4 import Dataset
import numpy as np
import pandas
import geopy.distance
#import rioxarray as rxr
import os
import pcraster as pcr

# Specify the directory path

interstationID_Africa = np.load('headcatchments_Africa_nolakes_nores.npy')
interstationID_directories_Africa = np.load('headcatchments_Africa_directories_nolakes_nores.npy')
print(len(interstationID_Africa))
interstationID_Asia = np.load('headcatchments_Asia_nolakes_nores.npy')
interstationID_directories_Asia = np.load('headcatchments_Asia_directories_nolakes_nores.npy')
print(len(interstationID_Asia))
interstationID_NorthAmerica = np.load('headcatchments_NorthAmerica_nolakes_nores.npy')
interstationID_directories_NorthAmerica = np.load('headcatchments_NorthAmerica_directories_nolakes_nores.npy')
print(len(interstationID_NorthAmerica))
interstationID_SouthAmerica = np.load('headcatchments_SouthAmerica_nolakes_nores.npy')
interstationID_directories_SouthAmerica = np.load('headcatchments_SouthAmerica_directories_nolakes_nores.npy')
print(len(interstationID_SouthAmerica))
interstationID_CentralAmerica = np.load('headcatchments_CentralAmerica_nolakes_nores.npy')
interstationID_directories_CentralAmerica = np.load('headcatchments_CentralAmerica_directories_nolakes_nores.npy')
print(len(interstationID_CentralAmerica))
interstationID_Europe = np.load('headcatchments_Europe_nolakes_nores.npy')
interstationID_directories_Europe = np.load('headcatchments_Europe_directories_nolakes_nores.npy')
print(len(interstationID_Europe))
interstationID_Oceania = np.load('headcatchments_Oceania_nolakes_nores.npy')
interstationID_directories_Oceania = np.load('headcatchments_Oceania_directories_nolakes_nores.npy')
print(len(interstationID_Oceania))
print(interstationID_Oceania)

interstationID = []
interstationID= np.concatenate([interstationID_Oceania,interstationID_Africa,interstationID_Asia,interstationID_NorthAmerica,interstationID_SouthAmerica,interstationID_CentralAmerica,interstationID_Europe])
print('Tot headcatchments, no res, no lakes =', len(interstationID))


interstationID_directories = []
interstationID_directories=np.concatenate([interstationID_directories_Oceania, interstationID_directories_Africa, interstationID_directories_Asia, interstationID_directories_NorthAmerica, interstationID_directories_SouthAmerica, interstationID_directories_CentralAmerica, interstationID_directories_Europe])
print('Tot headcatchments, no res, no lakes =', len(interstationID_directories))
interstationID_directories_ALL=interstationID_directories

#####################################################################################################################################################

print('number of SELECTED interregions calibration: NO-lakes, NO-reservoirs, NO-inlets')
print(len(interstationID))


filename = "headcatchments_ALL_nolakes_nores.txt"
with open(filename, "w") as file:
    for ID in np.arange(len(interstationID)):
        # Write text to the file
        value=interstationID[ID]
        file.write(f"{value}\n")

np.save('headcatchments_ALL_directories_nolakes_nores',interstationID_directories) 
np.save('headcatchments_ALL_nolakes_nores',interstationID) 

headcatchments_ALL_nolakes_nores_minKGE041=[]
headcatchments_ALL_nolakes_nores_minKGE041_directories=[]

for catchmentID in interstationID:
   indicesT = np.where(np.char.find(interstationID_directories,str(catchmentID))!=-1)
   spT = np.char.split(interstationID_directories[indicesT],sep='/')
   indicesTarray=np.array(indicesT)
   count = -1
   for rowT in spT:
    count=count+1
    for elementT in rowT:
        if elementT == str(catchmentID):
           selectedT = rowT
           fullpathselectedT = interstationID_directories[indicesTarray[0,count]]
           print(fullpathselectedT)
           
   continent = selectedT[-3]
   basin = selectedT[-2]
   
   calibKGE_file = pandas.read_csv("<PATH>/calibration3arcmin_workflow/catchments/v5_allresultsfromLeonardo_23dec2025/"+continent[:]+'/'+basin[:]+'/'+str(catchmentID)+'/pHistoryWRanks.csv')
   calibKGE=calibKGE_file['Kling Gupta Efficiency'][0]
   if calibKGE>-0.41:
      headcatchments_ALL_nolakes_nores_minKGE041.append(catchmentID)
      headcatchments_ALL_nolakes_nores_minKGE041_directories.append(fullpathselectedT)

filename = "headcatchments_ALL_nolakes_nores_minKGE041.txt"
with open(filename, "w") as file:
    for ID in np.arange(len(headcatchments_ALL_nolakes_nores_minKGE041)):
        # Write text to the file
        value=headcatchments_ALL_nolakes_nores_minKGE041[ID]
        file.write(f"{value}\n")
        
filename = "headcatchments_ALL_nolakes_nores_minKGE041_directories.txt"
with open(filename, "w") as file:
    for ID in np.arange(len(headcatchments_ALL_nolakes_nores_minKGE041_directories)):
        # Write text to the file
        value=headcatchments_ALL_nolakes_nores_minKGE041_directories[ID]
        file.write(f"{value}\n")
        
print('number of SELECTED interregions calibration: NO-lakes, NO-reservoirs, NO-inlets, minKGEcalib -0.41')
print(len(headcatchments_ALL_nolakes_nores_minKGE041))


interstationID = headcatchments_ALL_nolakes_nores_minKGE041[0:6]
interstationID_directories = headcatchments_ALL_nolakes_nores_minKGE041_directories[0:6]

#####################################################################################################################################################
# TARGET catchments
'''
# global basins  = IN THIS EXPERIMENT, global basins = selected calibrated catchments (LEAVE ONE OUT CROSS VALIDATION EXPERIMENT)
# globalbasins = rxr.open_rasterio('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/globalbasins.tif') ---> Global uncalibrated basins for the application of the regionalization
nf2=Dataset(foldername+"interstation_regions_ALL.nc",'r',format='NETCDF4_CLASSIC') #
globalbasins2 = nf2.variables['interstation_regions'][:]
globalbasins = globalbasins2 * 0.0
selected_calibrated_catchments = interstationID  # LEAVE ONE OUT CROSS VALIDATION EXPERIMENT
for selected in selected_calibrated_catchments:
    mask = np.where( globalbasins2==selected, selected, 0.0)
    globalbasins = globalbasins + mask

globalbasinsID_all=np.ma.unique(globalbasins)
globalbasinsID_selected = globalbasinsID_all[globalbasinsID_all>0]
print(globalbasinsID_selected)



# ALL interregions calibration:
nf2=Dataset(foldername+"interstation_regions_ALL.nc",'r',format='NETCDF4_CLASSIC') #
interstation=nf2.variables['interstation_regions'][:]
'''
globalbasinsID_selected = interstationID

#### DONORS DATABASE
# find centroid of all the donors  + find the statistics of the predictors for each donor

centroid1=np.zeros((len(interstationID),2))-9999.0
AI_average=np.zeros((len(interstationID),1))-9999.0
AI_median=np.zeros((len(interstationID),1))-9999.0
MAP_average=np.zeros((len(interstationID),1))-9999.0
MAP_median=np.zeros((len(interstationID),1))-9999.0
SNOW_average=np.zeros((len(interstationID),1))-9999.0
SNOW_median=np.zeros((len(interstationID),1))-9999.0
POCC_average=np.zeros((len(interstationID),1))-9999.0
POCC_median=np.zeros((len(interstationID),1))-9999.0
CLAY_average=np.zeros((len(interstationID),1))-9999.0
CLAY_median=np.zeros((len(interstationID),1))-9999.0
GEOPERM_average=np.zeros((len(interstationID),1))-9999.0
GEOPERM_median=np.zeros((len(interstationID),1))-9999.0
SLOPE_average=np.zeros((len(interstationID),1))-9999.0
SLOPE_median=np.zeros((len(interstationID),1))-9999.0
ELEV_average=np.zeros((len(interstationID),1))-9999.0
ELEV_median=np.zeros((len(interstationID),1))-9999.0


for ID in np.arange(len(interstationID)):  # donors: headcatchments, no lakes, no res, no inlets, KGE> 0.41
     IDinterstation = interstationID[ID]
     print(IDinterstation)
     fullpathselectedX=[]
     indicesX = np.where(np.char.find(interstationID_directories,str(IDinterstation))!=-1)
     print(interstationID_directories)
     print(indicesX)
     spX = np.char.split(interstationID_directories[indicesX],sep='/')
     indicesXarray=np.array(indicesX)
     count = -1
     for rowX in spX:
      count=count+1
      for elementX in rowX:
          if elementX == str(IDinterstation):
             fullpathselectedX = interstationID_directories_ALL[indicesXarray[0,count]]
             print(fullpathselectedX)

     interstationmap = fullpathselectedX +'/maps/mask.map' ## make the path more robust!
     interstationmapread = pcr.readmap(interstationmap)
     interstation = pcr.pcr2numpy(interstationmapread, 0.0)
     # CENTROID
     colrow=np.where(interstation>0.0)
     points=np.array(colrow)
     x=-180.0 + points[1]*0.05   ### TO BE EDITED FOR THE EUROPEAN DOMAIN!!!
     y=90.0 - points[0]*0.05   ### TO BE EDITED FOR THE EUROPEAN DOMAIN!!!
     centroid1[ID,:] = (sum(y) / len(y), sum(x) / len(x))
     print(centroid1[ID,:])
     
     # CLIMATE
     # aridity index ## ALL FROM GloFASv4 FOLDER !!!!
     AIdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/AI.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     AI_catch=[]
     AI_catch=AIdata[points[0],points[1]]
     AI_average[ID]=np.mean(AI_catch)
     AI_median[ID]=np.percentile(AI_catch,50.0)
     # Mean Annual Precipitation
     MAPdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/sqrtMAP.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     MAP_catch=[]
     MAP_catch=MAPdata[points[0],points[1]]
     MAP_average[ID]=np.mean(MAP_catch)
     MAP_median[ID]=np.percentile(MAP_catch,50.0)     
     # POCC
     POCCdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/POCC.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     POCC_catch=[]
     POCC_catch=POCCdata[points[0],points[1]]
     POCC_average[ID]=np.mean(POCC_catch)
     POCC_median[ID]=np.percentile(POCC_catch,50.0)    
     # SNOW
     SNOWdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/FSNOW.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     SNOW_catch=[]
     SNOW_catch=SNOWdata[points[0],points[1]]
     SNOW_average[ID]=np.mean(SNOW_catch)
     SNOW_median[ID]=np.percentile(SNOW_catch,50.0) 
     # GEOLOGY and SOIL    
     # geopermeability
     GEOPERMdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/GEOPERM.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     GEOPERM_catch=[]
     GEOPERM_catch=GEOPERMdata[points[0],points[1]]
     GEOPERM_average[ID]=np.mean(GEOPERM_catch)
     GEOPERM_median[ID]=np.percentile(GEOPERM_catch,50.0)     
     # clay
     CLAYdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/CLAY.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     CLAY_catch=[]
     CLAY_catch=CLAYdata[points[0],points[1]]
     CLAY_average[ID]=np.mean(CLAY_catch)
     CLAY_median[ID]=np.percentile(CLAY_catch,50.0)
     # topography     
     # slope
     SLOPEdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/SLOPE.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     SLOPE_catch=[]
     SLOPE_catch=SLOPEdata[points[0],points[1]]
     SLOPE_average[ID]=np.mean(SLOPE_catch)
     SLOPE_median[ID]=np.percentile(SLOPE_catch,50.0)
     # elevation
     ELEVdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/ELEV.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     ELEV_catch=[]
     ELEV_catch=ELEVdata[points[0],points[1]]
     ELEV_average[ID]=np.mean(ELEV_catch)
     ELEV_median[ID]=np.percentile(ELEV_catch,50.0)    


#### TARGET CATCHMENTS DATABASE
# find centroid of all the GLOBAL uncalibrated BASINS  + find the statistics of the predictors for each GLOBAL uncalibrated BASINS  - in the leave one out cross validation experiment these are the same catchments as above
centroid1_globalbasinsID_selected=np.zeros((len(globalbasinsID_selected),2))-9999.0
AI_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
AI_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
MAP_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
MAP_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
SNOW_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
SNOW_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
POCC_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
POCC_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
CLAY_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
CLAY_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
GEOPERM_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
GEOPERM_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
SLOPE_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
SLOPE_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
ELEV_average_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0
ELEV_median_globalbasins=np.zeros((len(globalbasinsID_selected),1))-9999.0


'''
NOT NEEDED WHEN WE DO CROSS VALIDATION
for ID in np.arange(len(globalbasinsID_selected)):

     IDinterstation = interstationID[ID]
     print(IDinterstation)
     interstationmap = interstationID_directories[ID]+'/maps/mask.map'
     pcr.setclone(interstationmap)
     interstationmapread = pcr.readmap(interstationmap)
     interstation = pcr.pcr2numpy(interstationmapread, 0.0)
     globalbasins = interstation
     IDglobalbasins = globalbasinsID_selected[ID]
     print(IDglobalbasins)
     colrow=np.where(globalbasins>0.0)
     
     points=np.array(colrow)
     x=-180.0 + points[1]*0.05 ### TO BE EDITED FOR THE EUROPEAN DOMAIN!!!
     y=90.0 - points[0]*0.05 ### TO BE EDITED FOR THE EUROPEAN DOMAIN!!!
     centroid1_globalbasinsID_selected[ID,:] = (sum(y) / len(y), sum(x) / len(x))

     # aridity index
     AIdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/AI.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     AI_catch=[]
     AI_catch=AIdata[points[0],points[1]]
     AI_average_globalbasins[ID]=np.mean(AI_catch)
     AI_median_globalbasins[ID]=np.percentile(AI_catch,50.0)
     # Mean Annual Precipitation
     MAPdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/sqrtMAP.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     MAP_catch=[]
     MAP_catch=MAPdata[points[0],points[1]]
     MAP_average_globalbasins[ID]=np.mean(MAP_catch)
     MAP_median_globalbasins[ID]=np.percentile(MAP_catch,50.0)     
     # POCC
     POCCdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/POCC.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     POCC_catch=[]
     POCC_catch=POCCdata[points[0],points[1]]
     POCC_average_globalbasins[ID]=np.mean(POCC_catch)
     POCC_median_globalbasins[ID]=np.percentile(POCC_catch,50.0)    
     # SNOW
     SNOWdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/FSNOW.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     SNOW_catch=[]
     SNOW_catch=SNOWdata[points[0],points[1]]
     SNOW_average_globalbasins[ID]=np.mean(SNOW_catch)
     SNOW_median_globalbasins[ID]=np.percentile(SNOW_catch,50.0)      
     # GEOLOGY and SOIL    
     # geopermeability
     GEOPERMdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/GEOPERM.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     GEOPERM_catch=[]
     GEOPERM_catch=GEOPERMdata[points[0],points[1]]
     GEOPERM_average_globalbasins[ID]=np.mean(GEOPERM_catch)
     GEOPERM_median_globalbasins[ID]=np.percentile(GEOPERM_catch,50.0)  
     # clay
     CLAYdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/CLAY.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     CLAY_catch=[]
     CLAY_catch=CLAYdata[points[0],points[1]]
     CLAY_average_globalbasins[ID]=np.mean(CLAY_catch)
     CLAY_median_globalbasins[ID]=np.percentile(CLAY_catch,50.0)
     # topography     
     # slope
     SLOPEdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/SLOPE.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     SLOPE_catch=[]
     SLOPE_catch=SLOPEdata[points[0],points[1]]
     SLOPE_average_globalbasins[ID]=np.mean(SLOPE_catch)
     SLOPE_median_globalbasins[ID]=np.percentile(SLOPE_catch,50.0)     
     # elevation
     ELEVdata=np.load('<PATH>/DEFAULT_PAR_Global_REGIONALIZATON/ELEV.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
     ELEV_catch=[]
     ELEV_catch=ELEVdata[points[0],points[1]]
     ELEV_average_globalbasins[ID]=np.mean(ELEV_catch)
     ELEV_median_globalbasins[ID]=np.percentile(ELEV_catch,50.0)  
'''


centroid1_globalbasinsID_selected = centroid1
AI_average_globalbasin = AI_average
AI_median_globalbasins = AI_median
MAP_average_globalbasins = MAP_average
MAP_median_globalbasins = MAP_median
POCC_average_globalbasins = POCC_average
POCC_median_globalbasins = POCC_median   
SNOW_average_globalbasins = SNOW_average
SNOW_median_globalbasins = SNOW_median 
GEOPERM_average_globalbasins = GEOPERM_average
GEOPERM_median_globalbasins = GEOPERM_median 
CLAY_average_globalbasins = CLAY_average
CLAY_median_globalbasins = CLAY_median
SLOPE_average_globalbasins = SLOPE_average
SLOPE_median_globalbasins = SLOPE_median
ELEV_average_globalbasins = ELEV_average
ELEV_median_globalbasins = ELEV_median

 
# compute euclidean distance between donors and target catchments     
dist=np.zeros((len(interstationID),len(globalbasinsID_selected)))
count=0
for ID in np.arange(len(globalbasinsID_selected)):
     for ID2 in np.arange(len(interstationID)):
         count = count +1
         print(count, ' - we must reach ', len(globalbasinsID_selected)*len(interstationID))
         dist[ID2,ID]=geopy.distance.geodesic(centroid1[ID2,:], centroid1_globalbasinsID_selected[ID,:]).km
         
print(dist)

         

dist_AI=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_AI[0:len(interstationID),ID:(ID+1)] =np.abs(AI_median_globalbasins[ID]-AI_median)/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0))
        

dist_CLAY=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_CLAY[0:len(interstationID),ID:(ID+1)] =np.abs(CLAY_median_globalbasins[ID]-CLAY_median)/(np.nanpercentile(CLAYdata,95.0)-np.nanpercentile(CLAYdata,5.0))
        
      
print('SLOPE -----------------------------------------------------------------------------')  
dist_SLOPE=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_SLOPE[0:len(interstationID),ID:(ID+1)] =np.abs(SLOPE_median_globalbasins[ID]-SLOPE_median)/(np.nanpercentile(SLOPEdata,95.0)-np.nanpercentile(SLOPEdata,5.0))      
   

print('ELEV -----------------------------------------------------------------------------')  
dist_ELEV=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_ELEV[0:len(interstationID),ID:(ID+1)] =np.abs(ELEV_median_globalbasins[ID]-ELEV_median)/(np.nanpercentile(ELEVdata,95.0)-np.nanpercentile(ELEVdata,5.0))      
  

print('SNOW -----------------------------------------------------------------------------')  
dist_SNOW=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_SNOW[0:len(interstationID),ID:(ID+1)] =np.abs(SNOW_median_globalbasins[ID]-SNOW_median)/(np.nanpercentile(SNOWdata,95.0)-np.nanpercentile(SNOWdata,5.0))      

    
print('POCC -----------------------------------------------------------------------------')  
dist_POCC=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_POCC[0:len(interstationID),ID:(ID+1)] =np.abs(POCC_median_globalbasins[ID]-POCC_median)/(np.nanpercentile(POCCdata,95.0)-np.nanpercentile(POCCdata,5.0))      
    

print('MAP -----------------------------------------------------------------------------')  
dist_MAP=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_MAP[0:len(interstationID),ID:(ID+1)] =np.abs(MAP_median_globalbasins[ID]-MAP_median)/(np.nanpercentile(MAPdata,95.0)-np.nanpercentile(MAPdata,5.0))      
 
   
print('GEOPERM -----------------------------------------------------------------------------')  
dist_GEOPERM=np.zeros((len(interstationID),len(globalbasinsID_selected)))
for ID in np.arange(len(globalbasinsID_selected)):
    dist_GEOPERM[0:len(interstationID),ID:(ID+1)] =np.abs(GEOPERM_median_globalbasins[ID]-GEOPERM_median)/(np.nanpercentile(GEOPERMdata,95.0)-np.nanpercentile(GEOPERMdata,5.0))      

       
for num_predictor in [1]: #,2,4,7]:
    '''
    if num_predictor == 1:   
       print('geographic distance only')    
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) 
    if num_predictor == 2:   
       print('geographic distance + CLIMATE')        
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) + dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:]/(np.nanpercentile(MAPdata,95.0)-np.nanpercentile(MAPdata,5.0)) + dist_SNOW[:,:]/(np.nanpercentile(SNOWdata,95.0)-np.nanpercentile(SNOWdata,5.0)) + dist_POCC[:,:]/(np.nanpercentile(POCCdata,95.0)-np.nanpercentile(POCCdata,5.0)) 
    if num_predictor == 3:   
       print('geographic distance + CLIMATE + GEOLOGY')     
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) + dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:]/(np.nanpercentile(MAPdata,95.0)-np.nanpercentile(MAPdata,5.0)) + dist_SNOW[:,:]/(np.nanpercentile(SNOWdata,95.0)-np.nanpercentile(SNOWdata,5.0)) + dist_POCC[:,:]/(np.nanpercentile(POCCdata,95.0)-np.nanpercentile(POCCdata,5.0)) + dist_CLAY[:,:]/(np.nanpercentile(CLAYdata,95.0)-np.nanpercentile(CLAYdata,5.0)) + dist_GEOPERM[:,:]/(np.nanpercentile(GEOPERMdata,95.0)-np.nanpercentile(GEOPERMdata,5.0))
    if num_predictor == 4:   
       print('geographic distance + CLIMATE + GEOLOGY + TOPOGRAPHY')       
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) + dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:]/(np.nanpercentile(MAPdata,95.0)-np.nanpercentile(MAPdata,5.0)) + dist_SNOW[:,:]/(np.nanpercentile(SNOWdata,95.0)-np.nanpercentile(SNOWdata,5.0)) + dist_POCC[:,:]/(np.nanpercentile(POCCdata,95.0)-np.nanpercentile(POCCdata,5.0)) + dist_CLAY[:,:]/(np.nanpercentile(CLAYdata,95.0)-np.nanpercentile(CLAYdata,5.0)) + dist_GEOPERM[:,:]/(np.nanpercentile(GEOPERMdata,95.0)-np.nanpercentile(GEOPERMdata,5.0)) + dist_SLOPE[:,:]/(np.nanpercentile(SLOPEdata,95.0)-np.nanpercentile(SLOPEdata,5.0)) + dist_ELEV[:,:]/(np.nanpercentile(ELEVdata,95.0)-np.nanpercentile(ELEVdata,5.0))
    if num_predictor == 5:   
       print('CLIMATE + GEOLOGY + TOPOGRAPHY')       
       dist_sum = dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:]/(np.nanpercentile(MAPdata,95.0)-np.nanpercentile(MAPdata,5.0)) + dist_SNOW[:,:]/(np.nanpercentile(SNOWdata,95.0)-np.nanpercentile(SNOWdata,5.0)) + dist_POCC[:,:]/(np.nanpercentile(POCCdata,95.0)-np.nanpercentile(POCCdata,5.0)) + dist_CLAY[:,:]/(np.nanpercentile(CLAYdata,95.0)-np.nanpercentile(CLAYdata,5.0)) + dist_GEOPERM[:,:]/(np.nanpercentile(GEOPERMdata,95.0))-np.nanpercentile(GEOPERMdata,5.0) + dist_SLOPE[:,:]/(np.nanpercentile(SLOPEdata,95.0)-np.nanpercentile(SLOPEdata,5.0)) + dist_ELEV[:,:]/(np.nanpercentile(ELEVdata,95.0)-np.nanpercentile(ELEVdata,5.0)) 
    if num_predictor == 6:   
       print('CLIMATE + TOPOGRAPHY')       
       dist_sum = dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:]/(np.nanpercentile(MAPdata,95.0)-np.nanpercentile(MAPdata,5.0)) + dist_SNOW[:,:]/(np.nanpercentile(SNOWdata,95.0)-np.nanpercentile(SNOWdata,5.0)) + dist_POCC[:,:]/(np.nanpercentile(POCCdata,95.0)-np.nanpercentile(POCCdata,5.0)) + dist_SLOPE[:,:]/(np.nanpercentile(SLOPEdata,95.0)-np.nanpercentile(SLOPEdata,5.0)) + dist_ELEV[:,:]/(np.nanpercentile(ELEVdata,95.0)-np.nanpercentile(ELEVdata,5.0))  
    if num_predictor == 7:   
       print('CLIMATE')       
       dist_sum = dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:]/(np.nanpercentile(MAPdata,95.0)-np.nanpercentile(MAPdata,5.0)) + dist_SNOW[:,:]/(np.nanpercentile(SNOWdata,95.0)-np.nanpercentile(SNOWdata,5.0)) + dist_POCC[:,:]/(np.nanpercentile(POCCdata,95.0)-np.nanpercentile(POCCdata,5.0)) 
    ''' 
    if num_predictor == 1:   
       print('geographic distance only')    
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) 
       #print(dist_sum[:,:])
       #print(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0))
    if num_predictor == 2:   
       print('geographic distance + CLIMATE')        
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) + dist_AI[:,:] + dist_MAP[:,:] + dist_SNOW[:,:] + dist_POCC[:,:] 
    if num_predictor == 3:   
       print('geographic distance + CLIMATE + GEOLOGY')     
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) + dist_AI[:,:] + dist_MAP[:,:] + dist_SNOW[:,:] + dist_POCC[:,:] + dist_CLAY[:,:] + dist_GEOPERM[:,:]
    if num_predictor == 4:   
       print('geographic distance + CLIMATE + GEOLOGY + TOPOGRAPHY')       
       dist_sum = dist[:,:]/(np.nanpercentile(dist,95.0)-np.nanpercentile(dist,5.0)) + dist_AI[:,:] + dist_MAP[:,:] + dist_SNOW[:,:] + dist_POCC[:,:] + dist_CLAY[:,:] + dist_GEOPERM[:,:] + dist_SLOPE[:,:] + dist_ELEV[:,:]
    if num_predictor == 5:   
       print('CLIMATE + GEOLOGY + TOPOGRAPHY')       
       dist_sum = dist_AI[:,:] + dist_MAP[:,:] + dist_SNOW[:,:] + dist_POCC[:,:] + dist_CLAY[:,:] + dist_GEOPERM[:,:] + dist_SLOPE[:,:] + dist_ELEV[:,:] 
    if num_predictor == 6:   
       print('CLIMATE + TOPOGRAPHY')       
       dist_sum = dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:] + dist_SNOW[:,:] + dist_POCC[:,:] + dist_SLOPE[:,:] + dist_ELEV[:,:] 
    if num_predictor == 7:   
       print('CLIMATE')       
       dist_sum = dist_AI[:,:]/(np.nanpercentile(AIdata,95.0)-np.nanpercentile(AIdata,5.0)) + dist_MAP[:,:] + dist_SNOW[:,:] + dist_POCC[:,:]
     
    dfwrite = pandas.read_csv("Proximity_closest_noLakes_noRes_minKGE041_robustpath.csv")  
    dfwriteGeoDist = pandas.read_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv")
    dfwriteIDs = pandas.read_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv")

    for ID in np.arange(len(globalbasinsID_selected)):
    
      sorted=np.sort(dist_sum[:,ID])
      argsorted=np.argsort(dist_sum[:,ID])
      #print(globalbasinsID_selected[ID])
      #print(interstationID[argsorted[1]])
      
      
      if num_predictor == 1: 
       for id in np.arange(len(dfwrite['stationID'])):
          if (dfwrite['stationID'][id]==int(globalbasinsID_selected[ID])):
             dfwriteIDs['geography'][id] = interstationID[argsorted[1]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['geography2'][id] = interstationID[argsorted[2]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['geography3'][id] = interstationID[argsorted[3]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwrite['geography'][id] = sorted[1]
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)
             dfwrite['geography2'][id] = sorted[2]
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)
             dfwrite['geography3'][id] = sorted[3]
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)
             dfwriteGeoDist['geography'][id] = dist[argsorted[1],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['geography2'][id] = dist[argsorted[2],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['geography3'][id] = dist[argsorted[3],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False) 
             print('CHECK DISTANCE ---------------------------------------------------------------------------')
             print(dist[argsorted[0],ID])
             print(dist[argsorted[1],ID])
             print(dist[argsorted[2],ID])
             print(dist[argsorted[3],ID])
      if num_predictor == 2: 
       for id in np.arange(len(dfwrite['stationID'])):
           if (dfwrite['stationID'][id]==int(globalbasinsID_selected[ID])):
             print('writing into csv file....')
             print(id)
             dfwriteIDs['geogrclimate'][id] = interstationID[argsorted[1]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['geogrclimate2'][id] = interstationID[argsorted[2]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['geogrclimate3'][id] = interstationID[argsorted[3]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)             
             dfwrite['geogrclimate'][id] = sorted[1] 
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)
             dfwrite['geogrclimate2'][id] = sorted[2] 
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)
             dfwrite['geogrclimate3'][id] = sorted[3] 
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)    
             dfwriteGeoDist['geogrclimate'][id] = dist[argsorted[1],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['geogrclimate2'][id] = dist[argsorted[2],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['geogrclimate3'][id] = dist[argsorted[3],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)                       
      if num_predictor == 4: 
       for id in np.arange(len(dfwrite['stationID'])):
          if (dfwrite['stationID'][id]==int(globalbasinsID_selected[ID])):
             print('writing into csv file....')
             print(id)
             dfwriteIDs['ALL'][id] = interstationID[argsorted[1]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['ALL2'][id] = interstationID[argsorted[2]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['ALL3'][id] = interstationID[argsorted[3]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)                
             dfwrite['ALL'][id] = sorted[1]  
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)    
             dfwrite['ALL2'][id] = sorted[2]  
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False) 
             dfwrite['ALL3'][id] = sorted[3] 
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False) 
             dfwriteGeoDist['ALL'][id] = dist[argsorted[1],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['ALL2'][id] = dist[argsorted[2],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['ALL3'][id] = dist[argsorted[3],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)                                    
      if num_predictor == 7: 
       for id in np.arange(len(dfwrite['stationID'])):
          if (dfwrite['stationID'][id]==int(globalbasinsID_selected[ID])):
             print('writing into csv file....')
             print(id)
             dfwriteIDs['climate'][id] = interstationID[argsorted[1]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['climate2'][id] = interstationID[argsorted[2]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteIDs['climate3'][id] = interstationID[argsorted[3]]
             dfwriteIDs.to_csv("IDs_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)               
             dfwrite['climate'][id] = sorted[1]
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)          
             dfwrite['climate2'][id] = sorted[2]
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False)  
             dfwrite['climate3'][id] = sorted[3]
             dfwrite.to_csv("Proximity_closest_KGEminsu041_ALL.csv", index=False) 
             dfwriteGeoDist['climate'][id] = dist[argsorted[1],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['climate2'][id] = dist[argsorted[2],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)
             dfwriteGeoDist['climate3'][id] = dist[argsorted[3],ID]
             dfwriteGeoDist.to_csv("GeogrDist_closest_noLakes_noRes_minKGE041_robustpath.csv", index=False)                                     




