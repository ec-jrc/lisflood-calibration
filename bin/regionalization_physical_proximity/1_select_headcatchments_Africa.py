import numpy as np
from netCDF4 import Dataset
import pcraster as pcr
import os
import gc



# Specify the directory path
root_dir = '/BGFS/DISASTER/russcar/cal_workflow_2025/catchments/Africa/'

headcatchments_Africa_nolakes_nores =[]
headcatchments_Africa_nolakes_nores_directories=[]
headcatchments_Africa =[]
headcatchments_Africa_directories=[]
ATTENTION_INCOMPLETE_DIRECTORIES=[]
stack = [(root_dir, 0)]

while stack:
  current_dir, depth = stack.pop()
  

  if depth ==2:
 
    lakes =[]
    reservoirs =[]
    maskmap_numpy_map = []

    if current_dir.find('/_')==-1:
    #if not isinstance('/_', current_dir):                 
       if os.path.exists(current_dir+"/maps/20250307_reservoirs_Global_03min.nc"):           
         nf2=Dataset(current_dir+"/maps/20250307_reservoirs_Global_03min.nc",'r',format='NETCDF4_CLASSIC') #
         reservoirs = nf2.variables['Band1'][:,:]
         reservoirs = np.where(reservoirs>0,1,0) 
         nf3=Dataset(current_dir+"/maps/20250307_lakes_Global_03min.nc",'r',format='NETCDF4_CLASSIC') #                
         lakes = nf3.variables['Band1'][:,:]
         lakes = np.where(lakes>0,1,0)   ## apply mask map  
         
         pcr.setclone(current_dir+"/maps/masksmall.map")
         maskmap_pcr_map = pcr.boolean(current_dir+"/maps/masksmall.map")
         maskmap_numpy_map = pcr.pcr2numpy(maskmap_pcr_map, 0.0)     
         lakes =np.where(maskmap_numpy_map>0.0,lakes,0.0)
         reservoirs = np.where(maskmap_numpy_map>0.0,reservoirs,0.0)          
          
         pcr.setclone(current_dir+"/inflow/"+'inflow.map')
         inflow_pcr_map = pcr.boolean(current_dir+"/inflow/"+'inflow.map')
         inflow_numpy_map = pcr.pcr2numpy(inflow_pcr_map, 0.0)
         inflow_all_zeros = np.sum(inflow_numpy_map)
 
         if inflow_all_zeros==0:           
            folder_name_basename = os.path.basename(os.path.normpath(current_dir))
            if folder_name_basename[0] != '_':
                if folder_name_basename[-1] != 'D':
                   headcatchments_Africa.append(folder_name_basename)
                   headcatchments_Africa_directories.append(current_dir)
                   np.save('headcatchments_Africa', headcatchments_Africa)
                   print(f'Level {depth}, Directory: {current_dir} HEADCATCHMENT')
                   np.save('headcatchments_Africa_directories',headcatchments_Africa_directories)                                      
                   if np.sum(lakes+reservoirs) == 0.0:  
                       headcatchments_Africa_nolakes_nores.append(folder_name_basename)
                       headcatchments_Africa_nolakes_nores_directories.append(current_dir)
                       np.save('headcatchments_Africa_nolakes_nores', headcatchments_Africa_nolakes_nores)
                       print(f'Level {depth}, Directory: {current_dir} NO lakes No res')
                       np.save('headcatchments_Africa_directories_nolakes_nores',headcatchments_Africa_nolakes_nores_directories)        
       else:
         print('ATTENTION!! Reservoir file for catchment ',current_dir,' does NOT exist!!') 
         ATTENTION_INCOMPLETE_DIRECTORIES.append(current_dir) 
         # Continue if the current depth is less than max_depth
  if depth < 2:
        try:
            for entry in os.listdir(current_dir):
                full_path = os.path.join(current_dir, entry)
                if os.path.isdir(full_path):
                    # Add subdirectories to the stack with incremented depth
                    stack.append((full_path, depth + 1))
                else:
                    print(f' ***************************************    File: {full_path}')
        except PermissionError:
            print(f"Permission denied: {current_dir}")



filename = "headcatchments_Africa.txt"
with open(filename, "w") as file:
    for ID in np.arange(len(headcatchments_Africa)):
        # Write text to the file
        value=headcatchments_Africa[ID]
        file.write(f"{value}\n")
        
filename_nolakes_nores = "headcatchments_Africa_nolakes_nores.txt"
with open(filename_nolakes_nores, "w") as file:
    for ID in np.arange(len(headcatchments_Africa_nolakes_nores)):
        # Write text to the file
        value=headcatchments_Africa_nolakes_nores[ID]
        file.write(f"{value}\n")        


filename = "ATTENTION_INCOMPLETE_DIRECTORIES_Africa.txt"
with open(filename, "w") as file:
    for ID in np.arange(len(ATTENTION_INCOMPLETE_DIRECTORIES)):
        # Write text to the file
        value=ATTENTION_INCOMPLETE_DIRECTORIES[ID]
        file.write(f"{value}\n")

