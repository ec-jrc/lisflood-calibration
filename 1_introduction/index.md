# About this tool

This is a collection of scripts to calibrate Lisflood model over a set of catchments.

This Readme text explains how to use the scripts with the LISFLOOD hydrological model against streamflow observations in an automated manner for multiple catchments.

The scripts loop through the catchments in ascending order of catchment area, calibrating LISFLOOD for each interstation region (i.e., the catchment area excluding the area of upstream catchments) using a genetic algorithm [DEAP](https://github.com/DEAP/deap).

The calibration tool was created by Hylke Beck 2014 (JRC, Princeton) hylkeb@princeton.edu
 
Modified by Feyera Aga Hirpa in 2015 (JRC) feyera.hirpa@ouce.ox.ac.uk
 
Modified by Valerio Lorini (valerio.lorini@ec.europa.eu) and Alfieri Lorenzo (lorenzo.alfieri@ec.europa.eu) in 2018

The submodule Hydrostats was created 2011 by Sat Kumar Tomer (modified by Hylke Beck)
