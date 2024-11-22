# About this tool

The LISFLOOD calibration repository provides a collection of Python tools and libraries to calibrate the LISFLOOD hydrological model for a dedicated computational domain composed of multiple hydrological catchments.

This documentation explains how to use the scripts to calibrate the LISFLOOD hydrological model against streamflow observations for a specific domain using a genetic algorithm [DEAP](https://github.com/DEAP/deap).

The calibration tools are also available through a Python library called liscal, that you can install from the lisflood-calibration repository (https://github.com/ec-jrc/lisflood-calibration).
```python
import liscal
```

The calibration tools was created by Hylke Beck 2014 (JRC, Princeton) hylkeb@princeton.edu.

The submodule Hydrostats was created 2011 by Sat Kumar Tomer (modified by Hylke Beck).
 
Modified by Feyera Aga Hirpa in 2015 (JRC) feyera.hirpa@ouce.ox.ac.uk.
 
Modified by Valerio Lorini (valerio.lorini@ec.europa.eu) and Alfieri Lorenzo (lorenzo.alfieri@ec.europa.eu) in 2018.

The calibration tools were completely refactored by ECMWF (corentin.carton@ecmwf.int) for the EFAS 5.0 and GloFAS 4.0 releases in 2023.
