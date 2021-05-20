# -*- coding: utf-8 -*-
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm, ticker, rcParams, rc, image, get_backend
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
from ConfigParser import SafeConfigParser
import xarray as xr
import json
from collections import OrderedDict
import binaryForecastsSkill as bfs
import shapefile
import pprint as p



def modify(key,value,e):
    for i in range(len(e.records)):
        if(e.records[i][0]==key):
            e.records[i][2]=value
            break


def updateOGR():
	path = '/home/rikl/Dokumente/Python/shapefile/customer_points.shp'
	import osgeo.ogr, osgeo.osr  # we will need some packages
	from osgeo import ogr  # and one more for the creation of a new field
	spatialReference = osgeo.osr.SpatialReference()  # will create a spatial reference locally to tell the system what the reference will be
	spatialReference.ImportFromProj4(
		'+proj=utm +zone=48N +ellps=WGS84 +datum=WGS84 +units=m')  # here we define this reference to be utm Zone 48N with wgs84...
	driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')  # will select the driver foir our shp-file creation.
	shapeData = driver.CreateDataSource(path)  # so there we will store our data
	layer = shapeData.CreateLayer('customs', spatialReference,
																osgeo.ogr.wkbPoint)  # this will create a corresponding layer for our data with given spatial information.
	layer_defn = layer.GetLayerDefn()  # gets parameters of the current shapefile
	point = osgeo.ogr.Geometry(osgeo.ogr.wkbPoint)
	point.AddPoint(474595, 5429281)  # create a new point at given ccordinates
	featureIndex = 0  # this will be the first point in our dataset
	##now lets write this into our layer/shape file:
	feature = osgeo.ogr.Feature(layer_defn)
	feature.SetGeometry(point)
	feature.SetFID(featureIndex)
	layer.CreateFeature(feature)
	## lets add now a second point with different coordinates:
	point.AddPoint(474598, 5429281)
	featureIndex = 1
	feature = osgeo.ogr.Feature(layer_defn)
	feature.SetGeometry(point)
	feature.SetFID(featureIndex)
	layer.CreateFeature(feature)
	shapeData.Destroy()  # lets close the shapefile
	shapeData = ogr.Open(path, 1)
	layer = shapeData.GetLayer()  # get possible layers. was source.GetLayer
	layer_defn = layer.GetLayerDefn()
	field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(
		layer_defn.GetFieldCount())]  # store the field names as a list of stringsprint len(field_names)# so there should be just one at the moment called "FID"
	print
	len(field_names)  # so there should be just one at the moment called "FID"
	field_names  # will show you the current field names
	new_field = ogr.FieldDefn('HOMETOWN', ogr.OFTString)  # we will create a new field called Hometown as String
	layer.CreateField(new_field)  # self explaining
	new_field = ogr.FieldDefn('VISITS', ogr.OFTInteger)  # and a second field 'VISITS' stored as integer
	layer.CreateField(new_field)  # self explaining
	field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
	field_names  # WOOHAA!
	feature = layer.GetFeature(0)  # lets get the first feature (FID=='0')
	i = feature.GetFieldIndex("HOMETOWN")  # so iterate along the field-names and store it in iIndex
	feature.SetField(i, 'Chicago')  # exactly at this position I would like to write 'Chicago'
	layer.SetFeature(feature)  # now make the change permanent
	feature = layer.GetFeature(1)
	i = feature.GetFieldIndex("HOMETOWN")
	feature.SetField(i, 'Berlin')
	layer.SetFeature(feature)
	shapeData = None  # lets close the shape file again.



def updateQmeta():
	index = QgsSpatialIndex(layer.getFeatures())

	# -*- coding: utf-8 -*-

	"""
	***************************************************************************
	*                                                                         *
	*   This program is free software; you can redistribute it and/or modify  *
	*   it under the terms of the GNU General Public License as published by  *
	*   the Free Software Foundation; either version 2 of the License, or     *
	*   (at your option) any later version.                                   *
	*                                                                         *
	***************************************************************************
	"""

	from PyQt5.QtCore import QCoreApplication, QVariant
	from Qmeta.core import (QgsProcessing,
												 QgsFeatureSink,
												 QgsFeature,
												 QgsField,
												 QgsFields,
												 QgsProcessingException,
												 QgsProcessingAlgorithm,
												 QgsProcessingParameterFeatureSource,
												 QgsProcessingParameterFeatureSink,
												 QgsProcessingParameterField,
												 )
	import processing

	class DissolveProcessingAlgorithm(QgsProcessingAlgorithm):
		"""
    Dissolve algorithm that dissolves features based on selected
    attribute and summarizes the selected field by cumputing the
    sum of dissolved features.
    """
		INPUT = 'INPUT'
		OUTPUT = 'OUTPUT'
		DISSOLVE_FIELD = 'dissolve_field'
		SUM_FIELD = 'sum_field'

		def tr(self, string):
			"""
      Returns a translatable string with the self.tr() function.
      """
			return QCoreApplication.translate('Processing', string)

		def createInstance(self):
			return DissolveProcessingAlgorithm()

		def name(self):
			"""
      Returns the algorithm name, used for identifying the algorithm. This
      string should be fixed for the algorithm, and must not be localised.
      The name should be unique within each provider. Names should contain
      lowercase alphanumeric characters only and no spaces or other
      formatting characters.
      """
			return 'dissolve_with_sum'

		def displayName(self):
			"""
      Returns the translated algorithm name, which should be used for any
      user-visible display of the algorithm name.
      """
			return self.tr('Dissolve with Sum')

		def group(self):
			"""
      Returns the name of the group this algorithm belongs to. This string
      should be localised.
      """
			return self.tr('scripts')

		def groupId(self):
			"""
      Returns the unique ID of the group this algorithm belongs to. This
      string should be fixed for the algorithm, and must not be localised.
      The group id should be unique within each provider. Group id should
      contain lowercase alphanumeric characters only and no spaces or other
      formatting characters.
      """
			return 'scripts'

		def shortHelpString(self):
			"""
      Returns a localised short helper string for the algorithm. This string
      should provide a basic description about what the algorithm does and the
      parameters and outputs associated with it..
      """
			return self.tr("Dissolves selected features and creates and sums values of features that were dissolved")

		def initAlgorithm(self, config=None):
			"""
      Here we define the inputs and output of the algorithm, along
      with some other properties.
      """
			# We add the input vector features source. It can have any kind of
			# geometry.
			self.addParameter(
				QgsProcessingParameterFeatureSource(
					self.INPUT,
					self.tr('Input layer'),
					[QgsProcessing.TypeVectorAnyGeometry]
				)
			)
			self.addParameter(
				QgsProcessingParameterField(
					self.DISSOLVE_FIELD,
					'Choose Dissolve Field',
					'',
					self.INPUT))
			self.addParameter(
				QgsProcessingParameterField(
					self.SUM_FIELD,
					'Choose Sum Field',
					'',
					self.INPUT))
			# We add a feature sink in which to store our processed features (this
			# usually takes the form of a newly created vector layer when the
			# algorithm is run in QGIS).
			self.addParameter(
				QgsProcessingParameterFeatureSink(
					self.OUTPUT,
					self.tr('Output layer')
				)
			)

		def processAlgorithm(self, parameters, context, feedback):
			"""
      Here is where the processing itself takes place.
      """
			source = self.parameterAsSource(
				parameters,
				self.INPUT,
				context
			)
			dissolve_field = self.parameterAsString(
				parameters,
				self.DISSOLVE_FIELD,
				context)
			sum_field = self.parameterAsString(
				parameters,
				self.SUM_FIELD,
				context)

			fields = QgsFields()
			fields.append(QgsField(dissolve_field, QVariant.String))
			fields.append(QgsField('SUM_' + sum_field, QVariant.Double))

			(sink, dest_id) = self.parameterAsSink(
				parameters,
				self.OUTPUT,
				context, fields, source.wkbType(), source.sourceCrs())

			# Create a dictionary to hold the unique values from the
			# dissolve_field and the sum of the values from the sum_field
			feedback.pushInfo('Extracting unique values from dissolve_field and computing sum')
			features = source.getFeatures()
			unique_values = set(f[dissolve_field] for f in features)
			# Get Indices of dissolve field and sum field
			dissolveIdx = source.fields().indexFromName(dissolve_field)
			sumIdx = source.fields().indexFromName(sum_field)

			# Find all unique values for the given dissolve_field and
			# sum the corresponding values from the sum_field
			sum_unique_values = {}
			attrs = [{dissolve_field: f[dissolveIdx], sum_field: f[sumIdx]}
							 for f in source.getFeatures()]
			for unique_value in unique_values:
				val_list = [f_attr[sum_field]
										for f_attr in attrs if f_attr[dissolve_field] == unique_value]
				sum_unique_values[unique_value] = sum(val_list)

			# Running the processing dissolve algorithm
			feedback.pushInfo('Dissolving features')
			dissolved_layer = processing.run("native:dissolve", {
				'INPUT': parameters[self.INPUT],
				'FIELD': dissolve_field,
				'OUTPUT': 'memory:'
			}, context=context, feedback=feedback)['OUTPUT']

			# Read the dissolved layer and create output features
			for f in dissolved_layer.getFeatures():
				new_feature = QgsFeature()
				# Set geometry to dissolved geometry
				new_feature.setGeometry(f.geometry())
				# Set attributes from sum_unique_values dictionary that we had computed
				new_feature.setAttributes([f[dissolve_field], sum_unique_values[f[dissolve_field]]])
				sink.addFeature(new_feature, QgsFeatureSink.FastInsert)

			return {self.OUTPUT: dest_id}



def main(*args, **kwargs):

	rc('font',**{'family':'DejaVu Sans','serif':['Palatino']})
	rc('text', usetex=True) # DD need to add these latex directives to interpret things like /huge and /bfseries
	rcParams.update({'figure.autolayout': True})

	iniFile = os.path.normpath(sys.argv[1])
	(drive, path) = os.path.splitdrive(iniFile)
	(path, fil)  = os.path.split(path)
	print(">> Reading settings file ("+fil+")...")

	file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

	parser = SafeConfigParser()
	parser.read(iniFile)

	ObservationsStart = dt.datetime.strptime(parser.get('DEFAULT', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
	ObservationsEnd = dt.datetime.strptime(parser.get('DEFAULT', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
	ForcingStart = dt.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
	ForcingEnd = dt.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing

	WarmupDays = 0 #int(parser.get('DEFAULT', 'WarmupDays'))

	CatchmentDataPath = parser.get('Path','CatchmentDataPath')
	SubCatchmentPath = parser.get('Path','SubCatchmentPath')

	path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
	path_result = parser.get('Path', 'Result')

	Qtss_csv = parser.get('CSV', 'Qtss')
	Qmeta_csv = parser.get('CSV', 'Qmeta')
	calibrationFreq = parser.get('DEFAULT', 'calibrationFreq')

	########################################################################
	#   Loop through catchments
	########################################################################

	# Reading station data
	print(">> Reading Qmeta2.csv file...")
	stationdataFile = "Qmeta2.csv"
	if os.path.exists(stationdataFile.replace(".csv", ".npy")) and os.path.getsize(
		stationdataFile.replace(".csv", ".npy")) > 0:
		stationdata = pd.DataFrame(np.load(stationdataFile.replace(".csv", ".npy"), allow_pickle=True))
		stationdata.index = np.load(stationdataFile.replace(".csv", "_index.npy"), allow_pickle=True)
		stationdata.columns = np.load(stationdataFile.replace(".csv", "_columns.npy"), allow_pickle=True)
	else:
		stationdata = pd.read_csv(os.path.join(path_result, stationdataFile), sep=",", index_col=0)
		np.save(stationdataFile.replace(".csv", ".npy"), stationdata)
		np.save(stationdataFile.replace(".csv", "_index.npy"), stationdata.index)
		np.save(stationdataFile.replace(".csv", "_columns.npy"), stationdata.columns.values)

	CatchmentsToProcess = pd.read_csv(file_CatchmentsToProcess,sep=",",header=None)

	for index, row in stationdata.iterrows():
		Series = CatchmentsToProcess.ix[:,0]
		if len(Series[Series==row["ObsID"]]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
			continue

		path_subcatch = path_result #os.path.join(SubCatchmentPath,str(row['ObsID']))

		# # Skip already processed plots
		# jsonFile = os.path.join(path_subcatch, 'WEB', str(row['ObsID']) + '.json')
		# if os.path.exists(jsonFile) and os.path.getsize(jsonFile) > 0:
		# 	print("Skipping " + jsonFile)
		# 	continue

		# Make figures directory
		try:
			os.stat(os.path.join(path_subcatch,"SHP"))
		except:
			os.mkdir(os.path.join(path_subcatch,"SHP"))

		# # Delete contents of figures directory
		# for filename in glob.glob(os.path.join(path_subcatch,"WEB",'*.*')):
		# 	os.remove(filename)

		e = shapefile.Editor("/home/agus/Desktop/converterPy/provicias/provinces800.shp")
		p.pprint(e.records[1])
		p.pprint(e.records[3])
		p.pprint(e.records[7])
		modify("Antofagasta", 1, e)
		modify("Tocopilla", 23, e)
		modify("Huasco", 90, e)
		print("--------- AFTER EDIT ---------")
		p.pprint(e.records[1])
		p.pprint(e.records[3])
		p.pprint(e.records[7])
		e.save("/home/agus/Desktop/converterPy/provicias/provinces800.shp")

		print("finito il " + str(row['ObsID']))



if __name__=="__main__":
    main(*sys.argv)


