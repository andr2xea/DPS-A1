# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:55:55 2020

@author: Marija
"""
import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

# initializing SparkContext and SparkSession

sc = SparkContext(appName="DPS_A1", master='local[*,4]')
sc.setLocalProperty("spark.scheduler.pool", "pool1")
print(sc.pythonVer)
print (sc.master)

ss = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "4g") \
    .appName('DPS_A1') \
    .getOrCreate()
    
# loading training data

train = sc.textFile("numerai_training_data.csv")

# checking number of columns

print("The data has {} columns".format(len(train.first().split(","))))

# removing header

header = train.first()
content = train.filter(lambda line: line != header)

# checking first line of data

print('The first line of the data is: ')
print(content.first())

# parse data into appropriate format for model
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[-1], values[:-1])

parsedData = content.map(parsePoint)

# build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")

print('The model finished training, now evaluating...') # the results are not important

# evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# stopping SparkContext
sc.stop()