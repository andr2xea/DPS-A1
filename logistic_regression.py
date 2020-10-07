# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:51:05 2020

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

# build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

print('The model finished training, now evaluating...') # the results are not important

# evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# stopping SparkContext
sc.stop()