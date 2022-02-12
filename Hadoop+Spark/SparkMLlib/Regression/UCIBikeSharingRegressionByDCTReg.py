from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
from time import time
import numpy as np
import math


def extract_features(field, featureEnd):
    # Imputing Missing Values on Numerical Features
    seasonFeatures = [convert_float(f) for f in field[2]]
    numericalFeatures = [convert_float(f) for f in field[4:featureEnd - 2]]

    return np.concatenate((seasonFeatures, numericalFeatures))


def convert_float(x):
    return 0 if x == '?' else float(x)


def extract_label(field):
    return float(field[-1])


def PrepareData(sc):
    # Set a Line in the Text as a RDD Object
    linesWithHeader = sc.textFile("UCIBikeSharingRegression_Train.csv", 5)
    header = linesWithHeader.first()
    linesData = linesWithHeader.filter(lambda x: x != header)

    # Parse lines
    linesRDD = linesData.map(lambda x: x.split(","))

    # Create a LabeledPointRDD
    labeledPointRDD = linesRDD.map(lambda x: LabeledPoint(extract_label(x),
                                                          extract_features(x, len(x) - 1)))

    # Random split (train:validation:test = 8:1:1)
    trainData, validationData, testData = labeledPointRDD.randomSplit(weights=[8, 1, 1], seed=42)
    print(f"Split into, test:{trainData.count()}, validation:{validationData.count()}, test:{testData.count()}")

    return trainData, validationData, testData


def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda x: x.features))
    scoreAndLabels = score.zip(validationData.map(lambda x: x.label))
    metrics = RegressionMetrics(scoreAndLabels)
    RMSE = round(metrics.rootMeanSquaredError, 3)

    return RMSE


def trainEvaluateModel(trainData, validationData, impurityParm, maxDepthParm, maxBinsParm):
    record = dict()
    startTime = time()
    model = DecisionTree.trainRegressor(trainData,
                                        categoricalFeaturesInfo={},
                                        impurity=impurityParm,
                                        maxDepth=maxDepthParm,
                                        maxBins=maxBinsParm)
    RMSE = evaluateModel(model, validationData)
    duration = round(time() - startTime, 2)

    record['impurity'] = impurityParm
    record['maxDepth'] = maxDepthParm
    record['maxBins'] = maxBinsParm
    record['RMSE'] = RMSE
    record['duration'] = duration
    record['model'] = model

    return record


def evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]

    bestParameter = sorted(metrics, key=lambda k: k['RMSE'], reverse=True)[0]
    print(f"Find Best Parameter: {bestParameter}")

    return bestParameter['model']


def PredictData(sc, model, predNumber=10):
    # Set a Line in the Text as a RDD Object
    linesWithHeader = sc.textFile("UCIBikeSharingRegression_Test.csv", 5)
    header = linesWithHeader.first()
    linesData = linesWithHeader.filter(lambda x: x != header)

    # Parse lines
    linesRDD = linesData.map(lambda x: x.split(","))

    # Create a LabeledPointRDD
    labeledPointRDD = linesRDD.map(lambda x: LabeledPoint(extract_label(x),
                                                          extract_features(x, len(x) - 1)))

    # Make a label description dictionary
    SeasonDict = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    HolidayDict = {0: 'Not-Holiday', 1: 'Holiday'}
    WeekDict = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    WorkdayDict = {0: 'Workday', 1: 'Holiday'}
    WeatherDict = {1: 'Sunny', 2: 'Cloudy', 3: 'Light Rain', 4: 'Heavy Rain'}

    # Only predict top N data
    for lp in labeledPointRDD.take(predNumber):
        pred = int(model.predict(lp.features))
        label = lp.label
        features = lp.features
        bias = math.fabs(label - pred)

        print(f"[Season: {SeasonDict[int(features[0])]}, "
              f"Month: {int(features[1])}, "
              f"Weekday: {WeekDict[int(features[3])]}, "
              f"Workday: {WorkdayDict[int(features[4])]}, "
              f"Holiday: {HolidayDict[int(features[2])]}, "
              f"Weather: {WeatherDict[int(features[5])]}, "
              f"Temperature: {round(features[6] * 41, 2)}, "
              f"FeelingTemperature: {round(features[7] * 50, 2)}, "
              f"Humidity: {round(features[8] * 100, 2)}, "
              f"WindSpeed: {round(features[9] * 67, 2)}] "
              f"--> predict cnt: {pred}, real cnt: {label}, bias: {bias}")

    return None


if __name__ == "__main__":
    # Set Spark Config. (run in local by a cpu and name the job as "UCIBikeSharingRegressionByDCTReg")
    # Make a SparkContext object, and Depreciate Spark Warning
    conf = SparkConf().setMaster("local").setAppName("UCIBikeSharingRegressionByDCTReg")
    conf.set('spark.executor.cores', '4')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Prepare Raw Data
    print(">>> Preparing Raw Data >>>")
    trainData, validationData, testData = PrepareData(sc)

    # Persist data in RAM
    print(">>> Persisting Data >>>")
    trainData.persist(), validationData.persist(), testData.persist()
    print("Status: Persist data Successfully!")

    # Train and Find best model
    print(">>> Training and Finding the Best Model >>>")
    impurityList = ['variance']
    maxDepthList = [5, 10]
    maxBinsList = [100, 200]

    bestModel = evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList)

    # Test best model on test data set to check whether there's overfitting or not
    print(">>> Evaluating testData with Best Model >>>")
    validationRMSE = evaluateModel(bestModel, validationData)
    testRMSE = evaluateModel(bestModel, testData)
    print(f"RMSE of validation set: {validationRMSE}")
    print(f"RMSE of test set: {testRMSE}")

    # Predict by best model
    predNumber = 10
    print(f">>> Predicting first {predNumber} data >>>")
    PredictData(sc, bestModel, predNumber)
