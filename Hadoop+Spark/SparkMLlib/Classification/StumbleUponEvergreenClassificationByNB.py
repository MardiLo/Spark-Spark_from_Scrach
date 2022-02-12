from pyspark import SparkConf, SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from time import time
# import pandas as pd
# import matplotlib.pyplot as plt


def extract_features(field, categoriesMap, featureEnd):
    # OneHot Encoding on Categorical Features
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryIdx = categoriesMap[field[3]]
    categoryFeatures[categoryIdx] = 1

    # Imputing Missing Values on Numerical Features
    numericalFeatures = [convert_float(f) for f in field[4:featureEnd]]

    return np.concatenate((categoryFeatures, numericalFeatures))


def convert_float(x):
    r = (0 if x == '?' else float(x))
    return 0 if r < 0 else r


def extract_label(field):
    return float(field[-1])


def PrepareData(sc):
    # Set a Line in the Text as a RDD Object
    linesData = sc.textFile("StumbleUponEvergreenClassification_Train.tsv", 30)

    # Remove Header
    header = linesData.first()
    linesData = linesData.filter(lambda x: x != header)

    # Parse lines
    linesRDD = linesData.map(lambda x: x.replace("\"", "")).map(lambda x: x.split("\t"))

    # Create a Map for Categorical Data
    categoriesMap = linesRDD.map(lambda x: x[3]).distinct().zipWithIndex().collectAsMap()

    # Standardization
    labelRDD = linesRDD.map(lambda x: extract_label(x))
    featureRDD = linesRDD.map(lambda x: extract_features(x, categoriesMap, len(x) - 1))
    stdScaler = StandardScaler(withMean=False, withStd=True).fit(featureRDD)
    StdFeatureRDD = stdScaler.transform(featureRDD)

    # Create a LabeledPointRDD
    labelpoint = labelRDD.zip(StdFeatureRDD)  # concatenate labelRDD and StdFeatureRDD to make labelpoint
    labeledPointRDD = labelpoint.map(lambda x: LabeledPoint(x[0], x[1]))

    # Random split (train:validation:test = 8:1:1)
    trainData, validationData, testData = labeledPointRDD.randomSplit(weights=[8, 1, 1], seed=42)
    print(f"Split into, test:{trainData.count()}, validation:{validationData.count()}, test:{testData.count()}")

    return trainData, validationData, testData, categoriesMap


def PredictData(sc, model, categoriesMap, predNumber=10):
    # Set a Line in the Text as a RDD Object
    linesData = sc.textFile("StumbleUponEvergreenClassification_Test.tsv", 5)

    # Remove Header
    header = linesData.first()
    linesData = linesData.filter(lambda x: x != header)

    # Parse lines
    linesRDD = linesData.map(lambda x: x.replace("\"", "")).map(lambda x: x.split("\t"))

    # Create a dataRDD
    dataRDD = linesRDD.map(lambda x: (x[0], extract_features(x, categoriesMap, len(x))))

    # Make a label description dictionary
    DescDict = {0: 'ephemeral', 1: 'evergreen'}

    # Only predict top N data
    for data in dataRDD.take(predNumber):
        pred = model.predict(data[1])
        print(f"url: {data[0]} --> predict as {DescDict[pred]}")

    return None


def evaluateModel(model, validationData):
    pred = model.predict(validationData.map(lambda x: x.features)).map(lambda x: float(x))
    predAndLabels = pred.zip(validationData.map(lambda x: x.label))
    metrics = BinaryClassificationMetrics(predAndLabels)
    AUC = round(metrics.areaUnderROC, 3)

    return AUC


def trainEvaluateModel(trainData, validationData, lambdaParam):
    record = dict()
    startTime = time()
    model = NaiveBayes.train(trainData, lambdaParam)
    AUC = evaluateModel(model, validationData)
    duration = round(time() - startTime, 2)

    record['lambdaParam'] = lambdaParam
    record['AUC'] = AUC
    record['duration'] = duration
    record['model'] = model

    return record


# def showchart(df, evalparm, barData, lineData, yMin, yMax):
#     ax = df[barData].plot(kind='bar', title=evalparm, figsize=(10, 6), legend=True, fontsize=12)
#     ax.set_xlabel(evalparm, fontsize=12)
#     ax.set_ylabel(barData, fontsize=12)
#     ax.set_ylim([yMin, yMax])
#     ax2 = ax.twinx()
#     ax2.plot(df[lineData].values, linestyle='-', marker='o', linewidth=2, color='r')
#     plt.show()


# def evalParameter(trainData, validationData, evalparm, lambdaParamList):
#     metrics = [trainEvaluateModel(trainData, validationData, lambdaParam)
#                for lambdaParam in lambdaParamList]
#
#     df = pd.DataFrame(metrics).set_index(evalparm)
#     showchart(df, evalparm, 'AUC', 'duration', 0.5, 0.7)


def evalAllParameter(trainData, validationData, lambdaParamList):
    metrics = [trainEvaluateModel(trainData, validationData, lambdaParam)
               for lambdaParam in lambdaParamList]

    bestParameter = sorted(metrics, key=lambda k: k['AUC'], reverse=True)[0]
    print(f"Find Best Parameter: {bestParameter}")

    return bestParameter['model']


if __name__ == "__main__":
    # Set Spark Config. (run in local by a cpu and name the job as "StumbleUponEvergreenClassificationByNB")
    # Make a SparkContext object, and Depreciate Spark Warning
    conf = SparkConf().setMaster("local").setAppName("StumbleUponEvergreenClassificationByNB")
    conf.set('spark.executor.cores', '4')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Prepare Raw Data
    print(">>> Preparing Raw Data >>>")
    trainData, validationData, testData, categoriesMap = PrepareData(sc)

    # Persist data in RAM
    print(">>> Persisting Data >>>")
    trainData.persist(), validationData.persist(), testData.persist()
    print("Status: Persist data Successfully!")

    # Train and Find best model
    print(">>> Training and Finding Best Model >>>")
    lambdaParamList = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    bestModel = evalAllParameter(trainData, validationData, lambdaParamList)

    # Test best model on test data set to check whether there's overfitting or not
    print(">>> Evaluating testData with Best Model >>>")
    validationAUC = evaluateModel(bestModel, validationData)
    testAUC = evaluateModel(bestModel, testData)
    print(f"AUC of validation set: {validationAUC}")
    print(f"AUC of test set: {testAUC}")

    # Predict by best model
    predNumber = 10
    print(f">>> Predicting Top {predNumber} urls >>>")
    PredictData(sc, bestModel, categoriesMap, predNumber)
