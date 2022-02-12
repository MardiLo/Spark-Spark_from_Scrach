from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
from time import time


def extract_features(field, featureEnd):
    # Imputing Missing Values on Numerical Features
    numericalFeatures = [convert_float(f) for f in field[0:featureEnd]]

    return numericalFeatures


def convert_float(x):
    return 0 if x == '?' else float(x)


def extract_label(field):
    return float(field[-1]) - 1


def PrepareData(sc):
    # Set a Line in the Text as a RDD Object
    linesData = sc.textFile("UCICovertypeMultiClassification_Train.data")

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
    pred = model.predict(validationData.map(lambda x: x.features))
    predAndLabels = pred.zip(validationData.map(lambda x: x.label))
    metrics = MulticlassMetrics(predAndLabels)
    ACC = round(metrics.accuracy, 3)

    return ACC


def trainEvaluateModel(trainData, validationData, impurityParm, maxDepthParm, maxBinsParm):
    record = dict()
    startTime = time()
    model = DecisionTree.trainClassifier(trainData,
                                         numClasses=7,
                                         categoricalFeaturesInfo={},
                                         impurity=impurityParm,
                                         maxDepth=maxDepthParm,
                                         maxBins=maxBinsParm)
    ACC = evaluateModel(model, validationData)
    duration = round(time() - startTime, 2)

    record['impurity'] = impurityParm
    record['maxDepth'] = maxDepthParm
    record['maxBins'] = maxBinsParm
    record['ACC'] = ACC
    record['duration'] = duration
    record['model'] = model

    return record


def evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]

    bestParameter = sorted(metrics, key=lambda k: k['ACC'], reverse=True)[0]
    print(f"Find Best Parameter: {bestParameter}")

    return bestParameter['model']


if __name__ == "__main__":
    # Set Spark Config. (run in local by a cpu and name the job as "UCICovertypeMultiClassificationByDCT")
    # Make a SparkContext object, and Depreciate Spark Warning
    conf = SparkConf().setMaster("local").setAppName("UCICovertypeMultiClassificationByDCT")
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
    impurityList = ['entropy']
    maxDepthList = [5, 10, 15]
    maxBinsList = [10, 50, 100]

    bestModel = evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList)

    # Test best model on test data set to check whether there's overfitting or not
    print(">>> Evaluating testData with Best Model >>>")
    validationACC = evaluateModel(bestModel, validationData)
    testACC = evaluateModel(bestModel, testData)
    print(f"ACC of validation set: {validationACC}")
    print(f"ACC of test set: {testACC}")
