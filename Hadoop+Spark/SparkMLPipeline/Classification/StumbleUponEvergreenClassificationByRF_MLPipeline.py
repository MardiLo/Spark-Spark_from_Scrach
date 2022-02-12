from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def replace_question(x):
    return "0" if x == "?" else x

def PredictData(df, predNumber=10):
    # Make a label description dictionary
    DescDict = {0: "ephemeral", 1: "evergreen"}

    # Only predict top N data
    for data in df.select("url", "prediction").take(predNumber):
        print(f"url: {data[0]} --> predict as {DescDict[data[1]]}")

    return None


# Create a SparkSession object and Name the Job as "StumbleUponEvergreenClassificationByRF_MLPipeline"
sparkSQL = SparkSession.builder.appName("StumbleUponEvergreenClassificationByRF_MLPipeline").getOrCreate()

# Set a Line in the Text as a RDD Object
print(">>> Preparing Raw Data >>>")
TrainFile = "StumbleUponEvergreenClassification_Train.tsv"
raw_df = sparkSQL.read.format("csv").option("header", "true").option("delimiter", "\t").load(TrainFile)

# Imputing Missing Values and Transforming data into Double type on Numerical Features
replace_question = udf(replace_question)
df = raw_df.select(["url", "alchemy_category"] +
                   [replace_question(col(c)).cast("double").alias(c) for c in raw_df.columns[4:]])

# Random Split (train:test = 9:1)
train_df, test_df = df.randomSplit([0.9, 0.1])
print(f"Split into, train:{train_df.count()}, test:{test_df.count()}")

# Persist Data in RAM
print(">>> Persisting Data >>>")
train_df.cache(), test_df.cache()
print("Status: Persist data Successfully!")

# Label Encoding on "alchemy_category"
categoryIndexer = StringIndexer(inputCol="alchemy_category",
                                outputCol="alchemy_category_Indexer")

# One Encoding on "alchemy_category_Indexer"
oneHotEncoder = OneHotEncoder(dropLast=False,
                              inputCol="alchemy_category_Indexer",
                              outputCol="alchemy_category_IndexVec")

# Create a List of Features
assemblerInputs = ["alchemy_category_IndexVec"] + raw_df.columns[4:-1]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# Build a Model with CrossValidation and GridSearch
rf = RandomForestClassifier(labelCol="label",
                            featuresCol="features",
                            impurity="entropy",
                            maxDepth=10,
                            maxBins=20)
ParamGrid = ParamGridBuilder().addGrid(rf.impurity, ["entropy"]) \
                              .addGrid(rf.maxDepth, [10, 20]) \
                              .addGrid(rf.maxBins, [50, 100]) \
                              .addGrid(rf.numTrees, [50, 100]).build()
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                          labelCol="label",
                                          metricName="areaUnderROC")
cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=ParamGrid, numFolds=5)

# Create a Model Training Pipeline
print(">>> Creating a Pipeline >>>")
pipeline = Pipeline(stages=[categoryIndexer, oneHotEncoder, assembler, cv])
print("Status: A Pipeline is Created Successfully!")

# Train and Find Best Model
print(">>> Training and Finding the Best Model >>>")
pipelineModel = pipeline.fit(train_df)
print(f"BestModel: {pipelineModel.stages[3].bestModel}")

# Predict by Best Model
print(f">>> Predicting by Best Model >>>")
predicted_df = pipelineModel.transform(test_df)
AUC = evaluator.evaluate(predicted_df)
print(f"AUC of the Best Model: {AUC}")

predNumber = 10
print(f">>> Predicting Top {predNumber} urls >>>")
PredictData(predicted_df, predNumber)
