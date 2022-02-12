from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a SparkSession object and Name the Job as "UCICovertypeMultiClassificationByDCT_MLPipeline"
sparkSQL = SparkSession.builder.appName("UCICovertypeMultiClassificationByDCT_MLPipeline").getOrCreate()

# Set a Line in the Text as a RDD Object
print(">>> Preparing Raw Data >>>")
TrainFile = "UCICovertypeMultiClassification_TrainMini.data"
raw_df = sparkSQL.read.format("csv").load(TrainFile)

# Transforming data into Double type on Numerical Features
df = raw_df.select([col(c).cast("double").alias(c) for c in raw_df.columns])

# Rename Label Column and Subtract all Elements by 1
df = df.withColumn("label", df["_c54"] - 1).drop("_c54")

# Random Split (train:test = 9:1)
train_df, test_df = df.randomSplit([0.9, 0.1])
print(f"Split into, train:{train_df.count()}, test:{test_df.count()}")

# Persist Data in RAM
print(">>> Persisting Data >>>")
train_df.cache(), test_df.cache()
print("Status: Persist data Successfully!")

# Create a List of Features
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")

# Build a Model with CrossValidation and GridSearch
rf = RandomForestClassifier(labelCol="label",
                            featuresCol="features")
ParamGrid = ParamGridBuilder().addGrid(rf.impurity, ["entropy"]) \
                              .addGrid(rf.maxDepth, [10, 20]) \
                              .addGrid(rf.maxBins, [100]) \
                              .addGrid(rf.numTrees, [100]).build()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                              labelCol="label",
                                              metricName="accuracy")
cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=ParamGrid, numFolds=5)

# Create a Model Training Pipeline
print(">>> Creating a Pipeline >>>")
pipeline = Pipeline(stages=[assembler, cv])
print("Status: A Pipeline is Created Successfully!")

# Train and Find Best Model
print(">>> Training and Finding the Best Model >>>")
pipelineModel = pipeline.fit(train_df)
print(f"BestModel: {pipelineModel.stages[1].bestModel}")

# Predict by Best Model
print(f">>> Predicting by Best Model >>>")
predicted_df = pipelineModel.transform(test_df)
Accuracy = evaluator.evaluate(predicted_df)
print(f"Accuracy of the Best Model: {Accuracy}")


