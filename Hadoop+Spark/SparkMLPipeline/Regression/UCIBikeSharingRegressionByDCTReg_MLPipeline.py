from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a SparkSession object and Name the Job as "UCIBikeSharingRegressionByDCTReg_MLPipeline"
sparkSQL = SparkSession.builder.appName("UCIBikeSharingRegressionByDCTReg_MLPipeline").getOrCreate()

# Set a Line in the Text as a RDD Object
print(">>> Preparing Raw Data >>>")
TrainFile = "UCIBikeSharingRegression_Train.csv"
raw_df = sparkSQL.read.format("csv").option("header", "true").load(TrainFile)

# Drop Useless Columns
raw_df = raw_df.drop("instant").drop("dteday").drop("yr").drop("casual").drop("registered")

# Transforming data into Double type on Numerical Features
df = raw_df.select([col(c).cast("double").alias(c) for c in raw_df.columns])

# Random Split (train:test = 9:1)
train_df, test_df = df.randomSplit([0.9, 0.1])
print(f"Split into, train:{train_df.count()}, test:{test_df.count()}")

# Persist Data in RAM
print(">>> Persisting Data >>>")
train_df.cache(), test_df.cache()
print("Status: Persist data Successfully!")

# Create a List of Features
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="meta_features")
vectorIndexer = VectorIndexer(inputCol="meta_features", outputCol="features", maxCategories=24)

# Build a Model with CrossValidation and GridSearch
dctReg = DecisionTreeRegressor(labelCol="cnt",
                               featuresCol="features")
ParamGrid = ParamGridBuilder().addGrid(dctReg.maxDepth, [5, 10]) \
                              .addGrid(dctReg.maxBins, [100, 200]).build()
evaluator = RegressionEvaluator(predictionCol="prediction",
                                labelCol="cnt",
                                metricName="rmse")
cv = CrossValidator(estimator=dctReg, evaluator=evaluator, estimatorParamMaps=ParamGrid, numFolds=5)

# Create a Model Training Pipeline
print(">>> Creating a Pipeline >>>")
pipeline = Pipeline(stages=[assembler, vectorIndexer, cv])
print("Status: A Pipeline is Created Successfully!")

# Train and Find Best Model
print(">>> Training and Finding the Best Model >>>")
pipelineModel = pipeline.fit(train_df)
print(f"BestModel: {pipelineModel.stages[2].bestModel}")

# Predict by Best Model
print(f">>> Predicting by Best Model >>>")
predicted_df = pipelineModel.transform(test_df)
RMSE = evaluator.evaluate(predicted_df)
print(f"RMSE of the Best Model: {RMSE}")


