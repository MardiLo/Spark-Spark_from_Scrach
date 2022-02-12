from __future__ import print_function
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.ml.linalg import Vectors


if __name__ == "__main__":

    # Create a SparkSession (Note, the config section is only for Windows!)
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

    # Load up our data and convert it to the format MLLib expects.
    lines = spark.sparkContext.textFile("regression.txt")  # -1.74,1.66

    # Map RDD Values
    lineSplit = lines.map(lambda x: x.split(","))  # [['-1.74', '1.66'], ['1.24', '-1.18'],...]
    print(f"lineSplit: {lineSplit.take(5)}\n")

    # Map RDD Values ((str, str) -> (float,  DenseVector))
    # DenseVector collects all the features as training input, and now we only have one feature
    data = lineSplit.map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))  # [(-1.74, DenseVector([1.66])), (1.24, DenseVector([-1.18])),...]
    print(f"data: {data.take(5)}\n")

    # Convert this RDD to a DataFrame
    colNames = ["label", "features"]
    df = data.toDF(colNames)
    print("df:")
    df.show(5)

    # Note, there are lots of cases where you can avoid going from an RDD to a DataFrame.
    # Perhaps you're importing data from a real database. Or you are using structured streaming
    # to get your data.

    # Let's split our data into training data and testing data
    trainTest = df.randomSplit([0.8, 0.2])  # [DataFrame[label: double, features: vector], DataFrame[label: double, features: vector]]
    trainingDF = trainTest[0]
    testDF = trainTest[1]

    # Now create our linear regression model
    lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Train the model using our training data
    model = lir.fit(trainingDF)

    # Now see if we can predict values in our test data.
    # Generate predictions using our linear regression model for all features in our
    # test dataframe:
    fullPredictions = model.transform(testDF).cache()
    fullPredictions = fullPredictions.select("label", func.round("prediction", 2).alias("prediction"))
    print("fullPredictions:")
    fullPredictions.show(fullPredictions.count())

    # Stop the session
    # spark.stop()
