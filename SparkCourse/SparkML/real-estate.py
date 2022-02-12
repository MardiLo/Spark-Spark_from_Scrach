from __future__ import print_function
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.ml.feature import VectorAssembler

if __name__ == "__main__":

    # Create a SparkSession (Note, the config section is only for Windows!)
    spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

    # Load up data as dataframe
    data = spark.read.option("header", "true").option("inferSchema", "true").csv("realestate.csv")

    assembler = VectorAssembler().setInputCols(["HouseAge",
                                                "DistanceToMRT",
                                                "NumberConvenienceStores"]).setOutputCol("features")  # <class 'pyspark.ml.feature.VectorAssembler'>

    df = assembler.transform(data).select("PriceOfUnitArea", "features")
    print("df:")
    df.show(5)

    # Let's split our data into training data and testing data
    trainTest = df.randomSplit([0.8, 0.2])
    trainingDF = trainTest[0]
    testDF = trainTest[1]

    # Now create our decision tree
    dtr = DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("PriceOfUnitArea")

    # Train the model using our training data
    model = dtr.fit(trainingDF)

    # Now see if we can predict values in our test data.
    # Generate predictions using our decision tree model for all features in our
    # test dataframe:
    fullPredictions = model.transform(testDF).cache()
    fullPredictions = fullPredictions.select("PriceOfUnitArea", func.round("prediction", 2).alias("prediction"))
    print("fullPredictions:")
    fullPredictions.show(fullPredictions.count())

    # Stop the session
    # spark.stop()
