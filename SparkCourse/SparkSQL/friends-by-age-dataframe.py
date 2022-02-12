from pyspark.sql import SparkSession
from pyspark.sql import functions as func

# Create a SparkSession object and name the job as "SparkSQL"
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# Create a DataFrame object
people = spark.read.option("header", "true").option("inferSchema", "true").csv("fakefriends-header.csv")

# Select only age and numFriends columns
friendsByAge = people.select("age", "friends")

# Get Result
friendsByAge.groupBy("age").agg(func.round(func.avg("friends"), 2).alias("friends_avg")).sort("age").show()

# spark.stop()
