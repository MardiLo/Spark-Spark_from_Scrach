from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
import codecs

# Define a map of movie name and movie ID (1:"Toy Story (1995)", 2:...,...)
def loadMovieNames():
    movieNames = {}

    with codecs.open("ml-100k/u.ITEM", "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


# Create a SparkSession object and name the job as "PopularMovies"
spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

# Create schema when reading u.data
schema = StructType([StructField("userID", IntegerType(), True),
                     StructField("movieID", IntegerType(), True),
                     StructField("rating", IntegerType(), True),
                     StructField("timestamp", LongType(), True)])

# Load up movie data as dataframe
moviesDF = spark.read.option("sep", "\t").schema(schema).csv("ml-100k/u.data")

movieCounts = moviesDF.groupBy("movieID").count()

# Create a broadcasted dictionary
nameDict = spark.sparkContext.broadcast(loadMovieNames())
print(f"type of nameDict: {type(nameDict)}")  # <class 'pyspark.broadcast.Broadcast'>

# Create a user-defined function to look up movie names from our broadcasted dictionary
def lookupName(movieID):
    return nameDict.value[movieID]

lookupNameUDF = func.udf(lookupName)
print(f"type of lookupNameUDF: {type(lookupNameUDF)}")  # <class 'function'>


# Add a movieTitle column using our new udf
moviesWithNames = movieCounts.withColumn("movieTitle", lookupNameUDF(func.col("movieID")))

# Sort the results
sortedMoviesWithNames = moviesWithNames.orderBy(func.desc("count"))

# Grab the top 10
sortedMoviesWithNames.show(10, False)

# Stop the session
spark.stop()
