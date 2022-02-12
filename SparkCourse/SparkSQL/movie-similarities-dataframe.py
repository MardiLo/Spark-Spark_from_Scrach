from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import sys
import random


def computeCosineSimilarity(data):
    # Compute xx, xy and yy columns
    pairScores = data.withColumn("xx", func.col("rating1") * func.col("rating1")) \
        .withColumn("yy", func.col("rating2") * func.col("rating2")) \
        .withColumn("xy", func.col("rating1") * func.col("rating2"))

    # Compute numerator, denominator and numPairs columns
    calculateSimilarity = pairScores.groupBy("movie1", "movie2") \
        .agg(func.sum(func.col("xy")).alias("numerator"),
             (func.sqrt(func.sum(func.col("xx"))) * func.sqrt(func.sum(func.col("yy")))).alias("denominator"),
             func.count(func.col("xy")).alias("numPairs"))

    # Calculate score and select only needed columns (movie1, movie2, score, numPairs)
    result = calculateSimilarity.withColumn("score",
                                            func.when(func.col("denominator") != 0,
                                                      func.col("numerator") / func.col("denominator")) \
                                            .otherwise(0)).select("movie1",
                                                                  "movie2",
                                                                  "score",
                                                                  "numPairs")

    return result


# Get movie name by given movie id
def getMovieName(movieNames, movieId):
    result = movieNames.filter(func.col("movieID") == movieId).select("movieTitle").collect()[0][0]  # 'Toy Story (1995)'

    return result


spark = SparkSession.builder.appName("MovieSimilarities").getOrCreate()

movieNamesSchema = StructType([StructField("movieID", IntegerType(), True),
                               StructField("movieTitle", StringType(), True)])

moviesSchema = StructType([StructField("userID", IntegerType(), True),
                           StructField("movieID", IntegerType(), True),
                           StructField("rating", IntegerType(), True),
                           StructField("timestamp", LongType(), True)])

# Create a broadcast dataset of movieID and movieTitle.
# Apply ISO-885901 charset
movieNames = spark.read.option("sep", "|").option("charset", "ISO-8859-1").schema(movieNamesSchema).csv(
    "ml-100k/u.item")

# Load up movie data as dataset
movies = spark.read.option("sep", "\t").schema(moviesSchema).csv("ml-100k/u.data")

ratings = movies.select("userId", "movieId", "rating")

# Emit every movie rated together by the same user.
# Self-join to find every combination.
# Select movie pairs and rating pairs
moviePairs = ratings.alias("r1").join(ratings.alias("r2"),
                                      (func.col("r1.userId") == func.col("r2.userId")) &
                                      (func.col("r1.movieId") < func.col("r2.movieId"))) \
    .select(func.col("r1.movieId").alias("movie1"),
            func.col("r2.movieId").alias("movie2"),
            func.col("r1.rating").alias("rating1"),
            func.col("r2.rating").alias("rating2"))

moviePairSimilarities = computeCosineSimilarity(moviePairs).cache()
print("moviePairSimilarities")
moviePairSimilarities.show()

if len(sys.argv):
    scoreThreshold = 0.97
    coOccurrenceThreshold = 50.0

    movieID = 1  # Toy Story (1995)

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = moviePairSimilarities.filter(((func.col("movie1") == movieID) |
                                                    (func.col("movie2") == movieID)) &
                                                    (func.col("score") > scoreThreshold) &
                                                    (func.col("numPairs") > coOccurrenceThreshold))

    # Sort by quality score.
    results = filteredResults.sort(func.col("score").desc()).take(10)  # [Row(movie1, movie2, score, numPairs),...]

    print("Top 10 similar movies for " + getMovieName(movieNames, movieID))

    for result in results:
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = result.movie1
        if similarMovieID == movieID:
            similarMovieID = result.movie2

        print(f"{getMovieName(movieNames, similarMovieID)}\tscore: {result.score:.3f}\tstrength: {result.numPairs}")
