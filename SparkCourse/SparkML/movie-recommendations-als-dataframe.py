from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as func
import codecs
import random


def loadMovieNames():
    movieNames = {}
    with codecs.open("ml-100k/u.ITEM", "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            fields = line.split('|')  # ['1', 'Toy Story (1995)',...]
            movieNames[int(fields[0])] = fields[1]  # {1: 'Toy Story (1995)', 2: 'GoldenEye (1995)'...}
    return movieNames


spark = SparkSession.builder.appName("ALSExample").getOrCreate()

moviesSchema = StructType([StructField("userID", IntegerType(), True),
                           StructField("movieID", IntegerType(), True),
                           StructField("rating", IntegerType(), True),
                           StructField("timestamp", LongType(), True)])

names = loadMovieNames()

ratings = spark.read.option("sep", "\t").schema(moviesSchema).csv("ml-100k/u.data")

print("Training recommendation model...")

als = ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userID").setItemCol("movieID").setRatingCol("rating")

model = als.fit(ratings)

# Manually construct a dataframe of the user ID's we want recs for
minUserID = ratings.agg(func.min("userID")).first()[0]
maxUserID = ratings.agg(func.max("userID")).first()[0]
userID = random.randint(minUserID, maxUserID)

userSchema = StructType([StructField("userID", IntegerType(), True)])
users = spark.createDataFrame([[userID, ]], userSchema)

recommendNumber = 10
recommendations = model.recommendForUserSubset(users, recommendNumber).collect()  # <class 'list'>
print(f"Top {recommendNumber} recommendations for user ID {userID}:")

i = 1
for userRecs in recommendations:
    Recs = userRecs[1]  # userRecs is (userID, [Row(movieId, rating), Row(movieID, rating)...])
    for rec in Recs:
        movie = rec[0]  # For each rec in the list, extract the movie ID
        rating = rec[1]  # For each rec in the list, extract the rating
        movieName = names[movie]
        print(f"{i}. {movieName}, {rating}")
        i += 1
