from pyspark.sql import SparkSession
from pyspark.sql import functions as func

# Create a SparkSession object and name the job as "WordCount"
spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Read each line of my book into a dataframe
inputDF = spark.read.text("book.txt")
print(f"type of inputDF: {type(inputDF)}")  # <class 'pyspark.sql.dataframe.DataFrame'>
print(f"inputDF schema: {inputDF}")  # DataFrame[value: string]
print(f"inputDF content: {inputDF.take(5)}\n")  # [Row(value='Self-Employment: ...'), Row(value='...'),...]

# Split using a regular expression that extracts words
splitDf = inputDF.select(func.split(inputDF.value, "\\W+"))
splitDf.show(5)

# Flattening by "func.explode()"
print("Flattening:")
wordsDF = inputDF.select(func.explode(func.split(inputDF.value, "\\W+")).alias("word"))
wordsDF.show(5)

# The "Where" Clause
wordsDF.filter(wordsDF.word != "")

# Normalize everything to lowercase
lowercaseWords = wordsDF.select(func.lower(wordsDF.word).alias("word"))

# Count up the occurrences of each word
wordCounts = lowercaseWords.groupBy("word").count()

# Sort by counts
wordCountsSorted = wordCounts.sort("count")

# Show all elements
wordCountsSorted.show(wordCountsSorted.count())
