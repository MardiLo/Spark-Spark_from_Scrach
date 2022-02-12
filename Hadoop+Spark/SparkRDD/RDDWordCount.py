from pyspark import SparkConf, SparkContext
import shutil

# Set Spark Config. (run in local by a cpu and name the job as "RDDWordCount")
conf = SparkConf().setMaster("local").setAppName("RDDWordCount")

# Main entry point for Spark functionality
sc = SparkContext(conf=conf)

# Set a Line in the Text as a RDD Object
lines = sc.textFile("WordCountTest.txt")
print(f"lines: {lines.collect()}\n")

# Map RDD Values (str -> str, str,..., str), Note that FlatMap can Flatten Lists
wordsRDD = lines.flatMap(lambda x: x.split())
print(f"wordsRDD: {wordsRDD.collect()}\n")

# Calculate Word Frequency
countsRDD = wordsRDD.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
print(f"Count wordsRDD by Key: {countsRDD.collect()}")

# Delete Existed Folder and Save Result
shutil.rmtree("WordCountStat", ignore_errors=True)
countsRDD.saveAsTextFile("WordCountStat")
