from pyspark import SparkConf, SparkContext
import collections

# Set Spark Config. (run in local by a cpu and name the job as "RatingsHistogram")
conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")

# Make a SparkContext object
sc = SparkContext(conf=conf)

# Set a Line in the Text as a RDD Object
lines = sc.textFile("ml-100k/u.data")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Map RDD Values (str -> str)
ratings = lines.map(lambda x: x.split()[2])
print(f"type of ratings: {type(ratings)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"ratings: {ratings.take(5)}\n")  # ['3', '3',...]

# Calculate Rating Frequency
result = ratings.countByValue()
print(f"type of result: {type(result)}")  # <class 'collections.defaultdict'>
print(f"result: {result}\n")  # defaultdict({'3': 27145, '1': 6110,...})

# Sort Result by Key
sortedResults = collections.OrderedDict(sorted(result.items()))
print(f"type of sortedResults: {type(sortedResults)}")  # <class 'collections.OrderedDict'>
print(f"sortedResults: {sortedResults}\n")  # OrderedDict([('1', 6110), ('2', 11370),...])

# Print Result
for key, value in sortedResults.items():
    print("%s %i" % (key, value))
