import re
from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "WordCountBetterSorted")
conf = SparkConf().setMaster("local").setAppName("WordCountBetterSorted")

# Make a SparkContext object
sc = SparkContext(conf=conf)


# Define a Regex (str -> str)
def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())


# Set a Line in the Text as a RDD Object
lines = sc.textFile("book.txt")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Apply Regex to RDD
words = lines.flatMap(normalizeWords)
print(f"type of words: {type(words)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"words: {words.take(5)}\n")  # ['self', 'employment', 'building', 'an', 'internet']

# Map RDD Values (str -> (str, 1))
wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
print(f"type of wordCounts: {type(wordCounts)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"wordCounts: {wordCounts.take(5)}\n")  # [('self', 111), ('employment', 75),...]

# Switch Key and Value (Key, Value -> Value, Key), and then Sort it by Key (which now is Value)
wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
print(f"type of wordCountsSorted: {type(wordCountsSorted)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"wordCountsSorted: {wordCountsSorted.take(5)}\n")  # [(1, 'achieving'), (1, 'contents'),...]

# Take Out All Elements in RDD object
results = wordCountsSorted.collect()
for key, value in results:
    count = str(key)
    word = value.encode('ascii', 'ignore')
    if word:
        print(word.decode() + " " + count)
