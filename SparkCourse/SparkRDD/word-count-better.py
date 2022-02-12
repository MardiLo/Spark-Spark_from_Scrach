import re
from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "WordCountBetter")
conf = SparkConf().setMaster("local").setAppName("WordCountBetter")

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

# Calculate Word Frequency
wordCounts = words.countByValue()
print(f"type of wordCounts: {type(wordCounts)}")  # <class 'collections.defaultdict'>
print(f"wordCounts: {wordCounts}\n")  # defaultdict({'self': 111, 'employment': 75,...})

# Print Result
for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if cleanWord:
        print(cleanWord.decode() + " " + str(count))
