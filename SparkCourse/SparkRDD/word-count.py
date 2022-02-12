from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "WordCount")
conf = SparkConf().setMaster("local").setAppName("WordCount")

# Make a SparkContext object
sc = SparkContext(conf=conf)

# Set a Line in the Text as a RDD Object
lines = sc.textFile("book.txt")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Map RDD Values (str -> str, str,..., str), Note that FlatMap can Flatten Lists
wordsflatMap = lines.flatMap(lambda x: x.split())
print(f"type of wordsflatMap: {type(wordsflatMap)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"wordsflatMap: {wordsflatMap.take(5)}\n")  # ['Self-Employment:', 'Building',...]

# Map RDD Values, These Codes Below Are for Comparison
wordsMap = lines.map(lambda x: x.split())
print(f"type of wordsMap: {type(wordsMap)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"wordsMap: {wordsMap.take(5)}\n")  # [['Self-Employment:', 'Building',...], ['Achieving', 'Financial',...]]

# Calculate Word Frequency
wordCounts = wordsflatMap.countByValue()
print(f"type of wordCounts: {type(wordCounts)}")  # <class 'collections.defaultdict'>
print(f"wordCounts: {wordCounts}\n")  # defaultdict({'Self-Employment:': 1, 'Building': 5,...})

# Print Result
for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if cleanWord:
        print(cleanWord.decode() + " " + str(count))
