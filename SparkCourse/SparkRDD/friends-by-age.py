from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "FriendsByAge")
conf = SparkConf().setMaster("local").setAppName("FriendsByAge")

# Make a SparkContext object
sc = SparkContext(conf=conf)


# Define a Parse Function (str -> (int, int))
def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)


# Set a Line in the Text as a RDD Object
lines = sc.textFile("fakefriends.csv")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Map RDD Values
rdd = lines.map(parseLine)
print(f"type of rdd: {type(rdd)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"rdd: {rdd.take(5)}\n")  # [(33, 385), (26, 2),...]

# Map RDD Values ((int, int) -> (int, (int, 1)))
totalsByAgeMeta = rdd.mapValues(lambda x: (x, 1))
print(f"type of totalsByAgeMeta: {type(totalsByAgeMeta)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"totalsByAgeMeta: {totalsByAgeMeta.take(5)}\n")  # [(33, (385, 1)), (26, (2, 1)),...]

# Sum RDD up by Key
totalsByAge = totalsByAgeMeta.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
print(f"type of totalsByAge: {type(totalsByAge)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"totalsByAge: {totalsByAge.take(5)}\n")  # [(33, (3904, 12)), (26, (4115, 17)),...]

# Map RDD Values ((int, (int, 1)) -> (int, int))
averagesByAge = totalsByAge.mapValues(lambda x: round(x[0] / x[1], 2))
print(f"type of result: {type(averagesByAge)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"result: {averagesByAge.take(5)}\n")  # [(33, 325.33), (26, 242.06),...]

# Take Out All Elements in RDD object
results = averagesByAge.collect()
for result in results:
    print(result)
