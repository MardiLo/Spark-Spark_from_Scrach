from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "MinTemperatures")
conf = SparkConf().setMaster("local").setAppName("MinTemperatures")

# Make a SparkContext object
sc = SparkContext(conf=conf)


# Define a Parse Function (str -> (str, str, float))
def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = round(float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0, 2)
    return (stationID, entryType, temperature)


# Set a Line in the Text as a RDD Object
lines = sc.textFile("1800.csv")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Map RDD Values (str -> (str, str, float))
parsedLines = lines.map(parseLine)
print(f"type of parsedLines: {type(parsedLines)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"parsedLines: {parsedLines.take(5)}\n")  # [('ITE00100554', 'TMAX', 18.5), ('ITE00100554', 'TMIN', 5.36),...]

# Filter RDD Values
minTempsOnly = parsedLines.filter(lambda x: "TMIN" in x[1])
print(f"type of minTempsOnly: {type(minTempsOnly)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"minTempsOnly: {minTempsOnly.take(5)}\n")  # [('ITE00100554', 'TMIN', 5.36), ('EZE00100082', 'TMIN', 7.7),...]

# Map RDD Values ((str, str, float) -> (str, float))
stationTemps = minTempsOnly.map(lambda x: (x[0], x[2]))
print(f"type of stationTemps: {type(stationTemps)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"stationTemps: {stationTemps.take(5)}\n")  # [('ITE00100554', 5.36), ('EZE00100082', 7.7),...]

# Get Minimum Value Key ((str, float) -> (str, float))
minTemps = stationTemps.reduceByKey(lambda x, y: min(x, y))
print(f"type of minTemps: {type(minTemps)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"minTemps: {minTemps.take(5)}\n")  # [('ITE00100554', 5.36), ('EZE00100082', 7.7)]

# Take Out All Elements in RDD object
results = minTemps.collect()
for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
