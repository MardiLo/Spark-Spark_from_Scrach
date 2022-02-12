from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "MaxTemperatures")
conf = SparkConf().setMaster("local").setAppName("MaxTemperatures")

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
maxTempsOnly = parsedLines.filter(lambda x: "TMAX" in x[1])
print(f"type of maxTempsOnly: {type(maxTempsOnly)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"maxTempsOnly: {maxTempsOnly.take(5)}\n")  # [('ITE00100554', 'TMAX', 18.5), ('EZE00100082', 'TMAX', 16.52),...]

# Map RDD Values ((str, str, float) -> (str, float))
stationTemps = maxTempsOnly.map(lambda x: (x[0], x[2]))
print(f"type of stationTemps: {type(stationTemps)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"stationTemps: {stationTemps.take(5)}\n")  # [('ITE00100554', 18.5), ('EZE00100082', 16.52),...]

# Get Maximum Value Key ((str, float) -> (str, float))
maxTemps = stationTemps.reduceByKey(lambda x, y: max(x, y))
print(f"type of maxTemps: {type(maxTemps)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"maxTemps: {maxTemps.take(5)}\n")  # [('ITE00100554', 90.14), ('EZE00100082', 90.14)]

# Take Out All Elements in RDD object
results = maxTemps.collect()
for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
