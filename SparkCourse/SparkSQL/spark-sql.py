from pyspark.sql import SparkSession, Row

# Create a SparkSession object and name the job as "SparkSQL"
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()


# Create a Row object (str -> Row(int, str, int, int))
def mapper(line):
    fields = line.split(',')
    return Row(ID=int(fields[0]),
               name=str(fields[1].encode("utf-8")),
               age=int(fields[2]),
               numFriends=int(fields[3]))


# Set a Line in the Text as a RDD Object
lines = spark.sparkContext.textFile("fakefriends.csv")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Create a lot of Row objects
people = lines.map(mapper)
print(f"type of people: {type(people)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"people: {people.take(5)}\n")  # [Row(ID=0, age=33, name="b'Will'", numFriends=385), Row(ID=1,...),...]

# Convert RDD into DataFrame
schemaPeople = spark.createDataFrame(people).cache()
print(f"type of schemaPeople: {type(schemaPeople)}")  # <class 'pyspark.sql.dataframe.DataFrame'>
print(f"schemaPeople schema: {schemaPeople}\n")  # DataFrame[ID: bigint, age: bigint, name: string, numFriends: bigint]

# Register the DataFrame as a Table called "people"
schemaPeople.createOrReplaceTempView("people")

# SQL can be run over DataFrames that have been registered as a table
teenagers = spark.sql("SELECT age, count(1) FROM people GROUP BY age ORDER BY age")
print(f"type of teenagers: {type(teenagers)}")  # <class 'pyspark.sql.dataframe.DataFrame'>
print(f"teenagers schema: {teenagers}\n")  # DataFrame[age: bigint, count(1): bigint]

# The results of SQL queries are RDDs and support all the normal RDD operations
for teen in teenagers.collect():
    print(teen)
print()

# Or we could just print it out:
teenagers.show(10)
print()

# We can also use functions instead of SQL queries:
schemaPeople.groupBy("age").count().orderBy("age").show(10)

# Stop a SparkSession
# spark.stop()
