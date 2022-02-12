from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Create a SparkSession object and name the job as "MostPopularSuperhero"
spark = SparkSession.builder.appName("MostPopularSuperhero").getOrCreate()

# Create schema when reading Marvel-Names.txt
schema = StructType([StructField("id", IntegerType(), True),
                     StructField("name", StringType(), True)])

# Create a "name" DataFrame
names = spark.read.schema(schema).option("sep", " ").csv("Marvel-Names.txt")

# Create a "lines" DataFrame
lines = spark.read.text("Marvel-Graph.txt")

# Find the hero who has most connection
graph = lines.withColumn("id", func.split(func.col("value"), " ")[0])
graph = graph.withColumn("connections", func.size(func.split(func.col("value"), " ")) - 1)
graph = graph.groupBy("id").agg(func.sum("connections").alias("connections"))
maxConnectionCount = graph.agg(func.max("connections")).first()[0]
graph = graph.filter(func.col("connections") == maxConnectionCount)
maxConnectionsWithNames = graph.join(names, "id").select("name").first()[0]
print(maxConnectionsWithNames + " is the most popular superhero with " + str(maxConnectionCount) + " co-appearances.")
