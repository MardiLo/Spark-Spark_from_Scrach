from pyspark.sql import SparkSession

# Create a SparkSession object and name the job as "SparkSQL"
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# Create a DataFrame object
people = spark.read.option("header", "true").option("inferSchema", "true").csv("fakefriends-header.csv")
print(f"type of people: {type(people)}")  # <class 'pyspark.sql.dataframe.DataFrame'>
print(f"people schema: {people}")  # DataFrame[userID: int, name: string, age: int, friends: int]
print(f"people content: {people.take(5)}\n")  # [Row(ID=0, age=33, name="b'Will'", numFriends=385), Row(ID=1,...),...]

# Find Schema
print("Here is our inferred schema:")
people.printSchema()

# The "Select" Clause
print("Let's display the name column:")
people.select("name").show()

# The "Where" Clause
print("Filter out anyone over 21:")
people.filter(people.age < 21).show()

# The "Group by" Clause
print("Group by age:")
people.groupBy("age").count().show()

# The "Create a New Column" Clause
print("Make everyone 10 years older:")
people.select(people.name, people.age + 10).show()

spark.stop()
