from pyspark.sql import SparkSession

# Create a SparkSession object and name the job as "SparkSQL"
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# Create a DataFrame object
df = spark.read.option("header", "true").option("inferSchema", "true").csv("fakefriends-header.csv")
print(f"type of df: {type(df)}")  # <class 'pyspark.sql.dataframe.DataFrame'>
print(f"df schema: {df}")  # DataFrame[userID: int, name: string, age: int, friends: int]
print(f"df content: {df.take(5)}\n")  # [Row(ID=0, age=33, name="b'Will'", numFriends=385), Row(ID=1,...),...]

# Register the DataFrame as a Table called "friends"
df.createOrReplaceTempView("friends")

# The "Select" Clause
print("Let's display the name column:")
people = spark.sql("SELECT name FROM friends")
people.show()

# The "Where" Clause
print("Filter out anyone over 21:")
people = spark.sql("SELECT * FROM friends WHERE age < 21")
people.show()

# The "Group by" Clause
print("Group by age:")
people = spark.sql("SELECT age, count(1) AS count FROM friends GROUP BY age")
people.show()

# The "Create a New Column" Clause
print("Make everyone 10 years older:")
people = spark.sql("SELECT name, age + 10 FROM friends")
people.show()

# spark.stop()
