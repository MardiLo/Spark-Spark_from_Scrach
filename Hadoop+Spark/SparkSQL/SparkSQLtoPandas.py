from pyspark.sql import SparkSession, Row
import pandas as pd
import matplotlib.pyplot as plt

# Create a Row object (str -> Row(str, str, str, str))
def zipMapper(line):
    fields = line.replace("\"", "").split(",")
    return Row(zipcode=fields[1],
               zipCodeType=fields[2],
               city=fields[3],
               state=fields[4])


# Create a Row object (str -> Row(int, int, str, str, str))
def userMapper(line):
    fields = line.split('|')
    return Row(userid=int(fields[0]),
               age=int(fields[1]),
               gender=fields[2],
               occupation=fields[3],
               zipcode=fields[4])


# Create a SparkSession object and name the job as "SparkSQLtoPandasDataFrame"
sparkSQL = SparkSession.builder.appName("SparkSQLtoPandasDataFrame").getOrCreate()

# Set a Line in the Text as a RDD Object
zip_lines = sparkSQL.sparkContext.textFile("free-zipcode-database.csv", 5)

# Save Header
header = zip_lines.first()

# Separate body and clean it
zipcodeData = zip_lines.filter(lambda x: x != header)
zipRDD = zipcodeData.map(zipMapper)

# Convert this PipelinedRDD into DataFrame
zipcode_df = sparkSQL.createDataFrame(zipRDD)

# Register the DataFrame as a Table called "zipcode"
zipcode_df.registerTempTable("zipcode")

# Set a Line in the Text as a RDD Object
user_lines = sparkSQL.sparkContext.textFile("ml-100k/u.user", 5)

# Create a PipelinedRDD contains a lot of Row objects
user_Rows = user_lines.map(userMapper)

# Convert this PipelinedRDD into DataFrame
user_df = sparkSQL.createDataFrame(user_Rows)

# Register the DataFrame as a Table called "user"
user_df.registerTempTable("user")

# Join in SparkSQL
join_df = user_df.join(zipcode_df, user_df.zipcode == zipcode_df.zipcode, "left_outer")
groupByState = join_df.groupBy("state").count()

# SparkSQL to pandas DataFrame
groupByState_df = groupByState.toPandas().set_index('state')
print(f"groupByState_df:\n {groupByState_df.head(5)}")

# Plot with matplotlib
ax = groupByState_df['count'].plot(kind='bar',
                                   title='State',
                                   figsize=(12, 6),
                                   legend=True,
                                   fontsize=12)
plt.show()
