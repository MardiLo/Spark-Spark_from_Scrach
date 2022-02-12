from pyspark.sql import SparkSession, Row


# Create a Row object (str -> Row(str, str, str, str))
def zipMapper(line):
    fields = line.replace("\"", "").split(",")
    return Row(zipcode=fields[0],
               zipCodeType=fields[1],
               city=fields[2],
               state=fields[3])


# Create a Row object (str -> Row(int, int, str, str, str))
def userMapper(line):
    fields = line.split('|')
    return Row(userid=int(fields[0]),
               age=int(fields[1]),
               gender=fields[2],
               occupation=fields[3],
               zipcode=fields[4])


# Create a SparkSession object and name the job as "SparkSQLJoin"
sparkSQL = SparkSession.builder.appName("SparkSQLJoin").getOrCreate()

# Set a Line in the Text as a RDD Object
zip_lines = sparkSQL.sparkContext.textFile("free-zipcode-database.csv", 5)

# Save Header
header = zip_lines.first()

# Separate body and clean it
zipcodeData = zip_lines.filter(lambda x: x != header)
zipRDD = zipcodeData.map(zipMapper)  # [Row(city='STANDARD', state='PARC PARQUE', zipCode=1, zipCodeType='00704'),...]

# Convert this PipelinedRDD into DataFrame
zipcode_df = sparkSQL.createDataFrame(zipRDD)
print(f"zipcode_df schema:")
zipcode_df.printSchema()

# Register the DataFrame as a Table called "zipcode"
zipcode_df.registerTempTable("zipcode")

# Set a Line in the Text as a RDD Object
user_lines = sparkSQL.sparkContext.textFile("ml-100k/u.user", 5)

# Create a PipelinedRDD contains a lot of Row objects
user_Rows = user_lines.map(userMapper)  # [Row(age=24, gender='M', occupation='technician', userid=1, zipcode='85711'),...]

# Convert this PipelinedRDD into DataFrame
user_df = sparkSQL.createDataFrame(user_Rows)
print(f"user_df schema:")
user_df.printSchema()

# Register the DataFrame as a Table called "user"
user_df.registerTempTable("user")

# Join in SparkSQL
print(f"Join in SparkSQL:")
join_df = user_df.join(zipcode_df, user_df.zipcode == zipcode_df.zipcode, "left_outer")
join_df.printSchema()
join_df.show(10)

