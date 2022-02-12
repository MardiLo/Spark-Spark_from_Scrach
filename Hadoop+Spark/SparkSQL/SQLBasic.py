from pyspark.sql import SparkSession, Row

# Create a SparkSession object and name the job as "SparkSQLBasic"
sparkSQL = SparkSession.builder.appName("SparkSQLBasic").getOrCreate()


# Create a Row object (str -> Row(int, str, int, int))
def mapper(line):
    fields = line.split('|')
    return Row(userid=int(fields[0]),
               age=int(fields[1]),
               gender=fields[2],
               occupation=fields[3],
               zipcode=fields[4])


# Set a Line in the Text as a RDD Object
lines = sparkSQL.sparkContext.textFile("ml-100k/u.user")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Create a PipelinedRDD contains a lot of Row objects
user_Rows = lines.map(mapper)
print(f"type of user_Rows: {type(user_Rows)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"user_Rows: {user_Rows.first()}\n")  # [Row(age=24, gender='M', occupation='technician', userid=1, zipcode='85711'),...]

# Convert this PipelinedRDD into DataFrame
user_df = sparkSQL.createDataFrame(user_Rows).cache()
print(f"type of user_df: {type(user_df)}")  # <class 'pyspark.sql.dataframe.DataFrame'>
print(f"user_df schema:")
user_df.printSchema()

# Register the DataFrame as a Table called "user"
user_df.registerTempTable("user")

# Select columns in SparkSQL
# By SQL Query
print(f"Select columns in SparkSQL:")
q = '''
    SELECT userid, occupation, gender, age
    FROM user 
    '''
sparkSQL.sql(q).show(5)

# By SparkSQL Method: select
user_df.select("userid", "occupation", "gender", "age").show(5)

# Add columns in SparkSQL
# By SQL Query
print(f"Add columns in SparkSQL:")
q = '''
    SELECT userid, occupation, gender, age, 2016-age AS birthyear
    FROM user 
    '''
sparkSQL.sql(q).show(5)

# By SparkSQL Method: select
user_df.select("userid", "occupation", "gender", "age", (2016 - user_df.age).alias("birthyear")).show(5)

# Filter data in SparkSQL
# By SQL Query
print(f"Filter data in SparkSQL:")
q = '''
    SELECT *
    FROM user 
    WHERE occupation='technician' AND gender='M' AND age=24
    '''
sparkSQL.sql(q).show(5)

# By SparkSQL Method: filter
user_df.filter((user_df.occupation == 'technician') & (user_df.gender == 'M') & (user_df.age == 24)).show(5)

# Sort data in SparkSQL
# By SQL Query
print(f"Sort data in SparkSQL:")
q = '''
    SELECT userid, occupation, gender, age
    FROM user 
    ORDER BY age DESC, gender
    '''
sparkSQL.sql(q).show(5)

# By SparkSQL Method: orderBy
user_df.select("userid", "occupation", "gender", "age").orderBy(["age", "gender"], ascending=[0, 1]).show(5)

# Find unique data in SparkSQL
# By SQL Query
print(f"Find unique data in SparkSQL:")
q = '''
    SELECT distinct gender, age
    FROM user 
    '''
sparkSQL.sql(q).show(5)

# By SparkSQL Method: distinct
user_df.select("gender", "age").distinct().show(5)

# Count data by Groups in SparkSQL
# By SQL Query
print(f"Count data by Groups in SparkSQL:")
q = '''
    SELECT gender, occupation, count(1) AS counts
    FROM user 
    GROUP BY gender, occupation
    '''
sparkSQL.sql(q).show(5)

# By SparkSQL Method: groupBy
user_df.select("gender", "occupation").groupBy("gender", "occupation").count().show(5)

# Crosstab analysis
user_df.stat.crosstab("occupation", "gender").show(5)