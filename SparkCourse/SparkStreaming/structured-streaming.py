# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:15:05 2019

@author: Frank
"""

from pyspark.sql import SparkSession

from pyspark.sql.functions import regexp_extract

# Create a SparkSession (the config bit is only for Windows!)
spark = SparkSession.builder.appName("StructuredStreaming").getOrCreate()
print(f"spark: {type(spark)}\n")  # <class 'pyspark.sql.session.SparkSession'>

# Monitor the logs directory for new log data, and read in the raw lines as accessLines
accessLines = spark.readStream.text("logs")
print(f"accessLines: {type(accessLines)}\n")  # <class 'pyspark.sql.dataframe.DataFrame'>

# Parse out the common log format to a DataFrame
contentSizeExp = r'\s(\d+)$'
statusExp = r'\s(\d{3})\s'  # 200
generalExp = r'\"(\S+)\s(\S+)\s*(\S*)\"'  # [GET, /robots.txt, HTTP/1.1]
timeExp = r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} \+\d{4})]'  # [29/Nov/2015:03:50:05 +0000]
hostExp = r'(^\S+\.[\S+\.]+\S+)\s'  # 66.249.75.159

logsDF = accessLines.select(regexp_extract('value', hostExp, 1).alias('host'),
                            regexp_extract('value', timeExp, 1).alias('timestamp'),
                            regexp_extract('value', generalExp, 1).alias('method'),
                            regexp_extract('value', generalExp, 2).alias('endpoint'),
                            regexp_extract('value', generalExp, 3).alias('protocol'),
                            regexp_extract('value', statusExp, 1).cast('integer').alias('status'),
                            regexp_extract('value', contentSizeExp, 1).cast('integer').alias('content_size'))

print(f"logsDF: {type(logsDF)}\n")  # <class 'pyspark.sql.dataframe.DataFrame'>

# Keep a running count of every access by status code
statusCountsDF = logsDF.groupBy(logsDF.status).count()  # <class 'pyspark.sql.dataframe.DataFrame'>

# Kick off our streaming query, dumping results to the console
query = (statusCountsDF.writeStream.outputMode("complete").format("console").queryName("counts").start())
print(f"query: {type(query)}\n")  # <class 'pyspark.sql.streaming.StreamingQuery'>

# Run forever until terminated
query.awaitTermination()

# Cleanly shut down the session
# spark.stop()
