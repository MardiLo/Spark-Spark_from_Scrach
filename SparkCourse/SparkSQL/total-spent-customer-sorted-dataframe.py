from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Create a SparkSession object and name the job as "TotalSpentByCustomer" and use all cpus
spark = SparkSession.builder.appName("TotalSpentByCustomer").master("local[*]").getOrCreate()

# Create schema when reading customer-orders
customerOrderSchema = StructType([StructField("customerID", StringType(), True),
                                  StructField("itemID", StringType(), True),
                                  StructField("amountSpent", FloatType(), True)])

# Load up the data into spark dataset
customersDF = spark.read.schema(customerOrderSchema).csv("customer-orders.csv")

customersDF = customersDF.select("customerID", "amountSpent")
customersDF = customersDF.groupBy("customerID").agg(func.round(func.sum("amountSpent"), 2).alias("totalSpent")).sort("totalSpent")
customersDF.show(customersDF.count())

# spark.stop()
