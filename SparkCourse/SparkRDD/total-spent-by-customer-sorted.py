from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "SpendByCustomer")
conf = SparkConf().setMaster("local").setAppName("SpendByCustomer")

# Make a SparkContext object
sc = SparkContext(conf=conf)


# Define a Parse Function (str -> (int, float))
def extractCustomerPricePairs(line):
    fields = line.split(',')
    return (int(fields[0]), float(fields[2]))


# Set a Line in the Text as a RDD Object
lines = sc.textFile("customer-orders.csv")
print(f"type of lines: {type(lines)}\n")  # <class 'pyspark.rdd.RDD'>

# Map RDD Values
mappedInput = lines.map(extractCustomerPricePairs)
print(f"type of mappedInput: {type(mappedInput)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"mappedInput: {mappedInput.take(5)}\n")  # [(44, 37.19), (35, 65.89),...]

# Sum RDD up by Key
totalByCustomer = mappedInput.reduceByKey(lambda x, y: x + y)
print(f"type of totalByCustomer: {type(totalByCustomer)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"totalByCustomer: {totalByCustomer.take(5)}\n")  # [(44, 4756.8899999999985), (35, 5155.419999999999),...]

# Switch Key and Value (Key, Value -> Value, Key), and then Sort it by Key (which now is Value)
totalByCustomerSorted = totalByCustomer.map(lambda x: (x[1], x[0])).sortByKey()
print(f"type of totalByCustomerSorted: {type(totalByCustomerSorted)}")  # <class 'pyspark.rdd.PipelinedRDD'>
print(f"totalByCustomerSorted: {totalByCustomerSorted.take(5)}\n")  # [(3309.38, 45), (3790.570000000001, 79),...]

# Take Out All Elements in RDD object
results = totalByCustomerSorted.collect()
for result in results:
    print(f"customer {result[1]} has spent {result[0]:.2f}.")
