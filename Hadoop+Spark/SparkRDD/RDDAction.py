from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "RDDAction")
conf = SparkConf().setMaster("local").setAppName("RDDAction")

# Main entry point for Spark functionality
sc = SparkContext(conf=conf)

# Distribute a local Python collection to form an RDD
# Using xrange is recommended if the input represents a range for performance
IntRDD = sc.parallelize([1, 2, 3, 4, 5, 5, 6, 7, 8, 8])

# Return the first element in this RDD
print(f"First Element in IntRDD: {IntRDD.first()}")

# Take the first num elements of the RDD
print(f"First 5 Elements in IntRDD: {IntRDD.take(5)}")

# Get the N elements from an RDD ordered in descending order or as specified by the optional key function
print(f"First 5 Elements in IntRDD in descending Order: {IntRDD.takeOrdered(5, key=lambda x: -x)}")

# Return a StatCounter object that captures the mean, variance and count of the RDD’s elements in one operation
print(f"IntRDD Stats: {IntRDD.stats()}")
