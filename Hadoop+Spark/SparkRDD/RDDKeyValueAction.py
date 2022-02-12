from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "RDDKeyValueAction")
conf = SparkConf().setMaster("local").setAppName("RDDKeyValueAction")

# Main entry point for Spark functionality
sc = SparkContext(conf=conf)

# Distribute a local Python collection to form an RDD
# Using xrange is recommended if the input represents a range for performance
print("Create a RDD:")
KVRDD = sc.parallelize([(3, 3), (3, 6), (5, 6), (1, 2), (5, 18)])
print(f"type of KVRDD: {type(KVRDD)}")
print(f"KVRDD: {KVRDD.collect()}\n")

# Return the first element in this RDD
print(f"First Element in KVRDD: {KVRDD.first()}")

# Take the first num elements of the RDD
print(f"First 5 Elements in KVRDD: {KVRDD.take(5)}")

# Count the number of elements for each key, and return the result to the master as a dictionary
print(f"Count KVRDD by Key: {sorted(KVRDD.countByKey().items())}")

# Count the number of elements for each value, and return the result to the master as a dictionary
print(f"Count KVRDD by Value: {sorted(KVRDD.countByValue().items())}")

# Return the key-value pairs in this RDD to the master as a dictionary
KVRDDColAsMap = KVRDD.collectAsMap()
print(f"Make a Map of KVRDD: 3 -> {KVRDDColAsMap[3]}, 5 -> {KVRDDColAsMap[5]}, 1 -> {KVRDDColAsMap[1]}")

# Return the list of values in the RDD for key key.
# This operation is done efficiently if the RDD has a known partitioner
# by only searching the partition that the key maps to
print(f"LookUp KVRDD's Value by a Given Key: 3 -> {KVRDD.lookup(3)}, 5 -> {KVRDD.lookup(5)}, 1 -> {KVRDD.lookup(1)}")
