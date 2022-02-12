from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "RDDKeyValueTransform")
conf = SparkConf().setMaster("local").setAppName("RDDKeyValueTransform")

# Main entry point for Spark functionality
sc = SparkContext(conf=conf)

# Distribute a local Python collection to form an RDD
# Using xrange is recommended if the input represents a range for performance
print("Create a RDD:")
KVRDD = sc.parallelize([(3, 3), (3, 6), (5, 6), (1, 2), (5, 18)])
print(f"type of KVRDD: {type(KVRDD)}")
print(f"KVRDD: {KVRDD.collect()}\n")

# Return an RDD with the keys/values of each tuple
print("RDD's Keys and Values:")
print(f"KVRDD's keys: {KVRDD.keys().collect()}")
print(f"KVRDD's values: {KVRDD.values().collect()}\n")

# Return a new RDD containing only the elements that satisfy a predicate
print("Filter with RDD's Keys and Values:")
print(f"Elements in KVRDD's Keys less than 5: {KVRDD.filter(lambda x: x[0] < 5).collect()}")
print(f"Elements in KVRDD's Values less than 5: {KVRDD.filter(lambda x: x[1] < 5).collect()}\n")

# Pass each value in the key-value pair RDD through a map function without changing the keys;
# this also retains the original RDD’s partitioning
print("Map RDD Values:")
print(f"Get Square of RDD Values: {KVRDD.mapValues(lambda x: x ** 2).collect()}\n")

# Sorts this RDD, which is assumed to consist of (key, value) pairs
print("Sort RDD by its Keys:")
print(f"Sort RDD by its Keys in Ascending Order: {KVRDD.sortByKey().collect()}")
print(f"Sort RDD by its Keys in Descending Order: {KVRDD.sortByKey(False).collect()}\n")

# Merge the values for each key using an associative and commutative reduce function
print("ReduceByKey:")
print(f"Add Values by Same Keys: {KVRDD.reduceByKey(lambda x, y: x + y).collect()}")
print(f"Minus Values by Same Keys: {KVRDD.reduceByKey(lambda x, y: x - y).collect()}\n")

# Multiple RDDs' Transformation
KVRDD1 = sc.parallelize([(3, 4), (3, 6), (5, 6), (1, 2)])
KVRDD2 = sc.parallelize([(3, 8)])

# Return an RDD containing all pairs of elements with matching keys in self and other
print("Join:")
print(f"Join KVRDD1 and KVRDD2: {KVRDD1.join(KVRDD2).collect()}\n")

# Perform a left outer join of self and other
print("Left Outer Join:")
print(f"KVRDD1 Left Outer Join KVRDD2: {KVRDD1.leftOuterJoin(KVRDD2).collect()}\n")

# Perform a right outer join of self and other
print("Right Outer Join:")
print(f"KVRDD1 Right Outer Join KVRDD2: {KVRDD1.rightOuterJoin(KVRDD2).collect()}\n")

# Return each (key, value) pair in self that has no pair with matching key in other
print("SubtractByKey:")
print(f"KVRDD1 Subtracts KVRDD2 By Key: {KVRDD1.subtractByKey(KVRDD2).collect()}\n")