from pyspark import SparkConf, SparkContext, StorageLevel

# Set Spark Config. (run in local by a cpu and name the job as "SharedVariable")
conf = SparkConf().setMaster("local").setAppName("SharedVariable")

# Main entry point for Spark functionality
sc = SparkContext(conf=conf)

# A broadcast variable that gets reused across tasks
print("Get FruitNames by Broadcast:")
KVFruit = sc.parallelize([(1, 'apple'), (2, 'orange'), (3, 'banana'), (4, 'grape')])
FruitMap = KVFruit.collectAsMap()
BCFruitMap = sc.broadcast(FruitMap)

FruitIds = sc.parallelize([1, 2, 3, 4])
FruitNames = FruitIds.map(lambda x: BCFruitMap.value[x]).collect()
print(f"FruitNames: {FruitNames}\n")

# An “add-only” shared variable that tasks can only add values to
print("Sum up IntRDD by Accumulator:")
IntRDD = sc.parallelize([1, 2, 3, 5, 5])

total = sc.accumulator(0.0)
num = sc.accumulator(0)

IntRDD.foreach(lambda i: [total.add(i), num.add(1)])
print(f"total = {total.value}, num = {num.value}\n")

# Set this RDD’s storage level to persist its values across operations after the first time it is computed.
# This can only be used to assign a new storage level if the RDD does not have a storage level set yet.
# If no storage level is specified defaults to (MEMORY_ONLY)
print("Persist RDD Object:")
print(f"Persist: {IntRDD.persist(StorageLevel.MEMORY_ONLY).is_cached}")
print(f"Unpersist: {IntRDD.unpersist().is_cached}")
