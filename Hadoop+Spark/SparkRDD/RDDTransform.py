from pyspark import SparkConf, SparkContext

# Set Spark Config. (run in local by a cpu and name the job as "RDDTransform")
conf = SparkConf().setMaster("local").setAppName("RDDTransform")

# Main entry point for Spark functionality
sc = SparkContext(conf=conf)

# Distribute a local Python collection to form an RDD
# Using xrange is recommended if the input represents a range for performance
print("Create a RDD:")
IntRDD = sc.parallelize([1, 2, 3, 4, 5, 5, 6, 7, 8, 8])
print(f"type of IntRDD: {type(IntRDD)}")
print(f"IntRDD: {IntRDD.collect()}")
print(f"type of IntRDD.collect(): {type(IntRDD.collect())}\n")

# Map RDD Values (int -> int), note that map is a transform, and collect() is an action
print("Map RDD:")

# Map with Lambda
IntRDDPlusOneByLambda = IntRDD.map(lambda x: x + 1)
print(f"IntRDDPlusOne: {IntRDDPlusOneByLambda.collect()}")


# Map with Function
def prefixwithtest(x: int) -> str:
    return 'test_' + str(x)


StrRDDByDef = IntRDD.map(prefixwithtest)
print(f"StrRDDByDef: {StrRDDByDef.collect()}\n")

# Return a new RDD containing only the elements that satisfy a predicate
print("Get Elements Greater Than 2 in IntRDD:")
IntRDDGreaterThanTwo = IntRDD.filter(lambda x: x > 2)
print(f"IntRDDGreaterThanTwo: {IntRDDGreaterThanTwo.collect()}\n")

# Return a new RDD containing the distinct elements in this RDD
print("Distinct Elements in IntRDD:")
DistinctIntRDD = IntRDD.distinct()
print(f"DistinctIntRDD: {DistinctIntRDD.collect()}\n")

# Randomly splits this RDD with the provided weights
print("Randomly Split Elements in IntRDD:")
RDD1, RDD2 = IntRDD.randomSplit(weights=[0.2, 0.8], seed=17)
print(f"RDD1: {RDD1.collect()}")
print(f"RDD2: {RDD2.collect()}\n")

# Return an RDD of grouped items
print("Group RDD Elements By Condition:")
OddOrEvenIntRDD = IntRDD.groupBy(lambda x: 'even' if x % 2 == 0 else 'odd')
OddIntRDD = sorted(OddOrEvenIntRDD.collect()[0][1])
EvenIntRDD = sorted(OddOrEvenIntRDD.collect()[1][1])
print(f"OddIntRDD: {OddIntRDD}")
print(f"EvenIntRDD: {EvenIntRDD}\n")

# Multiple RDDs' Transformation
IntRDD1 = sc.parallelize([3, 1, 2, 5, 5])
IntRDD2 = sc.parallelize([5, 6])
IntRDD3 = sc.parallelize([2, 7])

# Return the union of this RDD and another one
print("RDDs Union:")
UnionRDD = IntRDD1.union(IntRDD2).union(IntRDD3)
print(f"UnionRDD: {UnionRDD.collect()}\n")

# Return the intersection of this RDD and another one
# The output will not contain any duplicate elements, even if the input RDDs did
print("RDDs Intersection:")
IntersectionRDD = IntRDD1.intersection(IntRDD2)
print(f"IntersectionRDD: {IntersectionRDD.collect()}\n")

# Return each value in self that is not contained in other
print("RDDs Subtract:")
SubtractRDD = IntRDD1.intersection(IntRDD3)
print(f"SubtractRDD: {SubtractRDD.collect()}\n")

# Return the Cartesian product of this RDD and another one,
# that is, the RDD of all pairs of elements (a, b) where a is in self and b is in other
print("RDDs Cartesian:")
CartesianRDD = IntRDD2.cartesian(IntRDD3)
print(f"CartesianRDD: {CartesianRDD.collect()}\n")
