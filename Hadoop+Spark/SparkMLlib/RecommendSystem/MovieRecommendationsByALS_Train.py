from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
import shutil


def PrepareData(sc):
    # Set a Line in the Text as a RDD Object
    linesData = sc.textFile("ml-100k/u.data")

    # Parse lines, and get (userid, movieid, ratings)
    ratingsRDD = linesData.map(lambda x: x.split("\t")[:3]).map(lambda x: (x[0], x[1], x[2]))
    print("Status: Prepare Data Successfully!")

    return ratingsRDD


def SaveModel(model, sc):
    try:
        shutil.rmtree("MovieRecommendationsByALS_Model", ignore_errors=True)
        model.save(sc, "MovieRecommendationsByALS_Model")
        print("Status: Model Saved Successfully!")

    except Exception as e:
        print(f"Status: : {e}")


if __name__ == "__main__":
    # Set Spark Config. (run in local by a cpu and name the job as "MovieRecommendationsByALS_Train")
    # Make a SparkContext object, and Depreciate Spark Warning
    conf = SparkConf().setMaster("local").setAppName("MovieRecommendationsByALS_Train")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Prepare Raw Data
    print(">>> Preparing Raw Data >>>")
    ratingsRDD = PrepareData(sc)

    # Train model
    print(">>> Training model >>>")
    model = ALS.train(ratingsRDD, rank=10, iterations=10, lambda_=0.01)
    print("Status: Train Model Successfully!")

    # Save Model
    print(">>> Saving model >>>")
    SaveModel(model, sc)
