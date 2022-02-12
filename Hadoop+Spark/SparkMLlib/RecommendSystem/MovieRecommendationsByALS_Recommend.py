from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import MatrixFactorizationModel


def PrepareMovieMap(sc):
    # Set a Line in the Text as a RDD Object
    linesItem = sc.textFile("ml-100k/u.item")

    # Parse lines, and get (movieid, moviename)
    moviesRDD = linesItem.map(lambda x: x.split("|")).map(lambda x: (int(x[0]), x[1]))

    # Create a Map between movie id and movie name
    movieMap = moviesRDD.collectAsMap()  # movieMap: {1: 'Toy Story (1995)', 2: 'GoldenEye (1995)',...}
    print("Status: Create movieMap Successfully!")

    return movieMap


def LoadModel(sc):
    global model

    try:
        model = MatrixFactorizationModel.load(sc, "MovieRecommendationsByALS_Model")
        print("Status: Load Model Successfully!")

    except Exception as e:
        print(f"Status: : {e}")

    return model


def Recommend(model, userid=100, num=10):
    recommendMovie2User = model.recommendProducts(userid, num)

    return recommendMovie2User


if __name__ == "__main__":
    # Set Spark Config. (run in local by a cpu and name the job as "MovieRecommendationsByALS_Recommend")
    # Make a SparkContext object, and Depreciate Spark Warning
    conf = SparkConf().setMaster("local").setAppName("MovieRecommendationsByALS_Recommend")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Prepare a Map between movie id and movie name
    print(">>> Preparing a Map between movie id and movie name >>>")
    movieMap = PrepareMovieMap(sc)

    # Load model
    print(">>> Loading model >>>")
    model = LoadModel(sc)

    # Waiting for user input
    print(">>> Waiting for user input >>>")
    userid = int(input("Recommend for userid: "))

    # Start to Recommend
    print(">>> Starting to Recommend >>>")
    recommendMovie2User = Recommend(model, userid)

    # Print out results
    print("Here's the Final Result:")
    for result in recommendMovie2User:
        print(f"Recommend user {result[0]} with a movie {movieMap[result[1]]}, and the rating is {round(result[2], 3)}")
