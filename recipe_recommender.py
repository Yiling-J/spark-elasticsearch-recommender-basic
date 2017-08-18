from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql import types

# Install spark 2.1
# To submit this file, first change to bin folder, then run submit with your paramaters:

# spark-submit \
# --driver-class-path <folder>/mysql-connector-java-5.1.42-bin.jar \
# --jars <folder>/mysql-connector-java-5.1.42-bin.jar \
# --master local \
# --executor-cores 4 \
# --executor-memory 8G \
# <folder>/recipe_recommender.py


# Initial
sc = SparkContext("local", "simple")
spark = SparkSession.builder.master("local").appName("simple").getOrCreate()
username = "username"
password = "password"
MYSQL_URL = "jdbc:mysql://localhost:3306/djangodb?user={}&password={}&verifyServerCertificate=false&useSSL=true"

# Laod db
db_url = MYSQL_URL.format(username, password)
db = spark.read.format("jdbc").option("url", db_url)

# Load like table
favs = db.option("dbtable", "UserFav").load()
# Change column name
favs = favs.select('UserId', 'RecipeID').toDF('uid', 'rid')
# Add like rating column(10.0pt)
favs = favs.withColumn('rating', favs.uid * 0 + 10.0)

# Load cookbook and recipe table and join
cookbook = db.option("dbtable", "cookbook").load().select('id', 'owner_id').toDF('cid', 'uid')
cb_recipes = db.option("dbtable", "cookbookrecipe").load().select('cookbook_id', 'recipe_id').toDF('cid', 'rid')
# Add cookbook rating column(7.0pt)
cooks = cb_recipes.join(cookbook, 'cid').drop('cid').withColumn('rating2', cookbook.uid * 0 + 7.0)
cooks = cooks.dropDuplicates(['uid', 'rid'])

# Load click recipe events
click = db.option("dbtable", "events").load()
click = click.filter(click.event_type == 'RecipeTap').filter(click.user_id != 0).select('user_id', 'recipe_id').toDF('uid', 'rid')
click = click.dropDuplicates(['uid', 'rid'])
# Add click rating column(3.0pt)
click = click.withColumn('rating3', click.uid * 0 + 3.0)

# Outer join 3 DataFrame
all = favs.join(cooks, ['uid', 'rid'], 'outer').join(click, ['uid', 'rid'], 'outer').na.fill(0.0)
all = all.dropDuplicates(['uid', 'rid'])

# Get the max rating of each user-recipe pair
all = all.rdd.map(lambda r: (r.uid, r.rid, max(r.rating, r.rating2, r.rating3)))

# Train
model = ALS.train(all, 30, 10)  # 30 factors and 20 iterations

# Root mean squared error
schema = types.StructType(
    [
        types.StructField('uid', types.IntegerType(), False),
        types.StructField('rid', types.IntegerType(), False),
        types.StructField('cof', types.FloatType(), False)
    ]
)
all = spark.createDataFrame(all, schema)
testData = all.rdd.map(lambda p: (p.uid, p.rid))
predictions = model.predictAll(testData).map(lambda r: ((r.user, r.product), r.rating))
ratingsTuple = all.rdd.map(lambda r: ((r.uid, r.rid), r.cof))
scoreAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])
metrics = RegressionMetrics(scoreAndLabels)
print("RMSE = %s" % metrics.rootMeanSquaredError)

def save_facotr_to_database(factor_rdd, table):
    # Map vector to string, can be store to database
    factor_rdd = factor_rdd.map(lambda r: (r[0], ','.join(map(str, r[1]))))
    # Build schema
    schema = types.StructType([types.StructField('id', types.IntegerType(), False), types.StructField('vector', types.StringType(), True)])
    # RDD to DF
    factor_df = spark.createDataFrame(factor_rdd, schema)
    # Write back to database table(overwrite)
    factor_df.write.mode("overwrite").format("jdbc").\
        option("url", "jdbc:mysql://localhost:3306/djangodb").\
        option("dbtable", table).\
        option("user", username).\
        option("password", password).\
        save()

save_facotr_to_database(model.userFeatures(), "user_factor")

save_facotr_to_database(model.productFeatures(), "recipe_factor")


# Recommendations top N recipes for all user, also RDD
# model.recommendProductsForUsers(num)
