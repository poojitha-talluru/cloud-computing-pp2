# Importing necessary libraries
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initializing Spark session
spark = SparkSession.builder.appName("Training").getOrCreate()

# Defining schema for CSV data
wine_schema = StructType([
    StructField("fixed_acidity", DoubleType()),
    StructField("volatile_acidity", DoubleType()),
    StructField("citric_acid", DoubleType()),
    StructField("residual_sugar", DoubleType()),
    StructField("chlorides", DoubleType()),
    StructField("free_sulfur_dioxide", DoubleType()),
    StructField("total_sulfur_dioxide", DoubleType()),
    StructField("density", DoubleType()),
    StructField("pH", DoubleType()),
    StructField("sulphates", DoubleType()),
    StructField("alcohol", DoubleType()),
    StructField("quality", DoubleType())
])

# Reading and processing the dataset
wine_data = spark.read.format("csv").schema(wine_schema).options(header=True, delimiter=';', quote='"', ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True).load('file:///home/ec2-user/TrainingDataset.csv')
wine_data = wine_data.toDF(*[col.replace('"', '') for col in wine_data.columns])
wine_data = wine_data.withColumn("quality", func.when(func.col("quality") > 7, 1).otherwise(0))

# Feature vector preparation
feature_cols = wine_data.columns[:-1]
vectorizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
wine_data = vectorizer.transform(wine_data)

# Splitting the dataset
training, testing = wine_data.randomSplit([0.8, 0.2])

# Model training with Random Forest Classifier
rf_classifier = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=200)
trained_model = rf_classifier.fit(training)

# Model prediction and evaluation
predictions = trained_model.transform(testing)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print("F1 Score: {:.4f}".format(f1))

# Saving the trained model
trained_model.save("file:///home/ec2-user/trainingweights")
