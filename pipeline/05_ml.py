from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, Word2Vec
from pyspark.ml.recommendation import ALS
import mlflow
import mlflow.spark

BUCKET = "s3a://clickstream-analytics-akash"

spark = SparkSession.builder.appName("05-ML").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

mlflow.set_tracking_uri("file:///home/ubuntu/mlruns")
mlflow.set_experiment("clickstream-lakehouse-models")

print("=== Loading Bronze datasets ===")
rees46 = spark.read.parquet(f"{BUCKET}/bronze/rees46/")
criteo = spark.read.parquet(f"{BUCKET}/bronze/criteo/")

print("=== CTR model on Criteo schema ===")
with mlflow.start_run(run_name="ctr_model_gbt"):
    numeric_cols = [c for c in criteo.columns if c.startswith("I")]
    ctr_df = criteo.select(["label"] + numeric_cols).na.fill(0)
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    model_df = assembler.transform(ctr_df).select("label", "features")
    train, test = model_df.randomSplit([0.8, 0.2], seed=42)
    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=20, maxDepth=5, seed=42)
    model = gbt.fit(train)
    preds = model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(preds)
    print(f"CTR AUC-ROC: {auc}")
    mlflow.log_metric("auc_roc", auc)
    mlflow.spark.log_model(model, "ctr_model")

print("=== ALS recommender on REES46 implicit feedback ===")
with mlflow.start_run(run_name="als_recommender"):
    interactions = (
        rees46.select("user_id", "product_id", "event_type")
        .dropna(subset=["user_id", "product_id"])
        .withColumn("rating", F.when(F.col("event_type") == "purchase", 5.0).when(F.col("event_type") == "cart", 3.0).otherwise(1.0))
        .groupBy("user_id", "product_id")
        .agg(F.sum("rating").alias("rating"))
        .withColumn("user_id_int", F.pmod(F.col("user_id"), F.lit(2_000_000)).cast("int"))
        .withColumn("product_id_int", F.pmod(F.col("product_id"), F.lit(2_000_000)).cast("int"))
        .limit(5_000_000)
    )
    train, test = interactions.randomSplit([0.8, 0.2], seed=42)
    als = ALS(userCol="user_id_int", itemCol="product_id_int", ratingCol="rating", implicitPrefs=True, coldStartStrategy="drop", rank=20, maxIter=10, regParam=0.1)
    model = als.fit(train)
    preds = model.transform(test)
    rmse = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse").evaluate(preds)
    print(f"ALS RMSE proxy: {rmse}")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("precision_at_k_proxy", max(0.0, 1.0 / (1.0 + rmse)))
    mlflow.spark.log_model(model, "als_model")

print("=== Product2Vec from user sessions ===")
with mlflow.start_run(run_name="product2vec"):
    sessions = (
        rees46.dropna(subset=["user_session", "product_id"])
        .groupBy("user_session")
        .agg(F.collect_list(F.col("product_id").cast("string")).alias("items"))
        .filter(F.size("items") >= 2)
        .limit(2_000_000)
    )
    w2v = Word2Vec(vectorSize=32, minCount=5, inputCol="items", outputCol="embedding", maxIter=5, seed=42)
    model = w2v.fit(sessions)
    vectors = model.getVectors()
    print(f"Product2Vec vocabulary size: {vectors.count()}")
    mlflow.log_metric("vocab_size", vectors.count())
    mlflow.spark.log_model(model, "product2vec")
    vectors.write.mode("overwrite").parquet(f"{BUCKET}/gold/ml-artifacts/product2vec/vectors/")

print("=== KMeans customer segmentation ===")
with mlflow.start_run(run_name="kmeans_segments"):
    users = (
        rees46.groupBy("user_id")
        .agg(
            F.count("*").alias("total_events"),
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("views"),
            F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("carts"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
            F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0.0)).alias("spend"),
        )
        .na.fill(0)
    )
    features = ["total_events", "views", "carts", "purchases", "spend"]
    assembler = VectorAssembler(inputCols=features, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    kmeans = KMeans(k=4, seed=42, featuresCol="features", predictionCol="cluster")
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    model = pipeline.fit(users)
    preds = model.transform(users)
    silhouette = ClusteringEvaluator(featuresCol="features", predictionCol="cluster").evaluate(preds)
    print(f"KMeans silhouette: {silhouette}")
    mlflow.log_metric("silhouette", silhouette)
    mlflow.log_metric("clusters", 4)
    mlflow.spark.log_model(model, "kmeans")
    preds.write.mode("overwrite").parquet(f"{BUCKET}/gold/ml-artifacts/kmeans/segments/")

print("ML training complete")
spark.stop()
