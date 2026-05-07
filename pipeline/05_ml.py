from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, Word2Vec
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
import mlflow
import json
import boto3

BUCKET = "s3a://clickstream-analytics-akash"
S3_BUCKET = "clickstream-analytics-akash"

spark = (
    SparkSession.builder
    .appName("05-ML")
    .config("spark.kryoserializer.buffer", "64m")
    .config("spark.kryoserializer.buffer.max", "1024m")
    .config("spark.sql.shuffle.partitions", "300")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

mlflow.set_tracking_uri("file:/home/ubuntu/mlruns")
mlflow.set_experiment("clickstream-analytics-ml")

metrics = {}

print("=== 1. CTR Prediction (Criteo) ===")
criteo = spark.read.parquet(f"{BUCKET}/bronze/criteo/")
numeric_cols = [c for c in criteo.columns if c.startswith("I")]
for c in numeric_cols:
    criteo = criteo.withColumn(c, F.col(c).cast("double"))

criteo_ml = criteo.select(["label"] + numeric_cols).na.fill(0).limit(2_000_000)
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
ctr_ready = assembler.transform(criteo_ml).select(F.col("label").cast("double"), "features")
train, test = ctr_ready.randomSplit([0.8, 0.2], seed=42)

with mlflow.start_run(run_name="ctr_prediction"):
    try:
        model = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10, maxDepth=4, seed=42).fit(train)
    except Exception:
        model = LogisticRegression(labelCol="label", featuresCol="features", maxIter=20).fit(train)
    pred = model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(pred)
    metrics["ctr_auc_roc"] = float(auc)
    mlflow.log_metric("ctr_auc_roc", float(auc))
    print(f"CTR AUC-ROC: {auc:.4f}")

print("=== 2. ALS Recommender (REES46) ===")
rees46 = spark.read.parquet(f"{BUCKET}/bronze/rees46/")
ratings = (
    rees46.select("user_id", "product_id", "event_type")
    .dropna(subset=["user_id", "product_id"])
    .withColumn("rating", F.when(F.col("event_type") == "purchase", 5.0).when(F.col("event_type") == "cart", 3.0).otherwise(1.0))
    .withColumn("user_idx", F.pmod(F.col("user_id").cast("long"), F.lit(1_000_000)).cast("int"))
    .withColumn("product_idx", F.pmod(F.col("product_id").cast("long"), F.lit(1_000_000)).cast("int"))
    .groupBy("user_idx", "product_idx")
    .agg(F.sum("rating").alias("rating"))
    .limit(3_000_000)
)

with mlflow.start_run(run_name="als_recommender"):
    als = ALS(userCol="user_idx", itemCol="product_idx", ratingCol="rating", implicitPrefs=True, coldStartStrategy="drop", rank=10, maxIter=5, regParam=0.1)
    als_model = als.fit(ratings)
    recs = als_model.recommendForAllUsers(5).limit(10)
    print("Sample ALS recommendations:")
    recs.show(truncate=False)
    metrics["als_sample_recommendation_users"] = 10
    mlflow.log_metric("als_sample_recommendation_users", 10)

print("=== 3. Product2Vec Similar Products ===")
sessions = (
    rees46.dropna(subset=["user_session", "product_id"])
    .groupBy("user_session")
    .agg(F.collect_list(F.col("product_id").cast("string")).alias("items"))
    .filter(F.size("items") >= 2)
    .limit(2_000_000)
)

with mlflow.start_run(run_name="product2vec"):
    w2v = Word2Vec(vectorSize=32, minCount=5, inputCol="items", outputCol="embedding", maxIter=5, seed=42)
    p2v_model = w2v.fit(sessions)
    vectors = p2v_model.getVectors()
    embedding_count = vectors.count()
    print("Sample Product2Vec embeddings:")
    vectors.show(10, truncate=False)
    vectors.write.mode("overwrite").parquet(f"{BUCKET}/gold/ml-artifacts/product2vec/vectors/")
    metrics["product_embedding_count"] = int(embedding_count)
    mlflow.log_metric("product_embedding_count", int(embedding_count))

print("=== 4. K-Means Customer Segmentation ===")
profiles = spark.read.format("hudi").load(f"{BUCKET}/serving/hudi/user_profiles/").limit(2_000_000)
feature_cols = ["total_events", "total_spend", "purchases", "cart_adds", "views", "unique_products", "category_diversity", "avg_price_viewed", "cart_abandonment_rate"]
for c in feature_cols:
    profiles = profiles.withColumn(c, F.col(c).cast("double"))
profiles = profiles.na.fill(0, subset=feature_cols)
features = VectorAssembler(inputCols=feature_cols, outputCol="raw_features").transform(profiles)
scaled = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True).fit(features).transform(features)

with mlflow.start_run(run_name="kmeans_segments"):
    kmeans = KMeans(k=5, seed=42, featuresCol="features", predictionCol="cluster", maxIter=10)
    km_model = kmeans.fit(scaled)
    clusters = km_model.transform(scaled)
    cluster_counts = {str(r["cluster"]): int(r["count"]) for r in clusters.groupBy("cluster").count().collect()}
    print("K-Means segment counts:")
    clusters.groupBy("cluster").count().orderBy("cluster").show()
    metrics["kmeans_segment_counts"] = cluster_counts
    for k, v in cluster_counts.items():
        mlflow.log_metric(f"cluster_{k}_count", v)

summary = {
    "project": "E-Commerce Clickstream Analytics & Recommendation Engine",
    "models": {
        "ctr_model": "GBT/Logistic Regression on Criteo-schema numeric features",
        "als_model": "ALS recommender on REES46 user-product interactions using hashed IDs",
        "product2vec": "Word2Vec embeddings over REES46 product sequences",
        "kmeans": "K-Means customer segmentation on Hudi user profiles"
    },
    "metrics": metrics
}

print("=== Saving ML summary to S3 ===")
print(json.dumps(summary, indent=2))
boto3.client("s3", region_name="us-east-2").put_object(
    Bucket=S3_BUCKET,
    Key="ml-artifacts/ml_summary.json",
    Body=json.dumps(summary, indent=2).encode("utf-8"),
    ContentType="application/json"
)

print("ML pipeline complete")
spark.stop()
