from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BUCKET = "s3a://clickstream-analytics-akash"

spark = SparkSession.builder.appName("04-Hudi-Serving").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("=== Building user profiles for Hudi serving layer ===")
rees46 = spark.read.parquet(f"{BUCKET}/bronze/rees46/")

profiles = (
    rees46.groupBy("user_id")
    .agg(
        F.count("*").alias("total_events"),
        F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("total_views"),
        F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("total_carts"),
        F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),
        F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0.0)).alias("total_spend"),
        F.max("event_time").alias("last_event_time"),
    )
    .withColumn("cart_rate", F.col("total_carts") / F.col("total_events"))
    .withColumn("purchase_rate", F.col("total_purchases") / F.col("total_events"))
    .withColumn(
        "segment_label",
        F.when((F.col("total_purchases") >= 3) | (F.col("total_spend") >= 500), "Loyal Whales")
        .when((F.col("total_carts") >= 2) & (F.col("total_purchases") == 0), "Cart Abandoners")
        .when(F.col("total_events") >= 20, "Power Browsers")
        .otherwise("Casual Browsers"),
    )
    .withColumn("profile_updated_at", F.current_timestamp())
)

profiles.show(20, truncate=False)

hudi_options = {
    "hoodie.table.name": "user_profiles",
    "hoodie.datasource.write.table.type": "MERGE_ON_READ",
    "hoodie.datasource.write.recordkey.field": "user_id",
    "hoodie.datasource.write.precombine.field": "last_event_time",
    "hoodie.datasource.write.operation": "upsert",
    "hoodie.datasource.write.hive_style_partitioning": "false",
}

(
    profiles.write
    .format("hudi")
    .options(**hudi_options)
    .mode("overwrite")
    .save(f"{BUCKET}/serving/hudi/user_profiles/")
)

print("Hudi serving layer complete")
spark.stop()
