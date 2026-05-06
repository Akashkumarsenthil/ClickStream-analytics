from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BUCKET = "s3a://clickstream-analytics-akash"

spark = SparkSession.builder.appName("03-Delta-Speed").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("=== Writing REES46 to Delta (partitioned by event_type) ===")
rees46 = spark.read.parquet(f"{BUCKET}/bronze/rees46/")

(
    rees46.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("event_type")
    .save(f"{BUCKET}/speed/delta/events/")
)

print("=== Trending products (simulated real-time) ===")
# Use view events as a speed-layer proxy for live product demand.
trending = (
    rees46.filter(F.col("event_type") == "view")
    .groupBy("product_id", "category_code", "brand")
    .agg(
        F.count("*").alias("view_count"),
        F.avg("price").alias("avg_price"),
        F.countDistinct("user_id").alias("unique_viewers"),
    )
    .orderBy(F.desc("view_count"))
)

trending.show(10, truncate=False)

(
    trending.write
    .format("delta")
    .mode("overwrite")
    .save(f"{BUCKET}/speed/delta/trending/")
)

print("=== Hourly event volume ===")
hourly = (
    rees46.withColumn("hour", F.hour("event_time"))
    .groupBy("event_type", "hour")
    .count()
    .orderBy("event_type", "hour")
)
hourly.show(20)

(
    hourly.write
    .format("delta")
    .mode("overwrite")
    .save(f"{BUCKET}/speed/delta/hourly_volume/")
)

print("Delta speed layer complete")
spark.stop()
