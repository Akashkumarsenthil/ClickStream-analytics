from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BUCKET = "s3a://clickstream-analytics-akash"

spark = SparkSession.builder.appName("01-Bronze").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("=== REES46 Bronze (real events) ===")
rees46_raw = f"{BUCKET}/raw-csv/rees46/"
rees46_out = f"{BUCKET}/bronze/rees46/"

rees46 = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(rees46_raw)
)

# Expected fields: event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session
rees46 = (
    rees46
    .withColumn("event_time", F.to_timestamp("event_time"))
    .withColumn("event_date", F.to_date("event_time"))
    .withColumn("product_id", F.col("product_id").cast("long"))
    .withColumn("user_id", F.col("user_id").cast("long"))
    .withColumn("price", F.col("price").cast("double"))
    .filter(F.col("event_time").isNotNull())
)

rees46_count = rees46.count()
print(f"REES46 rows: {rees46_count:,}")

(
    rees46.write
    .mode("overwrite")
    .partitionBy("event_date")
    .option("compression", "snappy")
    .parquet(rees46_out)
)
print("REES46 Bronze written to S3")

print("=== Criteo Bronze (5M synthetic rows, Criteo schema) ===")
# This fallback keeps the pipeline demonstrable when the real day_0.gz is not yet downloaded.
# Replace this with a real Criteo reader when raw-csv/criteo/day_0.gz is available.
N = 5_000_000
criteo = spark.range(N).withColumnRenamed("id", "row_id")
criteo = criteo.withColumn("label", (F.rand(seed=42) > 0.5).cast("int"))
for i in range(1, 14):
    criteo = criteo.withColumn(f"I{i}", (F.rand(seed=100 + i) * 1000).cast("double"))
for i in range(1, 27):
    criteo = criteo.withColumn(f"C{i}", F.sha2(F.concat_ws("_", F.lit(i), F.col("row_id")), 256))

criteo_out = f"{BUCKET}/bronze/criteo/"
criteo_count = criteo.count()
print(f"Criteo rows: {criteo_count:,}")

(
    criteo.drop("row_id")
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(criteo_out)
)

print(f"TOTAL RECORDS: {rees46_count + criteo_count:,}")
spark.stop()
