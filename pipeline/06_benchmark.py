from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time
import json
import boto3

BUCKET = "s3a://clickstream-analytics-akash"

spark = (
    SparkSession.builder
    .appName("06-Benchmark")
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.local.type", "hadoop")
    .config("spark.sql.catalog.local.warehouse", f"{BUCKET}/batch/iceberg/")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

results = []

def benchmark(name, read_fn, query_fn, runs=3):
    times = []
    for _ in range(runs):
        df = read_fn()
        t0 = time.time()
        query_fn(df).collect()
        times.append(round(time.time() - t0, 2))
    avg = round(sum(times) / len(times), 2)
    print(f"{name}: avg={avg}s runs={times}")
    return avg, times

def top_products(df):
    return (
        df.filter(F.col("event_type") == "view")
        .groupBy("product_id")
        .count()
        .orderBy(F.desc("count"))
        .limit(20)
    )

print("=== Iceberg ===")
t1, r1 = benchmark(
    "Iceberg",
    lambda: spark.sql("SELECT * FROM local.clickstream.rees46_events"),
    top_products,
)
results.append({"format": "Apache Iceberg", "avg_query_sec": t1, "runs": r1, "feature": "Time travel, schema evolution"})

print("=== Delta Lake ===")
t2, r2 = benchmark(
    "Delta Lake",
    lambda: spark.read.format("delta").load(f"{BUCKET}/speed/delta/events/"),
    top_products,
)
results.append({"format": "Delta Lake", "avg_query_sec": t2, "runs": r2, "feature": "ACID transactions, streaming"})

print("=== Apache Hudi ===")
def hudi_query(df):
    return (
        df.groupBy("user_id")
        .agg(F.sum("total_spend").alias("spend"))
        .orderBy(F.desc("spend"))
        .limit(20)
    )

t3, r3 = benchmark(
    "Apache Hudi",
    lambda: spark.read.format("hudi").load(f"{BUCKET}/serving/hudi/user_profiles/"),
    hudi_query,
)
results.append({"format": "Apache Hudi", "avg_query_sec": t3, "runs": r3, "feature": "Upserts, MOR tables"})

print("\n=== BENCHMARK RESULTS ===")
for row in results:
    print(row)

s3 = boto3.client("s3", region_name="us-east-2")
s3.put_object(
    Bucket="clickstream-analytics-akash",
    Key="benchmark/results.json",
    Body=json.dumps(results, indent=2).encode("utf-8"),
)
print("Results saved to S3")
spark.stop()
