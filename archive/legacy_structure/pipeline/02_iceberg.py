from pyspark.sql import SparkSession

BUCKET = "s3a://clickstream-analytics-akash"

spark = (
    SparkSession.builder
    .appName("02-Iceberg")
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.local.type", "hadoop")
    .config("spark.sql.catalog.local.warehouse", f"{BUCKET}/batch/iceberg/")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

spark.sql("CREATE NAMESPACE IF NOT EXISTS local.clickstream")

print("=== Writing REES46 to Iceberg ===")
rees46 = spark.read.parquet(f"{BUCKET}/bronze/rees46/")

rees46.writeTo("local.clickstream.rees46_events") \
    .tableProperty("write.format.default", "parquet") \
    .tableProperty("write.parquet.compression-codec", "snappy") \
    .createOrReplace()

print("=== Batch analytics via SQL ===")
spark.sql("""
    SELECT category_code, event_type, COUNT(*) AS events, ROUND(AVG(price), 2) AS avg_price
    FROM local.clickstream.rees46_events
    WHERE category_code IS NOT NULL
    GROUP BY category_code, event_type
    ORDER BY events DESC
    LIMIT 20
""").show(truncate=False)

spark.sql("SELECT COUNT(*) AS total FROM local.clickstream.rees46_events").show()

print("=== Criteo to Iceberg ===")
criteo = spark.read.parquet(f"{BUCKET}/bronze/criteo/")

criteo.writeTo("local.clickstream.criteo_clicks") \
    .tableProperty("write.format.default", "parquet") \
    .createOrReplace()

spark.sql("""
    SELECT label, COUNT(*) AS cnt
    FROM local.clickstream.criteo_clicks
    GROUP BY label
""").show()

print("Iceberg batch layer complete")
spark.stop()
