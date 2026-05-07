"""
Batch Layer — Apache Iceberg
=============================
Full historical analytics with time-travel and schema evolution.
Processes Bronze Parquet data into Silver/Gold Iceberg tables for
analytical queries: daily metrics, category performance, funnel analysis.

Usage:
    spark-submit --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0 \
        batch_layer/iceberg_batch.py
"""

import sys
import os
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg, min as spark_min,
    max as spark_max, datediff, lag, lead, when, lit, round as spark_round,
    window, hour, dayofweek, expr, row_number, dense_rank, percent_rank,
    collect_list, size, array_distinct, first, last, struct
)
from pyspark.sql.window import Window

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    spark_config, REES46_BRONZE_DIR, CRITEO_BRONZE_DIR, ICEBERG_WAREHOUSE
)


def create_iceberg_session() -> SparkSession:
    """Create Spark session with Iceberg catalog configured."""
    builder = SparkSession.builder \
        .appName("IcebergBatchLayer") \
        .master(spark_config.master)

    for key, value in spark_config.iceberg_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def create_iceberg_tables(spark: SparkSession):
    """Create Iceberg tables with proper schema and partitioning."""

    # Create namespace
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.ecommerce")

    # ─── REES46 Events Table ───
    spark.sql("""
        CREATE TABLE IF NOT EXISTS local.ecommerce.events (
            event_time      TIMESTAMP,
            event_date      DATE,
            event_year      INT,
            event_month     INT,
            event_day       INT,
            event_type      STRING,
            product_id      BIGINT,
            category_id     BIGINT,
            category_code   STRING,
            main_category   STRING,
            sub_category    STRING,
            brand           STRING,
            price           DOUBLE,
            user_id         BIGINT
        )
        USING iceberg
        PARTITIONED BY (event_month, event_type)
    """)

    # ─── Daily Metrics (Gold) ───
    spark.sql("""
        CREATE TABLE IF NOT EXISTS local.ecommerce.daily_metrics (
            event_date          DATE,
            total_events        BIGINT,
            unique_users        BIGINT,
            unique_products     BIGINT,
            total_views         BIGINT,
            total_carts         BIGINT,
            total_purchases     BIGINT,
            total_revenue       DOUBLE,
            avg_order_value     DOUBLE,
            view_to_cart_rate   DOUBLE,
            cart_to_purchase    DOUBLE,
            overall_conversion  DOUBLE
        )
        USING iceberg
        PARTITIONED BY (months(event_date))
    """)

    # ─── Product Performance (Gold) ───
    spark.sql("""
        CREATE TABLE IF NOT EXISTS local.ecommerce.product_performance (
            product_id          BIGINT,
            brand               STRING,
            main_category       STRING,
            total_views         BIGINT,
            total_carts         BIGINT,
            total_purchases     BIGINT,
            total_revenue       DOUBLE,
            unique_viewers      BIGINT,
            unique_buyers       BIGINT,
            conversion_rate     DOUBLE,
            avg_price           DOUBLE
        )
        USING iceberg
    """)

    # ─── Criteo CTR Features (Silver) ───
    spark.sql("""
        CREATE TABLE IF NOT EXISTS local.ecommerce.criteo_features (
            row_id      BIGINT,
            day         INT,
            label       INT,
            int_feature_1  FLOAT, int_feature_2  FLOAT, int_feature_3  FLOAT,
            int_feature_4  FLOAT, int_feature_5  FLOAT, int_feature_6  FLOAT,
            int_feature_7  FLOAT, int_feature_8  FLOAT, int_feature_9  FLOAT,
            int_feature_10 FLOAT, int_feature_11 FLOAT, int_feature_12 FLOAT,
            int_feature_13 FLOAT,
            cat_feature_1  STRING, cat_feature_2  STRING, cat_feature_3  STRING,
            cat_feature_4  STRING, cat_feature_5  STRING, cat_feature_6  STRING,
            cat_feature_7  STRING, cat_feature_8  STRING, cat_feature_9  STRING,
            cat_feature_10 STRING, cat_feature_11 STRING, cat_feature_12 STRING,
            cat_feature_13 STRING, cat_feature_14 STRING, cat_feature_15 STRING,
            cat_feature_16 STRING, cat_feature_17 STRING, cat_feature_18 STRING,
            cat_feature_19 STRING, cat_feature_20 STRING, cat_feature_21 STRING,
            cat_feature_22 STRING, cat_feature_23 STRING, cat_feature_24 STRING,
            cat_feature_25 STRING, cat_feature_26 STRING
        )
        USING iceberg
        PARTITIONED BY (day)
    """)

    print("[INFO] Iceberg tables created successfully")


def load_bronze_to_iceberg(spark: SparkSession):
    """Load Bronze Parquet into Iceberg events table."""
    print("[INFO] Loading REES46 Bronze data into Iceberg...")

    events_df = spark.read.parquet(REES46_BRONZE_DIR)
    events_df.writeTo("local.ecommerce.events").overwritePartitions()

    count = spark.table("local.ecommerce.events").count()
    print(f"[INFO] Loaded {count:,} events into Iceberg")

    # Load Criteo data
    print("[INFO] Loading Criteo Bronze data into Iceberg...")
    try:
        criteo_df = spark.read.parquet(CRITEO_BRONZE_DIR)
        criteo_df.writeTo("local.ecommerce.criteo_features").overwritePartitions()
        criteo_count = spark.table("local.ecommerce.criteo_features").count()
        print(f"[INFO] Loaded {criteo_count:,} Criteo records into Iceberg")
    except Exception as e:
        print(f"[WARN] Criteo data not loaded: {e}")


def compute_daily_metrics(spark: SparkSession):
    """Compute daily aggregate metrics and write to Gold Iceberg table."""
    print("[INFO] Computing daily metrics...")

    events = spark.table("local.ecommerce.events")

    daily = events.groupBy("event_date").agg(
        count("*").alias("total_events"),
        countDistinct("user_id").alias("unique_users"),
        countDistinct("product_id").alias("unique_products"),
        spark_sum(when(col("event_type") == "view", 1).otherwise(0)).alias("total_views"),
        spark_sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("total_carts"),
        spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),
        spark_sum(
            when(col("event_type") == "purchase", col("price")).otherwise(0)
        ).alias("total_revenue"),
    )

    # Compute conversion rates
    daily = daily.withColumn(
        "avg_order_value",
        when(col("total_purchases") > 0,
             spark_round(col("total_revenue") / col("total_purchases"), 2))
        .otherwise(0.0)
    ).withColumn(
        "view_to_cart_rate",
        when(col("total_views") > 0,
             spark_round(col("total_carts") / col("total_views") * 100, 2))
        .otherwise(0.0)
    ).withColumn(
        "cart_to_purchase",
        when(col("total_carts") > 0,
             spark_round(col("total_purchases") / col("total_carts") * 100, 2))
        .otherwise(0.0)
    ).withColumn(
        "overall_conversion",
        when(col("total_views") > 0,
             spark_round(col("total_purchases") / col("total_views") * 100, 4))
        .otherwise(0.0)
    )

    daily.writeTo("local.ecommerce.daily_metrics").overwritePartitions()
    print(f"[INFO] Daily metrics written: {daily.count()} days")

    # Show sample
    daily.orderBy("event_date").show(10, truncate=False)


def compute_product_performance(spark: SparkSession):
    """Compute product-level performance metrics."""
    print("[INFO] Computing product performance...")

    events = spark.table("local.ecommerce.events")

    products = events.groupBy("product_id").agg(
        first("brand").alias("brand"),
        first("main_category").alias("main_category"),
        spark_sum(when(col("event_type") == "view", 1).otherwise(0)).alias("total_views"),
        spark_sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("total_carts"),
        spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),
        spark_sum(
            when(col("event_type") == "purchase", col("price")).otherwise(0)
        ).alias("total_revenue"),
        countDistinct(
            when(col("event_type") == "view", col("user_id"))
        ).alias("unique_viewers"),
        countDistinct(
            when(col("event_type") == "purchase", col("user_id"))
        ).alias("unique_buyers"),
        avg("price").alias("avg_price"),
    )

    products = products.withColumn(
        "conversion_rate",
        when(col("total_views") > 0,
             spark_round(col("total_purchases") / col("total_views") * 100, 4))
        .otherwise(0.0)
    )

    products.writeTo("local.ecommerce.product_performance").overwritePartitions()
    print(f"[INFO] Product performance written: {products.count()} products")


def demonstrate_time_travel(spark: SparkSession):
    """Demonstrate Iceberg time-travel capabilities."""
    print("\n" + "=" * 50)
    print("  ICEBERG TIME-TRAVEL DEMONSTRATION")
    print("=" * 50)

    # Show table history
    spark.sql("SELECT * FROM local.ecommerce.events.history").show(truncate=False)

    # Show snapshots
    spark.sql("SELECT * FROM local.ecommerce.events.snapshots").show(truncate=False)

    # Show manifests
    print("\n[INFO] Table metadata files:")
    spark.sql("SELECT * FROM local.ecommerce.events.manifests").show(truncate=False)


def run_analytical_queries(spark: SparkSession):
    """Run sample analytical queries on Iceberg tables."""
    print("\n" + "=" * 50)
    print("  SAMPLE ANALYTICAL QUERIES")
    print("=" * 50)

    # Top categories by revenue
    print("\n[QUERY] Top 10 Categories by Revenue:")
    spark.sql("""
        SELECT main_category,
               COUNT(*) as total_events,
               SUM(CASE WHEN event_type='purchase' THEN price ELSE 0 END) as revenue,
               COUNT(DISTINCT user_id) as unique_users
        FROM local.ecommerce.events
        GROUP BY main_category
        ORDER BY revenue DESC
        LIMIT 10
    """).show(truncate=False)

    # Hourly traffic pattern
    print("\n[QUERY] Hourly Traffic Pattern:")
    spark.sql("""
        SELECT HOUR(event_time) as hour_of_day,
               COUNT(*) as events,
               COUNT(DISTINCT user_id) as users
        FROM local.ecommerce.events
        GROUP BY HOUR(event_time)
        ORDER BY hour_of_day
    """).show(24, truncate=False)

    # Funnel analysis
    print("\n[QUERY] Overall Funnel:")
    spark.sql("""
        SELECT event_type,
               COUNT(*) as total,
               COUNT(DISTINCT user_id) as unique_users
        FROM local.ecommerce.events
        GROUP BY event_type
        ORDER BY total DESC
    """).show(truncate=False)


def main():
    spark = create_iceberg_session()
    print("=" * 70)
    print("  BATCH LAYER — APACHE ICEBERG")
    print("  Historical Analytics with Time-Travel & Schema Evolution")
    print("=" * 70)

    start_time = time.time()

    # 1. Create Iceberg tables
    create_iceberg_tables(spark)

    # 2. Load Bronze data
    load_bronze_to_iceberg(spark)

    # 3. Compute Gold aggregations
    compute_daily_metrics(spark)
    compute_product_performance(spark)

    # 4. Demonstrate time-travel
    demonstrate_time_travel(spark)

    # 5. Run analytical queries
    run_analytical_queries(spark)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Batch layer completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
