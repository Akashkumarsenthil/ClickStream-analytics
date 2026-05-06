"""
Serving Layer — Apache Hudi
============================
Merge-on-Read tables for mutable user profiles and product stats.
Hudi enables efficient upserts for maintaining up-to-date user profiles
and product statistics that change with each new event.

Usage:
    spark-submit --packages org.apache.hudi:hudi-spark3.5-bundle_2.12:0.14.1 \
        serving_layer/hudi_serving.py
"""

import sys
import os
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg,
    when, lit, round as spark_round, max as spark_max,
    min as spark_min, first, last, collect_set, size,
    datediff, current_date, concat_ws, array_distinct,
    struct, to_json, from_json, expr
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    spark_config, rees46_config,
    REES46_BRONZE_DIR, HUDI_PATH
)

# Hudi table paths
HUDI_USER_PROFILES_PATH = os.path.join(HUDI_PATH, "user_profiles")
HUDI_PRODUCT_STATS_PATH = os.path.join(HUDI_PATH, "product_stats")
HUDI_BRAND_METRICS_PATH = os.path.join(HUDI_PATH, "brand_metrics")


def create_hudi_session() -> SparkSession:
    """Create Spark session with Hudi configuration."""
    builder = SparkSession.builder \
        .appName("HudiServingLayer") \
        .master(spark_config.master)

    for key, value in spark_config.hudi_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def build_user_profiles(spark: SparkSession) -> DataFrame:
    """Build comprehensive user profiles from REES46 event data.
    
    Aggregates user behavior into a single profile record per user.
    """
    print("[INFO] Building user profiles...")

    events = spark.read.parquet(REES46_BRONZE_DIR)

    profiles = events.groupBy("user_id").agg(
        # Activity metrics
        count("*").alias("total_events"),
        spark_sum(when(col("event_type") == "view", 1).otherwise(0)).alias("total_views"),
        spark_sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("total_carts"),
        spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),

        # Revenue metrics
        spark_sum(
            when(col("event_type") == "purchase", col("price")).otherwise(0)
        ).alias("total_spend"),
        avg(
            when(col("event_type") == "purchase", col("price"))
        ).alias("avg_order_value"),

        # Diversity metrics
        countDistinct("product_id").alias("unique_products_seen"),
        countDistinct("main_category").alias("category_diversity"),
        countDistinct("brand").alias("brand_diversity"),

        # Temporal metrics
        spark_min("event_time").alias("first_seen"),
        spark_max("event_time").alias("last_seen"),
        countDistinct("event_date").alias("active_days"),

        # Favorite category and brand
        first("main_category").alias("top_category"),
        first("brand").alias("top_brand"),
    )

    # Compute derived metrics
    profiles = profiles.withColumn(
        "cart_abandonment_rate",
        when(col("total_carts") > 0,
             spark_round((col("total_carts") - col("total_purchases")) / col("total_carts") * 100, 2))
        .otherwise(0.0)
    ).withColumn(
        "view_to_purchase_rate",
        when(col("total_views") > 0,
             spark_round(col("total_purchases") / col("total_views") * 100, 4))
        .otherwise(0.0)
    ).withColumn(
        "is_buyer",
        when(col("total_purchases") > 0, lit(True)).otherwise(lit(False))
    ).withColumn(
        "lifetime_days",
        datediff(col("last_seen"), col("first_seen"))
    ).withColumn(
        "events_per_day",
        when(col("active_days") > 0,
             spark_round(col("total_events") / col("active_days"), 2))
        .otherwise(0.0)
    )

    print(f"[INFO] Built profiles for {profiles.count():,} users")
    return profiles


def write_hudi_user_profiles(spark: SparkSession, profiles: DataFrame):
    """Write user profiles to Hudi Merge-on-Read table."""
    print(f"[INFO] Writing user profiles to Hudi MoR: {HUDI_USER_PROFILES_PATH}")

    hudi_options = {
        "hoodie.table.name": "user_profiles",
        "hoodie.datasource.write.recordkey.field": "user_id",
        "hoodie.datasource.write.precombine.field": "total_events",
        "hoodie.datasource.write.table.type": "MERGE_ON_READ",
        "hoodie.datasource.write.operation": "upsert",
        "hoodie.datasource.write.partitionpath.field": "",
        "hoodie.upsert.shuffle.parallelism": "200",
        "hoodie.insert.shuffle.parallelism": "200",
        "hoodie.datasource.write.hive_style_partitioning": "true",
        "hoodie.cleaner.policy": "KEEP_LATEST_COMMITS",
        "hoodie.cleaner.commits.retained": "5",
    }

    profiles.write \
        .format("hudi") \
        .options(**hudi_options) \
        .mode("overwrite") \
        .save(HUDI_USER_PROFILES_PATH)

    print("[INFO] User profiles written to Hudi successfully")


def build_product_stats(spark: SparkSession) -> DataFrame:
    """Build mutable product statistics from REES46 data."""
    print("[INFO] Building product statistics...")

    events = spark.read.parquet(REES46_BRONZE_DIR)

    product_stats = events.groupBy("product_id").agg(
        first("brand").alias("brand"),
        first("main_category").alias("main_category"),
        first("sub_category").alias("sub_category"),
        first("category_code").alias("category_code"),
        avg("price").alias("current_price"),

        # Interaction counts
        count("*").alias("total_interactions"),
        spark_sum(when(col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
        spark_sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
        spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),

        # User metrics
        countDistinct("user_id").alias("unique_users"),
        countDistinct(
            when(col("event_type") == "purchase", col("user_id"))
        ).alias("unique_buyers"),

        # Revenue
        spark_sum(
            when(col("event_type") == "purchase", col("price")).otherwise(0)
        ).alias("total_revenue"),

        # Temporal
        spark_min("event_time").alias("first_interaction"),
        spark_max("event_time").alias("last_interaction"),
    )

    # Derived metrics
    product_stats = product_stats.withColumn(
        "conversion_rate",
        when(col("view_count") > 0,
             spark_round(col("purchase_count") / col("view_count") * 100, 4))
        .otherwise(0.0)
    ).withColumn(
        "cart_to_purchase_rate",
        when(col("cart_count") > 0,
             spark_round(col("purchase_count") / col("cart_count") * 100, 2))
        .otherwise(0.0)
    ).withColumn(
        "popularity_score",
        spark_round(
            col("view_count") * 1.0 +
            col("cart_count") * 3.0 +
            col("purchase_count") * 5.0,
            2
        )
    )

    print(f"[INFO] Built stats for {product_stats.count():,} products")
    return product_stats


def write_hudi_product_stats(spark: SparkSession, product_stats: DataFrame):
    """Write product stats to Hudi table with upsert capability."""
    print(f"[INFO] Writing product stats to Hudi: {HUDI_PRODUCT_STATS_PATH}")

    hudi_options = {
        "hoodie.table.name": "product_stats",
        "hoodie.datasource.write.recordkey.field": "product_id",
        "hoodie.datasource.write.precombine.field": "total_interactions",
        "hoodie.datasource.write.table.type": "MERGE_ON_READ",
        "hoodie.datasource.write.operation": "upsert",
        "hoodie.datasource.write.partitionpath.field": "main_category",
        "hoodie.upsert.shuffle.parallelism": "200",
    }

    product_stats.write \
        .format("hudi") \
        .options(**hudi_options) \
        .mode("overwrite") \
        .save(HUDI_PRODUCT_STATS_PATH)

    print("[INFO] Product stats written to Hudi successfully")


def build_brand_metrics(spark: SparkSession) -> DataFrame:
    """Build brand-level aggregate metrics."""
    print("[INFO] Building brand metrics...")

    events = spark.read.parquet(REES46_BRONZE_DIR)

    brand_metrics = events.groupBy("brand").agg(
        countDistinct("product_id").alias("product_count"),
        count("*").alias("total_interactions"),
        countDistinct("user_id").alias("total_users"),
        spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),
        spark_sum(
            when(col("event_type") == "purchase", col("price")).otherwise(0)
        ).alias("total_revenue"),
        avg("price").alias("avg_price"),
        countDistinct("main_category").alias("categories_present"),
    )

    brand_metrics = brand_metrics.withColumn(
        "revenue_per_user",
        when(col("total_users") > 0,
             spark_round(col("total_revenue") / col("total_users"), 2))
        .otherwise(0.0)
    )

    return brand_metrics


def write_hudi_brand_metrics(spark: SparkSession, brand_metrics: DataFrame):
    """Write brand metrics to Hudi."""
    print(f"[INFO] Writing brand metrics to Hudi: {HUDI_BRAND_METRICS_PATH}")

    hudi_options = {
        "hoodie.table.name": "brand_metrics",
        "hoodie.datasource.write.recordkey.field": "brand",
        "hoodie.datasource.write.precombine.field": "total_interactions",
        "hoodie.datasource.write.table.type": "MERGE_ON_READ",
        "hoodie.datasource.write.operation": "upsert",
        "hoodie.datasource.write.partitionpath.field": "",
    }

    brand_metrics.write \
        .format("hudi") \
        .options(**hudi_options) \
        .mode("overwrite") \
        .save(HUDI_BRAND_METRICS_PATH)

    print("[INFO] Brand metrics written to Hudi successfully")


def simulate_incremental_upsert(spark: SparkSession):
    """Demonstrate Hudi's upsert capability with incremental updates.
    
    Simulates new events arriving and updating user profiles.
    """
    print("\n" + "=" * 50)
    print("  HUDI INCREMENTAL UPSERT DEMONSTRATION")
    print("=" * 50)

    # Read existing profiles
    existing = spark.read.format("hudi").load(HUDI_USER_PROFILES_PATH)
    print(f"[INFO] Existing profiles: {existing.count():,}")

    # Simulate updates: increment event counts for top users
    top_users = existing.orderBy(col("total_events").desc()).limit(100)

    updated = top_users.withColumn(
        "total_events", col("total_events") + lit(10)
    ).withColumn(
        "total_views", col("total_views") + lit(5)
    )

    # Upsert
    hudi_options = {
        "hoodie.table.name": "user_profiles",
        "hoodie.datasource.write.recordkey.field": "user_id",
        "hoodie.datasource.write.precombine.field": "total_events",
        "hoodie.datasource.write.table.type": "MERGE_ON_READ",
        "hoodie.datasource.write.operation": "upsert",
        "hoodie.datasource.write.partitionpath.field": "",
    }

    updated.write \
        .format("hudi") \
        .options(**hudi_options) \
        .mode("append") \
        .save(HUDI_USER_PROFILES_PATH)

    # Verify upsert
    after_update = spark.read.format("hudi").load(HUDI_USER_PROFILES_PATH)
    print(f"[INFO] Profiles after upsert: {after_update.count():,}")
    print("[INFO] Upsert demonstration complete — 100 profiles updated")


def read_hudi_tables(spark: SparkSession):
    """Read and display Hudi table contents."""
    print("\n" + "=" * 50)
    print("  HUDI TABLE CONTENTS")
    print("=" * 50)

    # User profiles sample
    print("\n[INFO] User Profiles (top 10 by spend):")
    try:
        profiles = spark.read.format("hudi").load(HUDI_USER_PROFILES_PATH)
        profiles.orderBy(col("total_spend").desc()).select(
            "user_id", "total_events", "total_purchases",
            "total_spend", "avg_order_value", "cart_abandonment_rate",
            "category_diversity", "lifetime_days"
        ).show(10, truncate=False)
    except Exception as e:
        print(f"[WARN] {e}")

    # Product stats sample
    print("\n[INFO] Product Stats (top 10 by popularity):")
    try:
        products = spark.read.format("hudi").load(HUDI_PRODUCT_STATS_PATH)
        products.orderBy(col("popularity_score").desc()).select(
            "product_id", "brand", "main_category",
            "view_count", "purchase_count", "total_revenue",
            "conversion_rate", "popularity_score"
        ).show(10, truncate=False)
    except Exception as e:
        print(f"[WARN] {e}")


def main():
    spark = create_hudi_session()
    print("=" * 70)
    print("  SERVING LAYER — APACHE HUDI")
    print("  Merge-on-Read Tables for Mutable User Profiles & Product Stats")
    print("=" * 70)

    start_time = time.time()

    # 1. Build and write user profiles
    profiles = build_user_profiles(spark)
    write_hudi_user_profiles(spark, profiles)

    # 2. Build and write product stats
    product_stats = build_product_stats(spark)
    write_hudi_product_stats(spark, product_stats)

    # 3. Build and write brand metrics
    brand_metrics = build_brand_metrics(spark)
    write_hudi_brand_metrics(spark, brand_metrics)

    # 4. Demonstrate incremental upsert
    simulate_incremental_upsert(spark)

    # 5. Read and display results
    read_hudi_tables(spark)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Serving layer completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
