"""
Customer Segmentation — K-Means Clustering on REES46
=====================================================
Clusters users based on behavioral features:
  - Session frequency, average order value
  - Cart abandonment rate, category diversity
Identifies actionable segments: "Loyal Whales", "Cart Abandoners", etc.

Usage:
    spark-submit ml/segmentation/customer_segments.py
"""

import sys
import os
import time
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg,
    when, lit, round as spark_round, min as spark_min,
    max as spark_max, datediff, size, collect_set,
    unix_timestamp, stddev, percentile_approx
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    spark_config, segmentation_config,
    REES46_BRONZE_DIR, ML_ARTIFACTS_DIR
)


def create_spark_session() -> SparkSession:
    builder = SparkSession.builder \
        .appName("CustomerSegmentation") \
        .master(spark_config.master)

    for key, value in spark_config.base_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def build_user_features(spark: SparkSession) -> DataFrame:
    """Build comprehensive user-level features for clustering."""
    print("[INFO] Building user features for segmentation...")

    events = spark.read.parquet(REES46_BRONZE_DIR)

    # ─── Base aggregations ───
    user_features = events.groupBy("user_id").agg(
        # Session proxy: count distinct active days
        countDistinct("event_date").alias("total_sessions"),

        # Event counts by type
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
        countDistinct("main_category").alias("category_diversity"),
        countDistinct("brand").alias("brand_diversity"),

        # Temporal
        spark_min("event_time").alias("first_seen"),
        spark_max("event_time").alias("last_seen"),

        # Price sensitivity
        avg("price").alias("avg_viewed_price"),
        stddev("price").alias("price_std"),
    )

    # ─── Derived features ───
    user_features = user_features.withColumn(
        "cart_abandonment_rate",
        when(col("total_carts") > 0,
             spark_round(
                 (col("total_carts") - col("total_purchases")) / col("total_carts"),
                 4
             ))
        .otherwise(lit(0.0))
    ).withColumn(
        "purchase_frequency",
        when(col("total_sessions") > 0,
             spark_round(col("total_purchases") / col("total_sessions"), 4))
        .otherwise(lit(0.0))
    ).withColumn(
        "avg_session_duration_min",
        when(col("total_sessions") > 0,
             spark_round(
                 (unix_timestamp("last_seen") - unix_timestamp("first_seen")) /
                 60.0 / col("total_sessions"),
                 2
             ))
        .otherwise(lit(0.0))
    ).withColumn(
        "avg_order_value",
        when(col("avg_order_value").isNull(), lit(0.0))
        .otherwise(col("avg_order_value"))
    ).withColumn(
        "price_std",
        when(col("price_std").isNull(), lit(0.0))
        .otherwise(col("price_std"))
    )

    # Select features for clustering
    feature_cols = segmentation_config.features
    user_features = user_features.select(
        "user_id", *feature_cols
    ).na.fill(0.0)

    total_users = user_features.count()
    print(f"[INFO] User features built for {total_users:,} users")
    print(f"[INFO] Feature columns: {feature_cols}")

    return user_features


def find_optimal_k(features_df: DataFrame) -> dict:
    """Run K-Means for multiple K values and find optimal K via silhouette."""
    print("\n" + "=" * 50)
    print("  FINDING OPTIMAL K (Elbow Method + Silhouette)")
    print("=" * 50)

    feature_cols = segmentation_config.features

    # Assemble and scale
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features",
        withStd=True, withMean=True
    )

    pipeline = Pipeline(stages=[assembler, scaler])
    prep_model = pipeline.fit(features_df)
    scaled_df = prep_model.transform(features_df)
    scaled_df.cache()

    evaluator = ClusteringEvaluator(
        predictionCol="prediction",
        featuresCol="features",
        metricName="silhouette"
    )

    results = {}
    best_k = 0
    best_silhouette = -1

    for k in segmentation_config.k_range:
        start = time.time()
        kmeans = KMeans(
            k=k,
            featuresCol="features",
            predictionCol="prediction",
            maxIter=segmentation_config.max_iter,
            seed=segmentation_config.seed
        )

        model = kmeans.fit(scaled_df)
        predictions = model.transform(scaled_df)

        silhouette = evaluator.evaluate(predictions)
        inertia = model.summary.trainingCost
        elapsed = time.time() - start

        results[k] = {
            "silhouette": round(silhouette, 4),
            "inertia": round(inertia, 2),
            "time_sec": round(elapsed, 1),
        }

        print(f"  K={k}: Silhouette={silhouette:.4f}, Inertia={inertia:.0f}, Time={elapsed:.1f}s")

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k

    print(f"\n[INFO] Optimal K: {best_k} (Silhouette: {best_silhouette:.4f})")
    return results, scaled_df, best_k


def train_final_model(scaled_df: DataFrame, k: int) -> tuple:
    """Train final K-Means model with optimal K."""
    print(f"\n[INFO] Training final K-Means model with K={k}...")

    start = time.time()

    kmeans = KMeans(
        k=k,
        featuresCol="features",
        predictionCol="cluster",
        maxIter=segmentation_config.max_iter,
        seed=segmentation_config.seed
    )

    model = kmeans.fit(scaled_df)
    predictions = model.transform(scaled_df)

    elapsed = time.time() - start
    print(f"[INFO] Training completed in {elapsed:.1f}s")

    return model, predictions


def analyze_segments(predictions: DataFrame, k: int):
    """Analyze and label customer segments based on cluster characteristics."""
    print("\n" + "=" * 60)
    print("  CUSTOMER SEGMENT ANALYSIS")
    print("=" * 60)

    feature_cols = segmentation_config.features

    # Compute cluster statistics
    for cluster_id in range(k):
        cluster_df = predictions.filter(col("cluster") == cluster_id)
        cluster_size = cluster_df.count()
        total = predictions.count()
        pct = cluster_size / total * 100

        # Get segment label
        label = segmentation_config.segment_labels.get(cluster_id, f"Segment {cluster_id}")

        print(f"\n  ─── Cluster {cluster_id}: {label} ({cluster_size:,} users, {pct:.1f}%) ───")

        # Feature averages
        stats = cluster_df.agg(
            *[avg(c).alias(c) for c in feature_cols]
        ).first()

        for feat in feature_cols:
            val = stats[feat] if stats[feat] is not None else 0
            print(f"    {feat:<30}: {val:>10.2f}")

    # ─── Segment Distribution ───
    print(f"\n{'=' * 60}")
    print("  SEGMENT DISTRIBUTION SUMMARY")
    print(f"{'=' * 60}")

    dist = predictions.groupBy("cluster").agg(
        count("*").alias("users"),
        spark_round(avg("total_spend"), 2).alias("avg_spend"),
        spark_round(avg("total_purchases"), 2).alias("avg_purchases"),
        spark_round(avg("cart_abandonment_rate"), 4).alias("avg_cart_abandon"),
        spark_round(avg("category_diversity"), 1).alias("avg_cat_diversity"),
    ).orderBy("cluster")

    dist.show(truncate=False)

    return dist


def main():
    spark = create_spark_session()
    print("=" * 70)
    print("  CUSTOMER SEGMENTATION — K-MEANS CLUSTERING")
    print("  Identifying: Loyal Whales, Cart Abandoners, Window Shoppers...")
    print("=" * 70)

    os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)
    start_time = time.time()

    # 1. Build user features
    user_features = build_user_features(spark)

    # 2. Find optimal K
    k_results, scaled_df, best_k = find_optimal_k(user_features)

    # 3. Train final model
    model, predictions = train_final_model(scaled_df, best_k)

    # 4. Analyze segments
    segment_dist = analyze_segments(predictions, best_k)

    # 5. Save model and results
    model_path = os.path.join(ML_ARTIFACTS_DIR, "kmeans_model")
    model.write().overwrite().save(model_path)
    print(f"\n[INFO] K-Means model saved to: {model_path}")

    # Save segment assignments
    segment_path = os.path.join(ML_ARTIFACTS_DIR, "user_segments")
    predictions.select("user_id", "cluster").write \
        .mode("overwrite").parquet(segment_path)

    results = {
        "optimal_k": best_k,
        "k_search_results": k_results,
        "segment_labels": segmentation_config.segment_labels,
    }
    with open(os.path.join(ML_ARTIFACTS_DIR, "segmentation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Customer segmentation completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
