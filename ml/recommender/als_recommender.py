"""
Collaborative Filtering — ALS on REES46 Implicit Feedback
==========================================================
Spark MLlib ALS model for product recommendations.
Implicit feedback weighting: view=1, cart=3, purchase=5.
Evaluated with Precision@K, Recall@K, NDCG@K against popularity baseline.

Usage:
    spark-submit ml/recommender/als_recommender.py
"""

import sys
import os
import time
import json
import math

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg,
    when, lit, round as spark_round, collect_list,
    explode, row_number, dense_rank, size, array,
    struct, desc, asc, monotonically_increasing_id
)
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    spark_config, rees46_config, als_config,
    REES46_BRONZE_DIR, ML_ARTIFACTS_DIR
)


def create_spark_session() -> SparkSession:
    builder = SparkSession.builder \
        .appName("ALSRecommender") \
        .master(spark_config.master)

    for key, value in spark_config.base_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def prepare_interaction_matrix(spark: SparkSession) -> DataFrame:
    """Build user-item interaction matrix with implicit feedback weights.
    
    Weights: view=1, cart=3, purchase=5
    Aggregates by (user_id, product_id) to sum weights.
    """
    print("[INFO] Building interaction matrix...")

    events = spark.read.parquet(REES46_BRONZE_DIR)

    # Assign weights
    weights = rees46_config.implicit_weights
    interactions = events.withColumn(
        "weight",
        when(col("event_type") == "view", lit(weights["view"]))
        .when(col("event_type") == "cart", lit(weights["cart"]))
        .when(col("event_type") == "purchase", lit(weights["purchase"]))
        .otherwise(lit(0))
    )

    # Aggregate by user-product pair
    user_item = interactions.groupBy("user_id", "product_id").agg(
        spark_sum("weight").alias("rating"),
        count("*").alias("interaction_count"),
    )

    # Convert IDs to integer indices for ALS
    # ALS requires integer user/item IDs
    user_item = user_item.withColumn(
        "user_idx", (col("user_id") % 2147483647).cast("int")
    ).withColumn(
        "item_idx", (col("product_id") % 2147483647).cast("int")
    )

    total_interactions = user_item.count()
    unique_users = user_item.select(countDistinct("user_id")).first()[0]
    unique_items = user_item.select(countDistinct("product_id")).first()[0]

    print(f"[INFO] Interaction matrix:")
    print(f"  Total interactions: {total_interactions:,}")
    print(f"  Unique users: {unique_users:,}")
    print(f"  Unique items: {unique_items:,}")
    print(f"  Sparsity: {1 - total_interactions / (unique_users * unique_items):.6f}")

    return user_item


def build_popularity_baseline(spark: SparkSession, user_item: DataFrame) -> DataFrame:
    """Build popularity-based recommendation baseline.
    
    Simply recommends the most popular items to all users.
    """
    print("[INFO] Building popularity baseline...")

    popular_items = user_item.groupBy("product_id", "item_idx").agg(
        spark_sum("rating").alias("total_rating"),
        countDistinct("user_id").alias("user_count"),
    ).orderBy(col("total_rating").desc())

    return popular_items


def train_als(train_df: DataFrame, test_df: DataFrame) -> tuple:
    """Train ALS model on implicit feedback data."""
    print("\n" + "=" * 50)
    print("  TRAINING ALS MODEL")
    print("=" * 50)

    start = time.time()

    als = ALS(
        rank=als_config.rank,
        maxIter=als_config.max_iter,
        regParam=als_config.reg_param,
        alpha=als_config.alpha,
        implicitPrefs=als_config.implicit_prefs,
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        coldStartStrategy=als_config.cold_start_strategy,
        nonnegative=True,
        seed=42
    )

    print(f"[INFO] ALS params: rank={als_config.rank}, maxIter={als_config.max_iter}, "
          f"regParam={als_config.reg_param}, alpha={als_config.alpha}")

    model = als.fit(train_df)

    elapsed = time.time() - start
    print(f"[INFO] ALS training completed in {elapsed:.1f}s")

    return model, elapsed


def evaluate_recommendations(
    model: ALSModel,
    test_df: DataFrame,
    popular_items: DataFrame,
    k: int = 10
) -> dict:
    """Evaluate ALS model with Precision@K, Recall@K, NDCG@K.
    
    Compares against popularity baseline.
    """
    print(f"\n[INFO] Evaluating recommendations (K={k})...")

    # Get ground truth: items each user actually interacted with in test
    ground_truth = test_df.groupBy("user_idx").agg(
        collect_list("item_idx").alias("actual_items"),
        count("*").alias("num_actual")
    )

    # ALS recommendations
    user_recs = model.recommendForAllUsers(k)

    # Flatten recommendations
    user_recs_flat = user_recs.select(
        col("user_idx"),
        col("recommendations.item_idx").alias("rec_items")
    )

    # Join with ground truth
    eval_df = user_recs_flat.join(ground_truth, "user_idx", "inner")

    # Compute metrics using Python UDFs
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType, ArrayType, IntegerType

    @udf(DoubleType())
    def precision_at_k(rec_items, actual_items):
        if not rec_items or not actual_items:
            return 0.0
        hits = len(set(rec_items) & set(actual_items))
        return hits / len(rec_items)

    @udf(DoubleType())
    def recall_at_k(rec_items, actual_items):
        if not rec_items or not actual_items:
            return 0.0
        hits = len(set(rec_items) & set(actual_items))
        return hits / len(actual_items)

    @udf(DoubleType())
    def ndcg_at_k(rec_items, actual_items):
        if not rec_items or not actual_items:
            return 0.0
        actual_set = set(actual_items)
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, item in enumerate(rec_items)
            if item in actual_set
        )
        ideal_hits = min(len(actual_set), len(rec_items))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    # Compute per-user metrics
    metrics_df = eval_df.withColumn(
        "precision", precision_at_k(col("rec_items"), col("actual_items"))
    ).withColumn(
        "recall", recall_at_k(col("rec_items"), col("actual_items"))
    ).withColumn(
        "ndcg", ndcg_at_k(col("rec_items"), col("actual_items"))
    )

    # Average metrics
    avg_metrics = metrics_df.agg(
        avg("precision").alias("avg_precision"),
        avg("recall").alias("avg_recall"),
        avg("ndcg").alias("avg_ndcg"),
        count("*").alias("num_users_evaluated"),
    ).first()

    als_metrics = {
        f"precision@{k}": round(avg_metrics["avg_precision"], 4),
        f"recall@{k}": round(avg_metrics["avg_recall"], 4),
        f"ndcg@{k}": round(avg_metrics["avg_ndcg"], 4),
        "users_evaluated": avg_metrics["num_users_evaluated"],
    }

    # ─── Popularity Baseline ───
    top_k_popular = [row["item_idx"] for row in popular_items.limit(k).collect()]

    @udf(DoubleType())
    def pop_precision(actual_items):
        if not actual_items:
            return 0.0
        hits = len(set(top_k_popular) & set(actual_items))
        return hits / k

    @udf(DoubleType())
    def pop_recall(actual_items):
        if not actual_items:
            return 0.0
        hits = len(set(top_k_popular) & set(actual_items))
        return hits / len(actual_items)

    pop_metrics_df = ground_truth.withColumn(
        "pop_precision", pop_precision(col("actual_items"))
    ).withColumn(
        "pop_recall", pop_recall(col("actual_items"))
    )

    pop_avg = pop_metrics_df.agg(
        avg("pop_precision").alias("avg_precision"),
        avg("pop_recall").alias("avg_recall"),
    ).first()

    pop_metrics = {
        f"precision@{k}": round(pop_avg["avg_precision"], 4),
        f"recall@{k}": round(pop_avg["avg_recall"], 4),
    }

    # ─── Print Comparison ───
    print(f"\n{'=' * 55}")
    print(f"  RECOMMENDATION MODEL COMPARISON (K={k})")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<20} {'ALS':>12} {'Popularity':>12} {'Lift':>10}")
    print(f"  {'-' * 55}")
    for metric in [f"precision@{k}", f"recall@{k}"]:
        als_val = als_metrics[metric]
        pop_val = pop_metrics[metric]
        lift = ((als_val / pop_val) - 1) * 100 if pop_val > 0 else float('inf')
        print(f"  {metric:<20} {als_val:>12.4f} {pop_val:>12.4f} {lift:>9.1f}%")
    print(f"  {'ndcg@' + str(k):<20} {als_metrics[f'ndcg@{k}']:>12.4f} {'N/A':>12}")
    print(f"{'=' * 55}")

    return als_metrics, pop_metrics


def generate_sample_recommendations(model: ALSModel, spark: SparkSession, n_users: int = 5):
    """Generate and display sample recommendations."""
    print(f"\n[INFO] Sample Recommendations for {n_users} Users:")

    user_recs = model.recommendForAllUsers(als_config.top_k)

    sample = user_recs.limit(n_users).collect()
    for row in sample:
        user = row["user_idx"]
        recs = row["recommendations"]
        items = [(r["item_idx"], round(r["rating"], 3)) for r in recs[:5]]
        print(f"  User {user}: {items}")


def main():
    spark = create_spark_session()
    print("=" * 70)
    print("  COLLABORATIVE FILTERING — ALS RECOMMENDER")
    print("  Implicit Feedback: view=1, cart=3, purchase=5")
    print("=" * 70)

    os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)
    start_time = time.time()

    # 1. Build interaction matrix
    user_item = prepare_interaction_matrix(spark)

    # 2. Train/test split (temporal: last 20% of interactions per user)
    train_df, test_df = user_item.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    print(f"[INFO] Train: {train_df.count():,} | Test: {test_df.count():,}")

    # 3. Popularity baseline
    popular_items = build_popularity_baseline(spark, train_df)

    # 4. Train ALS
    model, train_time = train_als(train_df, test_df)

    # 5. Evaluate
    als_metrics, pop_metrics = evaluate_recommendations(model, test_df, popular_items, k=10)

    # 6. Sample recommendations
    generate_sample_recommendations(model, spark)

    # 7. Save model and metrics
    model_path = os.path.join(ML_ARTIFACTS_DIR, "als_model")
    model.write().overwrite().save(model_path)
    print(f"\n[INFO] ALS model saved to: {model_path}")

    results = {
        "als": {**als_metrics, "training_time_sec": train_time},
        "popularity_baseline": pop_metrics,
        "config": {
            "rank": als_config.rank,
            "max_iter": als_config.max_iter,
            "reg_param": als_config.reg_param,
            "alpha": als_config.alpha,
        }
    }
    with open(os.path.join(ML_ARTIFACTS_DIR, "als_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n[DONE] ALS recommender completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
