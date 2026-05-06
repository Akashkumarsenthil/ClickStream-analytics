"""
Product2Vec — Session-Based Next-Click Recommendations
=======================================================
Uses Word2Vec on browsing sessions to learn product embeddings
from co-browsing patterns. Products browsed in the same session
are treated like words in a sentence.

Usage:
    spark-submit ml/session/product2vec.py
"""

import sys
import os
import time
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg,
    when, lit, round as spark_round, collect_list, sort_array,
    struct, lag, unix_timestamp, size, explode, row_number,
    monotonically_increasing_id, desc, asc, array
)
from pyspark.sql.window import Window
from pyspark.ml.feature import Word2Vec, Word2VecModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    spark_config, product2vec_config,
    REES46_BRONZE_DIR, ML_ARTIFACTS_DIR
)


def create_spark_session() -> SparkSession:
    builder = SparkSession.builder \
        .appName("Product2Vec") \
        .master(spark_config.master)

    for key, value in spark_config.base_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def build_browsing_sessions(spark: SparkSession) -> DataFrame:
    """Build browsing sessions from REES46 events.
    
    A session is defined as a sequence of events by the same user
    where consecutive events are within 30 minutes of each other.
    Products within a session form a "sentence" for Word2Vec.
    """
    print("[INFO] Building browsing sessions...")

    events = spark.read.parquet(REES46_BRONZE_DIR)

    # Sort events by user and time
    user_window = Window.partitionBy("user_id").orderBy("event_time")

    events_ordered = events.withColumn(
        "prev_event_time", lag("event_time").over(user_window)
    ).withColumn(
        "time_diff_min",
        (unix_timestamp("event_time") - unix_timestamp("prev_event_time")) / 60.0
    )

    # Assign session IDs: new session if gap > 30 minutes or first event
    events_ordered = events_ordered.withColumn(
        "is_new_session",
        when(
            (col("prev_event_time").isNull()) |
            (col("time_diff_min") > product2vec_config.session_timeout_minutes),
            lit(1)
        ).otherwise(lit(0))
    )

    # Cumulative sum for session IDs
    session_window = Window.partitionBy("user_id").orderBy("event_time") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)

    events_ordered = events_ordered.withColumn(
        "session_id",
        spark_sum("is_new_session").over(session_window)
    )

    # Build session sequences: list of product_ids per session
    sessions = events_ordered.groupBy("user_id", "session_id").agg(
        collect_list(
            struct(col("event_time"), col("product_id").cast("string").alias("product_str"))
        ).alias("events"),
        count("*").alias("session_length"),
        spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
    )

    # Extract ordered product sequences
    # Sort by event_time within session and collect product_ids
    session_products = events_ordered.orderBy("event_time") \
        .groupBy("user_id", "session_id") \
        .agg(
            collect_list(col("product_id").cast("string")).alias("products"),
            count("*").alias("session_length"),
        )

    # Filter sessions with at least 2 products (need context for Word2Vec)
    session_products = session_products.filter(col("session_length") >= 2)

    total_sessions = session_products.count()
    avg_length = session_products.agg(avg("session_length")).first()[0]

    print(f"[INFO] Sessions built:")
    print(f"  Total sessions (≥2 products): {total_sessions:,}")
    print(f"  Average session length: {avg_length:.1f}")

    return session_products


def train_product2vec(sessions: DataFrame) -> tuple:
    """Train Word2Vec (Product2Vec) model on session sequences."""
    print("\n" + "=" * 50)
    print("  TRAINING PRODUCT2VEC (WORD2VEC)")
    print("=" * 50)

    start = time.time()

    word2vec = Word2Vec(
        vectorSize=product2vec_config.vector_size,
        minCount=product2vec_config.min_count,
        windowSize=product2vec_config.window_size,
        maxIter=product2vec_config.max_iter,
        numPartitions=product2vec_config.num_partitions,
        inputCol="products",
        outputCol="session_vector",
        seed=42
    )

    print(f"[INFO] Product2Vec params:")
    print(f"  Vector size: {product2vec_config.vector_size}")
    print(f"  Window size: {product2vec_config.window_size}")
    print(f"  Min count: {product2vec_config.min_count}")
    print(f"  Max iterations: {product2vec_config.max_iter}")

    model = word2vec.fit(sessions)

    elapsed = time.time() - start
    print(f"[INFO] Product2Vec training completed in {elapsed:.1f}s")

    # Vocabulary size
    vocab_size = model.getVectors().count()
    print(f"[INFO] Vocabulary size: {vocab_size:,} unique products")

    return model, elapsed


def find_similar_products(model: Word2VecModel, spark: SparkSession):
    """Find similar products using cosine similarity in embedding space."""
    print("\n" + "=" * 50)
    print("  SIMILAR PRODUCTS (Product2Vec)")
    print("=" * 50)

    # Get product vectors
    vectors = model.getVectors()
    print(f"[INFO] Product embeddings: {vectors.count():,} products")

    # Show sample product vectors
    print("\n[INFO] Sample product embeddings:")
    vectors.show(5, truncate=40)

    # Find similar products for sample items
    sample_products = vectors.limit(5).select("word").collect()

    for row in sample_products:
        product_id = row["word"]
        try:
            synonyms = model.findSynonyms(product_id, product2vec_config.top_n_similar)
            print(f"\n  Products similar to {product_id}:")
            similar = synonyms.collect()
            for sim in similar[:5]:
                print(f"    {sim['word']} (similarity: {sim['similarity']:.4f})")
        except Exception as e:
            print(f"  [WARN] Could not find similarities for {product_id}: {e}")


def evaluate_product2vec(
    model: Word2VecModel,
    sessions: DataFrame,
    spark: SparkSession
) -> dict:
    """Evaluate Product2Vec using hit rate and coverage.
    
    For each session, hide the last product and check if it appears
    in the top-K similar products of the second-to-last product.
    """
    print("\n[INFO] Evaluating Product2Vec...")

    # Sessions with at least 3 products for evaluation
    eval_sessions = sessions.filter(col("session_length") >= 3)

    # Get product vectors
    vectors = model.getVectors()
    product_set = set(row["word"] for row in vectors.collect())

    # Sample for evaluation (full eval can be expensive)
    sample = eval_sessions.limit(1000).collect()

    hits = 0
    total = 0

    for row in sample:
        products = row["products"]
        if len(products) < 3:
            continue

        query_product = products[-2]  # Second to last
        target_product = products[-1]  # Last (held out)

        if query_product not in product_set:
            continue

        try:
            synonyms = model.findSynonyms(query_product, 10)
            similar_ids = set(r["word"] for r in synonyms.collect())
            if target_product in similar_ids:
                hits += 1
            total += 1
        except Exception:
            continue

    hit_rate = hits / total if total > 0 else 0
    coverage = len(product_set) / sessions.select(
        explode("products")
    ).distinct().count()

    metrics = {
        "hit_rate@10": round(hit_rate, 4),
        "coverage": round(coverage, 4),
        "vocab_size": len(product_set),
        "sessions_evaluated": total,
    }

    print(f"\n  Hit Rate @10: {metrics['hit_rate@10']}")
    print(f"  Coverage: {metrics['coverage']}")
    print(f"  Sessions evaluated: {metrics['sessions_evaluated']}")

    return metrics


def main():
    spark = create_spark_session()
    print("=" * 70)
    print("  PRODUCT2VEC — SESSION-BASED NEXT-CLICK RECOMMENDATIONS")
    print("  Learning Product Embeddings from Co-Browsing Patterns")
    print("=" * 70)

    os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)
    start_time = time.time()

    # 1. Build browsing sessions
    sessions = build_browsing_sessions(spark)
    sessions.cache()

    # 2. Train Product2Vec
    model, train_time = train_product2vec(sessions)

    # 3. Find similar products
    find_similar_products(model, spark)

    # 4. Evaluate
    metrics = evaluate_product2vec(model, sessions, spark)

    # 5. Save model and results
    model_path = os.path.join(ML_ARTIFACTS_DIR, "product2vec_model")
    model.write().overwrite().save(model_path)
    print(f"\n[INFO] Product2Vec model saved to: {model_path}")

    results = {
        "metrics": metrics,
        "training_time_sec": round(train_time, 1),
        "config": {
            "vector_size": product2vec_config.vector_size,
            "window_size": product2vec_config.window_size,
            "min_count": product2vec_config.min_count,
            "max_iter": product2vec_config.max_iter,
        }
    }
    with open(os.path.join(ML_ARTIFACTS_DIR, "product2vec_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Product2Vec completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
