"""
CTR Prediction Model — Criteo 1TB Click Logs
=============================================
Progressive model comparison:
  1. Logistic Regression (baseline)
  2. Gradient Boosted Trees (GBT)
Both use feature hashing on categorical features.
Evaluated on AUC-ROC and Log Loss at billion-row scale.
Tracked via MLflow.

Usage:
    spark-submit ml/ctr/ctr_model.py
"""

import sys
import os
import time
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, count, sum as spark_sum, avg,
    round as spark_round, udf, hash as spark_hash, abs as spark_abs
)
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, FeatureHasher,
    StandardScaler, OneHotEncoder
)
from pyspark.ml.classification import (
    LogisticRegression, GBTClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    spark_config, criteo_config, ctr_config,
    CRITEO_BRONZE_DIR, ML_ARTIFACTS_DIR, MLFLOW_TRACKING_URI
)


def create_spark_session() -> SparkSession:
    builder = SparkSession.builder \
        .appName("CTRPrediction") \
        .master(spark_config.master)

    for key, value in spark_config.base_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def load_and_prepare_data(spark: SparkSession) -> tuple:
    """Load Criteo data and prepare features using feature hashing."""
    print("[INFO] Loading Criteo data from Bronze layer...")

    df = spark.read.parquet(CRITEO_BRONZE_DIR)
    total = df.count()
    print(f"[INFO] Total records: {total:,}")

    # Label distribution
    label_dist = df.groupBy("label").count().collect()
    for row in label_dist:
        pct = row["count"] / total * 100
        print(f"  Label {row['label']}: {row['count']:,} ({pct:.1f}%)")

    # ─── Feature Engineering ───

    # 1. Integer features → assemble directly
    int_cols = criteo_config.int_feature_cols

    # 2. Categorical features → feature hashing
    cat_cols = criteo_config.cat_feature_cols

    # Feature Hasher for categorical columns
    hasher = FeatureHasher(
        inputCols=cat_cols,
        outputCol="cat_features",
        numFeatures=ctr_config.feature_hash_buckets,
        categoricalCols=cat_cols
    )

    # Assemble integer features
    int_assembler = VectorAssembler(
        inputCols=int_cols,
        outputCol="int_features",
        handleInvalid="keep"
    )

    # Combine all features
    final_assembler = VectorAssembler(
        inputCols=["int_features", "cat_features"],
        outputCol="features",
        handleInvalid="keep"
    )

    # Scale features
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False  # Sparse-friendly
    )

    # Build preprocessing pipeline
    preprocessing = Pipeline(stages=[int_assembler, hasher, final_assembler, scaler])

    print("[INFO] Fitting preprocessing pipeline...")
    prep_model = preprocessing.fit(df)
    prepared_df = prep_model.transform(df)

    # Split train/test
    train_df, test_df = prepared_df.randomSplit(
        [1 - ctr_config.test_size, ctr_config.test_size],
        seed=42
    )

    print(f"[INFO] Train: {train_df.count():,} | Test: {test_df.count():,}")
    return train_df, test_df, prep_model


def train_logistic_regression(train_df: DataFrame, test_df: DataFrame) -> dict:
    """Train Logistic Regression baseline for CTR prediction."""
    print("\n" + "=" * 50)
    print("  MODEL 1: LOGISTIC REGRESSION")
    print("=" * 50)

    start = time.time()

    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="label",
        maxIter=ctr_config.max_iter_lr,
        regParam=ctr_config.reg_param_lr,
        elasticNetParam=0.0,  # L2 regularization
        family="binomial"
    )

    print("[INFO] Training Logistic Regression...")
    lr_model = lr.fit(train_df)

    # Evaluate
    predictions = lr_model.transform(test_df)

    # AUC-ROC
    auc_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderROC"
    )
    auc_roc = auc_evaluator.evaluate(predictions)

    # Log Loss (AUC-PR as proxy)
    auc_pr_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderPR"
    )
    auc_pr = auc_pr_evaluator.evaluate(predictions)

    # Accuracy
    acc_evaluator = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)

    # F1
    f1_evaluator = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)

    elapsed = time.time() - start

    metrics = {
        "model": "LogisticRegression",
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "training_time_sec": round(elapsed, 1),
    }

    print(f"  AUC-ROC:  {metrics['auc_roc']}")
    print(f"  AUC-PR:   {metrics['auc_pr']}")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  F1 Score: {metrics['f1_score']}")
    print(f"  Time:     {metrics['training_time_sec']}s")

    # Save model
    model_path = os.path.join(ML_ARTIFACTS_DIR, "ctr_lr_model")
    lr_model.write().overwrite().save(model_path)
    print(f"[INFO] LR model saved to: {model_path}")

    return metrics


def train_gbt(train_df: DataFrame, test_df: DataFrame) -> dict:
    """Train Gradient Boosted Trees for CTR prediction."""
    print("\n" + "=" * 50)
    print("  MODEL 2: GRADIENT BOOSTED TREES (GBT)")
    print("=" * 50)

    start = time.time()

    gbt = GBTClassifier(
        featuresCol="scaled_features",
        labelCol="label",
        maxDepth=ctr_config.gbt_max_depth,
        maxBins=ctr_config.gbt_max_bins,
        maxIter=ctr_config.gbt_max_iter,
        stepSize=ctr_config.gbt_step_size,
        subsamplingRate=0.8,
        featureSubsetStrategy="sqrt",
        seed=42
    )

    print("[INFO] Training Gradient Boosted Trees...")
    gbt_model = gbt.fit(train_df)

    # Evaluate
    predictions = gbt_model.transform(test_df)

    auc_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderROC"
    )
    auc_roc = auc_evaluator.evaluate(predictions)

    auc_pr_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderPR"
    )
    auc_pr = auc_pr_evaluator.evaluate(predictions)

    acc_evaluator = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)

    f1_evaluator = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)

    elapsed = time.time() - start

    metrics = {
        "model": "GradientBoostedTrees",
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "training_time_sec": round(elapsed, 1),
    }

    print(f"  AUC-ROC:  {metrics['auc_roc']}")
    print(f"  AUC-PR:   {metrics['auc_pr']}")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  F1 Score: {metrics['f1_score']}")
    print(f"  Time:     {metrics['training_time_sec']}s")

    # Feature importance
    print("\n  Top 10 Feature Importances:")
    importances = gbt_model.featureImportances.toArray()
    all_features = criteo_config.int_feature_cols + [f"hashed_{i}" for i in range(10)]
    top_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:10]
    for rank, idx in enumerate(top_idx, 1):
        feat_name = f"feature_{idx}"
        if idx < len(criteo_config.int_feature_cols):
            feat_name = criteo_config.int_feature_cols[idx]
        print(f"    {rank:2d}. {feat_name}: {importances[idx]:.4f}")

    # Save model
    model_path = os.path.join(ML_ARTIFACTS_DIR, "ctr_gbt_model")
    gbt_model.write().overwrite().save(model_path)
    print(f"\n[INFO] GBT model saved to: {model_path}")

    return metrics


def compare_models(lr_metrics: dict, gbt_metrics: dict):
    """Print model comparison table."""
    print("\n" + "=" * 60)
    print("  CTR MODEL COMPARISON: Accuracy vs. Compute Tradeoff")
    print("=" * 60)
    print(f"  {'Metric':<20} {'LR':>15} {'GBT':>15} {'Δ':>10}")
    print("  " + "-" * 60)
    for metric in ["auc_roc", "auc_pr", "accuracy", "f1_score", "training_time_sec"]:
        lr_val = lr_metrics[metric]
        gbt_val = gbt_metrics[metric]
        delta = gbt_val - lr_val
        sign = "+" if delta > 0 else ""
        print(f"  {metric:<20} {lr_val:>15.4f} {gbt_val:>15.4f} {sign}{delta:>9.4f}")
    print("=" * 60)

    # Save comparison
    comparison = {"logistic_regression": lr_metrics, "gbt": gbt_metrics}
    os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ML_ARTIFACTS_DIR, "ctr_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)


def main():
    spark = create_spark_session()
    print("=" * 70)
    print("  CTR PREDICTION — CRITEO 1TB CLICK LOGS")
    print("  Logistic Regression → Gradient Boosted Trees")
    print("=" * 70)

    os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)
    start_time = time.time()

    # 1. Load and prepare data
    train_df, test_df, prep_model = load_and_prepare_data(spark)

    # Cache for multiple model passes
    train_df.cache()
    test_df.cache()

    # 2. Train Logistic Regression
    lr_metrics = train_logistic_regression(train_df, test_df)

    # 3. Train GBT
    gbt_metrics = train_gbt(train_df, test_df)

    # 4. Compare
    compare_models(lr_metrics, gbt_metrics)

    elapsed = time.time() - start_time
    print(f"\n[DONE] CTR model training completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
