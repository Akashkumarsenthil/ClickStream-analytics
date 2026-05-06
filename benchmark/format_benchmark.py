"""
Table Format Benchmark — Iceberg vs Delta Lake vs Apache Hudi
==============================================================
Comparative analysis of query performance, write throughput,
and storage efficiency across all three open table formats.

Usage:
    spark-submit benchmark/format_benchmark.py
"""

import sys
import os
import time
import json
import shutil

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg,
    when, lit, round as spark_round, desc
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    spark_config, REES46_BRONZE_DIR,
    ICEBERG_WAREHOUSE, DELTA_PATH, HUDI_PATH,
    BENCHMARK_OUTPUT_DIR
)


def create_unified_session() -> SparkSession:
    """Create Spark session with all three format extensions."""
    builder = SparkSession.builder \
        .appName("FormatBenchmark") \
        .master(spark_config.master) \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true")

    # Iceberg
    builder = builder \
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.local.type", "hadoop") \
        .config("spark.sql.catalog.local.warehouse", ICEBERG_WAREHOUSE)

    return builder.getOrCreate()


def get_dir_size_mb(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)


def benchmark_write(spark: SparkSession, events_df: DataFrame) -> dict:
    """Benchmark write performance across all three formats."""
    print("\n" + "=" * 60)
    print("  WRITE BENCHMARK")
    print("=" * 60)

    results = {}
    benchmark_dir = os.path.join(BENCHMARK_OUTPUT_DIR, "tables")

    # ─── Parquet (Baseline) ───
    parquet_path = os.path.join(benchmark_dir, "parquet_bench")
    start = time.time()
    events_df.write.mode("overwrite").parquet(parquet_path)
    parquet_time = time.time() - start
    parquet_size = get_dir_size_mb(parquet_path)
    results["parquet"] = {
        "write_time_sec": round(parquet_time, 2),
        "storage_mb": parquet_size,
    }
    print(f"  Parquet:  {parquet_time:.2f}s | {parquet_size:.1f} MB")

    # ─── Delta Lake ───
    delta_bench_path = os.path.join(benchmark_dir, "delta_bench")
    start = time.time()
    events_df.write.format("delta").mode("overwrite").save(delta_bench_path)
    delta_time = time.time() - start
    delta_size = get_dir_size_mb(delta_bench_path)
    results["delta"] = {
        "write_time_sec": round(delta_time, 2),
        "storage_mb": delta_size,
    }
    print(f"  Delta:    {delta_time:.2f}s | {delta_size:.1f} MB")

    # ─── Apache Hudi ───
    hudi_bench_path = os.path.join(benchmark_dir, "hudi_bench")
    start = time.time()
    events_df.write \
        .format("hudi") \
        .option("hoodie.table.name", "bench_events") \
        .option("hoodie.datasource.write.recordkey.field", "user_id") \
        .option("hoodie.datasource.write.precombine.field", "event_day") \
        .option("hoodie.datasource.write.table.type", "COPY_ON_WRITE") \
        .option("hoodie.datasource.write.operation", "insert") \
        .mode("overwrite") \
        .save(hudi_bench_path)
    hudi_time = time.time() - start
    hudi_size = get_dir_size_mb(hudi_bench_path)
    results["hudi"] = {
        "write_time_sec": round(hudi_time, 2),
        "storage_mb": hudi_size,
    }
    print(f"  Hudi:     {hudi_time:.2f}s | {hudi_size:.1f} MB")

    # ─── Apache Iceberg ───
    start = time.time()
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.benchmark")
    events_df.writeTo("local.benchmark.events_bench").createOrReplace()
    iceberg_time = time.time() - start
    iceberg_path = os.path.join(ICEBERG_WAREHOUSE, "benchmark", "events_bench")
    iceberg_size = get_dir_size_mb(iceberg_path)
    results["iceberg"] = {
        "write_time_sec": round(iceberg_time, 2),
        "storage_mb": iceberg_size,
    }
    print(f"  Iceberg:  {iceberg_time:.2f}s | {iceberg_size:.1f} MB")

    return results


def benchmark_read_queries(spark: SparkSession) -> dict:
    """Benchmark read performance with standard analytical queries."""
    print("\n" + "=" * 60)
    print("  READ QUERY BENCHMARK")
    print("=" * 60)

    benchmark_dir = os.path.join(BENCHMARK_OUTPUT_DIR, "tables")
    results = {}

    queries = {
        "full_scan_count": lambda df: df.count(),
        "filter_purchases": lambda df: df.filter(col("event_type") == "purchase").count(),
        "group_by_category": lambda df: df.groupBy("main_category").agg(
            count("*").alias("cnt"),
            spark_sum("price").alias("revenue")
        ).collect(),
        "top_products": lambda df: df.filter(col("event_type") == "purchase") \
            .groupBy("product_id").agg(count("*").alias("purchases")) \
            .orderBy(desc("purchases")).limit(10).collect(),
        "user_activity": lambda df: df.groupBy("user_id").agg(
            countDistinct("event_date").alias("active_days"),
            count("*").alias("events")
        ).filter(col("events") > 100).count(),
    }

    formats = {
        "parquet": lambda: spark.read.parquet(os.path.join(benchmark_dir, "parquet_bench")),
        "delta": lambda: spark.read.format("delta").load(os.path.join(benchmark_dir, "delta_bench")),
        "hudi": lambda: spark.read.format("hudi").load(os.path.join(benchmark_dir, "hudi_bench")),
        "iceberg": lambda: spark.table("local.benchmark.events_bench"),
    }

    for fmt_name, reader_fn in formats.items():
        fmt_results = {}
        try:
            df = reader_fn()
            df.cache()
            df.count()  # Warm up cache

            for query_name, query_fn in queries.items():
                start = time.time()
                query_fn(df)
                elapsed = time.time() - start
                fmt_results[query_name] = round(elapsed, 3)

            df.unpersist()
        except Exception as e:
            print(f"  [WARN] {fmt_name} queries failed: {e}")
            fmt_results = {q: None for q in queries}

        results[fmt_name] = fmt_results

    # Print comparison table
    print(f"\n  {'Query':<25} {'Parquet':>10} {'Delta':>10} {'Hudi':>10} {'Iceberg':>10}")
    print("  " + "-" * 65)
    for query_name in queries:
        row = f"  {query_name:<25}"
        for fmt in ["parquet", "delta", "hudi", "iceberg"]:
            val = results[fmt].get(query_name)
            if val is not None:
                row += f" {val:>9.3f}s"
            else:
                row += f" {'N/A':>10}"
        print(row)

    return results


def benchmark_upsert(spark: SparkSession, events_df: DataFrame) -> dict:
    """Benchmark upsert/merge operations (Delta & Hudi only)."""
    print("\n" + "=" * 60)
    print("  UPSERT BENCHMARK (Delta & Hudi)")
    print("=" * 60)

    results = {}
    benchmark_dir = os.path.join(BENCHMARK_OUTPUT_DIR, "tables")

    # Create update batch: modify 10% of records
    update_batch = events_df.limit(int(events_df.count() * 0.1))
    update_batch = update_batch.withColumn("price", col("price") * 1.1)

    # ─── Delta Merge ───
    try:
        from delta.tables import DeltaTable

        delta_path = os.path.join(benchmark_dir, "delta_bench")
        delta_table = DeltaTable.forPath(spark, delta_path)

        start = time.time()
        delta_table.alias("target").merge(
            update_batch.alias("source"),
            "target.user_id = source.user_id AND target.product_id = source.product_id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        delta_merge_time = time.time() - start

        results["delta_merge"] = round(delta_merge_time, 2)
        print(f"  Delta Merge: {delta_merge_time:.2f}s")
    except Exception as e:
        print(f"  [WARN] Delta merge failed: {e}")
        results["delta_merge"] = None

    # ─── Hudi Upsert ───
    try:
        hudi_path = os.path.join(benchmark_dir, "hudi_bench")
        start = time.time()
        update_batch.write \
            .format("hudi") \
            .option("hoodie.table.name", "bench_events") \
            .option("hoodie.datasource.write.recordkey.field", "user_id") \
            .option("hoodie.datasource.write.precombine.field", "event_day") \
            .option("hoodie.datasource.write.operation", "upsert") \
            .mode("append") \
            .save(hudi_path)
        hudi_upsert_time = time.time() - start

        results["hudi_upsert"] = round(hudi_upsert_time, 2)
        print(f"  Hudi Upsert: {hudi_upsert_time:.2f}s")
    except Exception as e:
        print(f"  [WARN] Hudi upsert failed: {e}")
        results["hudi_upsert"] = None

    return results


def generate_report(write_results: dict, read_results: dict, upsert_results: dict):
    """Generate final benchmark report."""
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK REPORT")
    print("  Iceberg vs Delta Lake vs Apache Hudi")
    print("=" * 70)

    print("\n  ─── WRITE PERFORMANCE ───")
    print(f"  {'Format':<12} {'Time (s)':>10} {'Size (MB)':>12} {'Overhead':>10}")
    print("  " + "-" * 45)
    baseline_size = write_results.get("parquet", {}).get("storage_mb", 1)
    for fmt in ["parquet", "delta", "hudi", "iceberg"]:
        if fmt in write_results:
            t = write_results[fmt]["write_time_sec"]
            s = write_results[fmt]["storage_mb"]
            overhead = f"{((s / baseline_size) - 1) * 100:.1f}%" if baseline_size > 0 else "N/A"
            print(f"  {fmt:<12} {t:>10.2f} {s:>12.1f} {overhead:>10}")

    print("\n  ─── FEATURE COMPARISON ───")
    features = {
        "ACID Transactions": {"parquet": "No", "delta": "Yes", "hudi": "Yes", "iceberg": "Yes"},
        "Time Travel": {"parquet": "No", "delta": "Yes", "hudi": "Yes", "iceberg": "Yes"},
        "Schema Evolution": {"parquet": "Limited", "delta": "Yes", "hudi": "Yes", "iceberg": "Full"},
        "Upsert/Merge": {"parquet": "No", "delta": "Yes", "hudi": "Yes (MoR)", "iceberg": "Yes"},
        "Partition Evolution": {"parquet": "No", "delta": "No", "hudi": "No", "iceberg": "Yes"},
        "Hidden Partitioning": {"parquet": "No", "delta": "No", "hudi": "No", "iceberg": "Yes"},
        "Streaming Support": {"parquet": "Limited", "delta": "Native", "hudi": "Yes", "iceberg": "Yes"},
    }

    print(f"  {'Feature':<25} {'Parquet':>10} {'Delta':>10} {'Hudi':>10} {'Iceberg':>10}")
    print("  " + "-" * 65)
    for feature, vals in features.items():
        print(f"  {feature:<25} {vals['parquet']:>10} {vals['delta']:>10} "
              f"{vals['hudi']:>10} {vals['iceberg']:>10}")

    # Save full results
    os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)
    full_results = {
        "write_benchmark": write_results,
        "read_benchmark": read_results,
        "upsert_benchmark": upsert_results,
    }
    with open(os.path.join(BENCHMARK_OUTPUT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[INFO] Full results saved to: {BENCHMARK_OUTPUT_DIR}/benchmark_results.json")


def main():
    spark = create_unified_session()
    print("=" * 70)
    print("  TABLE FORMAT BENCHMARK")
    print("  Parquet vs Delta Lake vs Apache Hudi vs Apache Iceberg")
    print("=" * 70)

    os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # Load source data
    print("[INFO] Loading source data from Bronze layer...")
    events_df = spark.read.parquet(REES46_BRONZE_DIR)
    row_count = events_df.count()
    print(f"[INFO] Source data: {row_count:,} rows")

    events_df.cache()

    # 1. Write benchmark
    write_results = benchmark_write(spark, events_df)

    # 2. Read query benchmark
    read_results = benchmark_read_queries(spark)

    # 3. Upsert benchmark
    upsert_results = benchmark_upsert(spark, events_df)

    # 4. Generate report
    generate_report(write_results, read_results, upsert_results)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Benchmark completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
