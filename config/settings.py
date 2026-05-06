"""
Central Configuration for E-Commerce Clickstream Analytics Pipeline
DATA 228 — Spring 2026 | Team 3
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ─────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")

# Dataset paths
CRITEO_RAW_DIR = os.path.join(RAW_DATA_DIR, "criteo")
REES46_RAW_DIR = os.path.join(RAW_DATA_DIR, "rees46")

CRITEO_BRONZE_DIR = os.path.join(BRONZE_DIR, "criteo")
REES46_BRONZE_DIR = os.path.join(BRONZE_DIR, "rees46")

# Table format paths
ICEBERG_WAREHOUSE = os.path.join(DATA_DIR, "iceberg_warehouse")
DELTA_PATH = os.path.join(DATA_DIR, "delta_tables")
HUDI_PATH = os.path.join(DATA_DIR, "hudi_tables")

# ML artifacts
ML_ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml_artifacts")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")

# Benchmark outputs
BENCHMARK_OUTPUT_DIR = os.path.join(BASE_DIR, "benchmark_results")

# ─────────────────────────────────────────────
# SPARK CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class SparkConfig:
    app_name: str = "ECommerceClickstreamAnalytics"
    master: str = "local[*]"
    driver_memory: str = "8g"
    executor_memory: str = "8g"
    shuffle_partitions: int = 200
    max_result_size: str = "4g"

    # Package versions
    iceberg_version: str = "1.5.0"
    delta_version: str = "3.1.0"
    hudi_version: str = "0.14.1"
    spark_version: str = "3.5"

    @property
    def packages(self) -> List[str]:
        return [
            f"org.apache.iceberg:iceberg-spark-runtime-{self.spark_version}_2.12:{self.iceberg_version}",
            f"io.delta:delta-spark_2.12:{self.delta_version}",
            f"org.apache.hudi:hudi-spark{self.spark_version}-bundle_2.12:{self.hudi_version}",
        ]

    @property
    def base_conf(self) -> Dict[str, str]:
        return {
            "spark.driver.memory": self.driver_memory,
            "spark.executor.memory": self.executor_memory,
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.driver.maxResultSize": self.max_result_size,
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
        }

    @property
    def iceberg_conf(self) -> Dict[str, str]:
        return {
            **self.base_conf,
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            "spark.sql.catalog.local": "org.apache.iceberg.spark.SparkCatalog",
            "spark.sql.catalog.local.type": "hadoop",
            "spark.sql.catalog.local.warehouse": ICEBERG_WAREHOUSE,
        }

    @property
    def delta_conf(self) -> Dict[str, str]:
        return {
            **self.base_conf,
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
            "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        }

    @property
    def hudi_conf(self) -> Dict[str, str]:
        return {
            **self.base_conf,
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.hive.convertMetastoreParquet": "false",
        }


# ─────────────────────────────────────────────
# CRITEO DATASET CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class CriteoConfig:
    """Criteo 1TB Click Logs configuration."""
    num_int_features: int = 13
    num_cat_features: int = 26
    label_col: str = "label"
    # Number of days to use (full dataset has 24 days, day_0 to day_23)
    num_days_subset: int = 5  # ~150 GB subset
    delimiter: str = "\t"
    # Feature names
    int_feature_prefix: str = "int_feature"
    cat_feature_prefix: str = "cat_feature"

    @property
    def int_feature_cols(self) -> List[str]:
        return [f"{self.int_feature_prefix}_{i}" for i in range(1, self.num_int_features + 1)]

    @property
    def cat_feature_cols(self) -> List[str]:
        return [f"{self.cat_feature_prefix}_{i}" for i in range(1, self.num_cat_features + 1)]

    @property
    def all_columns(self) -> List[str]:
        return [self.label_col] + self.int_feature_cols + self.cat_feature_cols


# ─────────────────────────────────────────────
# REES46 DATASET CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class REES46Config:
    """REES46 eCommerce Behavior Data configuration."""
    columns: List[str] = field(default_factory=lambda: [
        "event_time", "event_type", "product_id",
        "category_id", "category_code", "brand", "price", "user_id"
    ])
    event_types: List[str] = field(default_factory=lambda: ["view", "cart", "purchase"])
    # Implicit feedback weights for ALS
    implicit_weights: Dict[str, int] = field(default_factory=lambda: {
        "view": 1,
        "cart": 3,
        "purchase": 5,
    })
    # Files
    files: List[str] = field(default_factory=lambda: [
        "2019-Oct.csv",
        "2019-Nov.csv",
    ])


# ─────────────────────────────────────────────
# ML MODEL CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class CTRModelConfig:
    """CTR model training configuration."""
    test_size: float = 0.2
    feature_hash_buckets: int = 2**18  # 262144
    max_iter_lr: int = 100
    reg_param_lr: float = 0.01
    # GBT params
    gbt_max_depth: int = 8
    gbt_max_bins: int = 128
    gbt_max_iter: int = 50
    gbt_step_size: float = 0.1


@dataclass
class ALSConfig:
    """ALS collaborative filtering configuration."""
    rank: int = 64
    max_iter: int = 15
    reg_param: float = 0.1
    alpha: float = 1.0
    implicit_prefs: bool = True
    cold_start_strategy: str = "drop"
    top_k: int = 10


@dataclass
class Product2VecConfig:
    """Product2Vec (Word2Vec) configuration."""
    vector_size: int = 128
    min_count: int = 5
    window_size: int = 5
    max_iter: int = 10
    num_partitions: int = 8
    top_n_similar: int = 10
    # Session timeout in minutes
    session_timeout_minutes: int = 30


@dataclass
class SegmentationConfig:
    """Customer segmentation configuration."""
    k_range: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    optimal_k: int = 5
    max_iter: int = 100
    seed: int = 42
    features: List[str] = field(default_factory=lambda: [
        "total_sessions",
        "avg_session_duration_min",
        "total_views",
        "total_carts",
        "total_purchases",
        "avg_order_value",
        "cart_abandonment_rate",
        "category_diversity",
        "brand_diversity",
        "purchase_frequency",
    ])
    segment_labels: Dict[int, str] = field(default_factory=lambda: {
        0: "Window Shoppers",
        1: "Cart Abandoners",
        2: "Casual Buyers",
        3: "Loyal Whales",
        4: "Bargain Hunters",
    })


# ─────────────────────────────────────────────
# API CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 4
    title: str = "E-Commerce Clickstream Analytics API"
    version: str = "1.0.0"


# ─────────────────────────────────────────────
# INSTANTIATE CONFIGS
# ─────────────────────────────────────────────

spark_config = SparkConfig()
criteo_config = CriteoConfig()
rees46_config = REES46Config()
ctr_config = CTRModelConfig()
als_config = ALSConfig()
product2vec_config = Product2VecConfig()
segmentation_config = SegmentationConfig()
api_config = APIConfig()
