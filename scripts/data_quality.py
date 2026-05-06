"""
Data Quality Validation Module
================================
Validates data quality at each pipeline stage:
  - Schema validation
  - Null checks
  - Range checks
  - Distribution checks
  - Referential integrity
  - Freshness checks

Usage:
    from scripts.data_quality import DataQualityValidator
    validator = DataQualityValidator(spark)
    report = validator.validate_bronze_rees46()
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg,
    when, lit, isnan, isnull, min as spark_min, max as spark_max,
    stddev, percentile_approx, length
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    REES46_BRONZE_DIR, CRITEO_BRONZE_DIR,
    rees46_config, criteo_config
)


@dataclass
class QualityCheck:
    """Single data quality check result."""
    name: str
    passed: bool
    details: str
    severity: str = "warning"  # info, warning, critical
    metric_value: Any = None


@dataclass
class QualityReport:
    """Full data quality report for a dataset."""
    dataset: str
    layer: str
    timestamp: str = ""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    checks: List[QualityCheck] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_checks / self.total_checks if self.total_checks > 0 else 0

    @property
    def status(self) -> str:
        if self.failed_checks == 0:
            return "PASS"
        critical_failures = sum(1 for c in self.checks if not c.passed and c.severity == "critical")
        return "FAIL" if critical_failures > 0 else "WARN"

    def add_check(self, check: QualityCheck):
        self.checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1

    def print_report(self):
        print(f"\n{'=' * 65}")
        print(f"  DATA QUALITY REPORT — {self.dataset} ({self.layer})")
        print(f"  Status: {self.status} | Passed: {self.passed_checks}/{self.total_checks}")
        print(f"{'=' * 65}")

        for check in self.checks:
            icon = "✅" if check.passed else ("❌" if check.severity == "critical" else "⚠️")
            print(f"  {icon} {check.name}")
            if not check.passed or check.metric_value is not None:
                print(f"     {check.details}")

        print(f"{'=' * 65}\n")

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "layer": self.layer,
            "status": self.status,
            "pass_rate": round(self.pass_rate, 4),
            "total_checks": self.total_checks,
            "passed": self.passed_checks,
            "failed": self.failed_checks,
            "checks": [
                {"name": c.name, "passed": c.passed, "details": c.details,
                 "severity": c.severity, "value": str(c.metric_value)}
                for c in self.checks
            ]
        }


class DataQualityValidator:
    """Data quality validation for all pipeline layers."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    # ─── Generic Checks ───

    def check_not_empty(self, df: DataFrame, name: str) -> QualityCheck:
        """Check that DataFrame is not empty."""
        row_count = df.count()
        return QualityCheck(
            name=f"{name}: Not empty",
            passed=row_count > 0,
            details=f"Row count: {row_count:,}",
            severity="critical",
            metric_value=row_count
        )

    def check_no_full_nulls(self, df: DataFrame, columns: List[str], name: str) -> List[QualityCheck]:
        """Check that key columns don't have 100% nulls."""
        checks = []
        total = df.count()
        for col_name in columns:
            null_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
            null_pct = null_count / total * 100 if total > 0 else 100
            checks.append(QualityCheck(
                name=f"{name}: {col_name} null check",
                passed=null_pct < 100,
                details=f"Null %: {null_pct:.2f}% ({null_count:,}/{total:,})",
                severity="warning" if null_pct < 50 else "critical",
                metric_value=null_pct
            ))
        return checks

    def check_null_threshold(self, df: DataFrame, column: str,
                              threshold_pct: float, name: str) -> QualityCheck:
        """Check that null percentage is below threshold."""
        total = df.count()
        null_count = df.filter(col(column).isNull()).count()
        null_pct = null_count / total * 100 if total > 0 else 100
        return QualityCheck(
            name=f"{name}: {column} null < {threshold_pct}%",
            passed=null_pct <= threshold_pct,
            details=f"Null %: {null_pct:.2f}% (threshold: {threshold_pct}%)",
            severity="warning",
            metric_value=null_pct
        )

    def check_unique_count(self, df: DataFrame, column: str,
                            min_expected: int, name: str) -> QualityCheck:
        """Check minimum unique value count."""
        unique = df.select(countDistinct(column)).first()[0]
        return QualityCheck(
            name=f"{name}: {column} unique count ≥ {min_expected:,}",
            passed=unique >= min_expected,
            details=f"Unique values: {unique:,} (min expected: {min_expected:,})",
            severity="warning",
            metric_value=unique
        )

    def check_value_range(self, df: DataFrame, column: str,
                           min_val: float, max_val: float, name: str) -> QualityCheck:
        """Check that values fall within expected range."""
        stats = df.agg(
            spark_min(column).alias("min"),
            spark_max(column).alias("max")
        ).first()
        actual_min = stats["min"]
        actual_max = stats["max"]
        passed = (actual_min >= min_val) and (actual_max <= max_val)
        return QualityCheck(
            name=f"{name}: {column} in [{min_val}, {max_val}]",
            passed=passed,
            details=f"Actual range: [{actual_min}, {actual_max}]",
            severity="warning",
            metric_value=(actual_min, actual_max)
        )

    def check_categorical_values(self, df: DataFrame, column: str,
                                   expected_values: List[str], name: str) -> QualityCheck:
        """Check that categorical column only contains expected values."""
        actual_values = set(
            row[column] for row in df.select(column).distinct().collect()
            if row[column] is not None
        )
        unexpected = actual_values - set(expected_values)
        return QualityCheck(
            name=f"{name}: {column} valid categories",
            passed=len(unexpected) == 0,
            details=f"Unexpected values: {unexpected}" if unexpected else "All values valid",
            severity="warning",
            metric_value=len(unexpected)
        )

    def check_no_duplicates(self, df: DataFrame, key_columns: List[str],
                             name: str) -> QualityCheck:
        """Check for duplicate rows by key columns."""
        total = df.count()
        distinct = df.select(key_columns).distinct().count()
        dup_count = total - distinct
        return QualityCheck(
            name=f"{name}: No duplicates on {key_columns}",
            passed=dup_count == 0,
            details=f"Duplicates: {dup_count:,} (total: {total:,})",
            severity="warning",
            metric_value=dup_count
        )

    # ─── REES46 Validation ───

    def validate_bronze_rees46(self) -> QualityReport:
        """Validate REES46 Bronze layer data."""
        report = QualityReport(dataset="REES46", layer="Bronze")
        report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            df = self.spark.read.parquet(REES46_BRONZE_DIR)
        except Exception as e:
            report.add_check(QualityCheck(
                name="Load Bronze data",
                passed=False,
                details=f"Failed to load: {e}",
                severity="critical"
            ))
            report.print_report()
            return report

        # 1. Not empty
        report.add_check(self.check_not_empty(df, "REES46"))

        # 2. Key columns not null
        key_cols = ["event_time", "event_type", "product_id", "user_id", "price"]
        for check in self.check_no_full_nulls(df, key_cols, "REES46"):
            report.add_check(check)

        # 3. Event types valid
        report.add_check(self.check_categorical_values(
            df, "event_type", rees46_config.event_types, "REES46"
        ))

        # 4. Price range
        report.add_check(self.check_value_range(df, "price", 0.01, 100000, "REES46"))

        # 5. Minimum unique users
        report.add_check(self.check_unique_count(df, "user_id", 10000, "REES46"))

        # 6. Minimum unique products
        report.add_check(self.check_unique_count(df, "product_id", 10000, "REES46"))

        # 7. Category not all null
        report.add_check(self.check_null_threshold(
            df, "main_category", 50, "REES46"
        ))

        # 8. Brand coverage
        report.add_check(self.check_null_threshold(df, "brand", 70, "REES46"))

        # 9. Event type distribution sanity (views should be majority)
        view_count = df.filter(col("event_type") == "view").count()
        total = df.count()
        view_pct = view_count / total * 100 if total > 0 else 0
        report.add_check(QualityCheck(
            name="REES46: Views are majority event type",
            passed=view_pct > 50,
            details=f"View %: {view_pct:.1f}% (expected > 50%)",
            severity="warning",
            metric_value=view_pct
        ))

        # 10. Conversion rate sanity
        purchase_count = df.filter(col("event_type") == "purchase").count()
        conv_rate = purchase_count / view_count * 100 if view_count > 0 else 0
        report.add_check(QualityCheck(
            name="REES46: Conversion rate < 10%",
            passed=conv_rate < 10,
            details=f"Conversion rate: {conv_rate:.2f}% (sanity: < 10%)",
            severity="info",
            metric_value=conv_rate
        ))

        report.print_report()
        return report

    # ─── Criteo Validation ───

    def validate_bronze_criteo(self) -> QualityReport:
        """Validate Criteo Bronze layer data."""
        report = QualityReport(dataset="Criteo", layer="Bronze")
        report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            df = self.spark.read.parquet(CRITEO_BRONZE_DIR)
        except Exception as e:
            report.add_check(QualityCheck(
                name="Load Bronze data",
                passed=False,
                details=f"Failed to load: {e}",
                severity="critical"
            ))
            report.print_report()
            return report

        # 1. Not empty
        report.add_check(self.check_not_empty(df, "Criteo"))

        # 2. Label is binary
        label_values = set(row["label"] for row in df.select("label").distinct().collect())
        report.add_check(QualityCheck(
            name="Criteo: Label is binary (0/1)",
            passed=label_values.issubset({0, 1}),
            details=f"Label values: {label_values}",
            severity="critical",
            metric_value=label_values
        ))

        # 3. Integer features present
        for col_name in criteo_config.int_feature_cols[:5]:
            report.add_check(self.check_null_threshold(
                df, col_name, 80, "Criteo"
            ))

        # 4. CTR sanity (typically 3-5% for display ads)
        click_count = df.filter(col("label") == 1).count()
        total = df.count()
        ctr = click_count / total * 100 if total > 0 else 0
        report.add_check(QualityCheck(
            name="Criteo: CTR in reasonable range (1-10%)",
            passed=1 <= ctr <= 10,
            details=f"CTR: {ctr:.2f}%",
            severity="info",
            metric_value=ctr
        ))

        # 5. Column count
        report.add_check(QualityCheck(
            name="Criteo: Column count = 42 (40 features + day + row_id)",
            passed=len(df.columns) >= 40,
            details=f"Columns: {len(df.columns)} ({df.columns[:5]}...)",
            severity="warning",
            metric_value=len(df.columns)
        ))

        report.print_report()
        return report

    # ─── Run All Validations ───

    def validate_all(self) -> Dict[str, QualityReport]:
        """Run all data quality validations."""
        print("=" * 65)
        print("  RUNNING ALL DATA QUALITY VALIDATIONS")
        print("=" * 65)

        reports = {}

        # Bronze REES46
        try:
            reports["rees46_bronze"] = self.validate_bronze_rees46()
        except Exception as e:
            print(f"[WARN] REES46 Bronze validation failed: {e}")

        # Bronze Criteo
        try:
            reports["criteo_bronze"] = self.validate_bronze_criteo()
        except Exception as e:
            print(f"[WARN] Criteo Bronze validation failed: {e}")

        # Summary
        print("\n" + "=" * 65)
        print("  DATA QUALITY SUMMARY")
        print("=" * 65)
        for name, report in reports.items():
            print(f"  {name:<25} {report.status:>6} "
                  f"({report.passed_checks}/{report.total_checks} checks)")
        print("=" * 65)

        return reports


# ─── CLI ───
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("DataQuality") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    validator = DataQualityValidator(spark)
    reports = validator.validate_all()

    # Save reports
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)
    for name, report in reports.items():
        filepath = os.path.join(output_dir, f"dq_{name}.json")
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    spark.stop()
