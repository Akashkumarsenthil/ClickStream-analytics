"""
Integration Tests — E-Commerce Clickstream Analytics
=====================================================
Tests for data ingestion, transformations, and ML model outputs.

Usage:
    pytest tests/test_pipeline.py -v
"""

import os
import sys
import pytest
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Test configuration settings."""

    def test_config_imports(self):
        from config.settings import (
            spark_config, criteo_config, rees46_config,
            ctr_config, als_config, product2vec_config, segmentation_config
        )
        assert spark_config.app_name == "ECommerceClickstreamAnalytics"

    def test_criteo_schema(self):
        from config.settings import criteo_config
        assert len(criteo_config.int_feature_cols) == 13
        assert len(criteo_config.cat_feature_cols) == 26
        assert len(criteo_config.all_columns) == 40

    def test_rees46_config(self):
        from config.settings import rees46_config
        assert "view" in rees46_config.event_types
        assert "cart" in rees46_config.event_types
        assert "purchase" in rees46_config.event_types
        assert rees46_config.implicit_weights["view"] == 1
        assert rees46_config.implicit_weights["cart"] == 3
        assert rees46_config.implicit_weights["purchase"] == 5

    def test_spark_config_packages(self):
        from config.settings import spark_config
        packages = spark_config.packages
        assert len(packages) == 3
        assert any("iceberg" in p for p in packages)
        assert any("delta" in p for p in packages)
        assert any("hudi" in p for p in packages)

    def test_path_configuration(self):
        from config.settings import (
            BASE_DIR, DATA_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
            ICEBERG_WAREHOUSE, DELTA_PATH, HUDI_PATH
        )
        assert os.path.isabs(BASE_DIR)
        assert "data" in DATA_DIR


class TestCriteoIngestion:
    """Test Criteo data ingestion logic."""

    def test_schema_builder(self):
        from ingestion.criteo_ingest import build_criteo_schema
        schema = build_criteo_schema()
        assert len(schema.fields) == 40  # 1 label + 13 int + 26 cat
        assert schema.fields[0].name == "label"

    def test_int_features_present(self):
        from ingestion.criteo_ingest import build_criteo_schema
        schema = build_criteo_schema()
        field_names = [f.name for f in schema.fields]
        for i in range(1, 14):
            assert f"int_feature_{i}" in field_names

    def test_cat_features_present(self):
        from ingestion.criteo_ingest import build_criteo_schema
        schema = build_criteo_schema()
        field_names = [f.name for f in schema.fields]
        for i in range(1, 27):
            assert f"cat_feature_{i}" in field_names


class TestREES46Ingestion:
    """Test REES46 data ingestion logic."""

    def test_schema_builder(self):
        from ingestion.rees46_ingest import build_rees46_schema
        schema = build_rees46_schema()
        assert len(schema.fields) == 8
        field_names = [f.name for f in schema.fields]
        assert "event_time" in field_names
        assert "event_type" in field_names
        assert "product_id" in field_names
        assert "user_id" in field_names
        assert "price" in field_names


class TestMLConfigs:
    """Test ML model configurations."""

    def test_ctr_config(self):
        from config.settings import ctr_config
        assert ctr_config.test_size == 0.2
        assert ctr_config.feature_hash_buckets == 2**18
        assert ctr_config.gbt_max_depth == 8

    def test_als_config(self):
        from config.settings import als_config
        assert als_config.rank == 64
        assert als_config.implicit_prefs is True
        assert als_config.cold_start_strategy == "drop"
        assert als_config.top_k == 10

    def test_product2vec_config(self):
        from config.settings import product2vec_config
        assert product2vec_config.vector_size == 128
        assert product2vec_config.session_timeout_minutes == 30

    def test_segmentation_config(self):
        from config.settings import segmentation_config
        assert len(segmentation_config.features) == 10
        assert len(segmentation_config.segment_labels) == 5
        assert "Loyal Whales" in segmentation_config.segment_labels.values()
        assert "Cart Abandoners" in segmentation_config.segment_labels.values()


class TestAPI:
    """Test FastAPI endpoint definitions."""

    def test_api_imports(self):
        from api.main import app
        assert app.title == "E-Commerce Clickstream Analytics API"

    def test_api_routes(self):
        from api.main import app
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/recommend/{user_id}" in routes
        assert "/predict/ctr" in routes
        assert "/trending" in routes
        assert "/user/{user_id}/segment" in routes
        assert "/similar/{product_id}" in routes

    def test_segment_labels(self):
        from api.main import SEGMENT_LABELS
        assert len(SEGMENT_LABELS) == 5
        assert 0 in SEGMENT_LABELS
        assert SEGMENT_LABELS[3] == "Loyal Whales"


class TestArtifacts:
    """Test ML artifact file existence (after training)."""

    @pytest.fixture
    def artifacts_dir(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ml_artifacts"
        )

    def test_artifacts_dir_exists(self, artifacts_dir):
        """This test passes after pipeline has run."""
        if os.path.exists(artifacts_dir):
            assert os.path.isdir(artifacts_dir)
        else:
            pytest.skip("ML artifacts not yet generated")

    def test_ctr_comparison_json(self, artifacts_dir):
        path = os.path.join(artifacts_dir, "ctr_comparison.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            assert "logistic_regression" in data
            assert "gbt" in data
            assert "auc_roc" in data["gbt"]
        else:
            pytest.skip("CTR results not yet generated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
