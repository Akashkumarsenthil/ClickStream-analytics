"""
MLflow Experiment Tracking Wrapper
===================================
Centralized MLflow tracking for all ML experiments.
Logs parameters, metrics, models, and artifacts for full reproducibility.

Usage:
    from ml.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker("CTR_Prediction")
    with tracker.start_run("gbt_v1"):
        tracker.log_params({"max_depth": 8, "max_iter": 50})
        tracker.log_metrics({"auc_roc": 0.78, "log_loss": 0.45})
        tracker.log_model(model, "gbt_model")
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MLFLOW_TRACKING_URI, ML_ARTIFACTS_DIR

try:
    import mlflow
    import mlflow.spark
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN] MLflow not installed. Tracking will use local JSON fallback.")


class MLflowTracker:
    """Wrapper for MLflow experiment tracking with JSON fallback."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.run_name = None
        self._run = None
        self._metrics = {}
        self._params = {}
        self._start_time = None

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)
            print(f"[MLflow] Experiment: {experiment_name}")
            print(f"[MLflow] Tracking URI: {MLFLOW_TRACKING_URI}")
        else:
            self._fallback_dir = os.path.join(ML_ARTIFACTS_DIR, "tracking", experiment_name)
            os.makedirs(self._fallback_dir, exist_ok=True)

    @contextmanager
    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """Start an MLflow run (or JSON fallback)."""
        self.run_name = run_name
        self._metrics = {}
        self._params = {}
        self._start_time = time.time()

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                self._run = run
                print(f"[MLflow] Run started: {run_name} (ID: {run.info.run_id})")
                yield self
                elapsed = time.time() - self._start_time
                mlflow.log_metric("total_time_sec", round(elapsed, 1))
                print(f"[MLflow] Run completed: {run_name} ({elapsed:.1f}s)")
        else:
            print(f"[Tracker] Run started: {run_name}")
            yield self
            elapsed = time.time() - self._start_time
            self._metrics["total_time_sec"] = round(elapsed, 1)
            self._save_fallback()
            print(f"[Tracker] Run completed: {run_name} ({elapsed:.1f}s)")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self._params.update(params)
        if MLFLOW_AVAILABLE and self._run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        print(f"[Tracker] Params logged: {list(params.keys())}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log evaluation metrics."""
        self._metrics.update(metrics)
        if MLFLOW_AVAILABLE and self._run:
            mlflow.log_metrics(metrics, step=step)
        for name, value in metrics.items():
            print(f"[Tracker] Metric: {name} = {value}")

    def log_metric(self, name: str, value: float, step: int = None):
        """Log a single metric."""
        self.log_metrics({name: value}, step=step)

    def log_model(self, model, artifact_path: str, model_type: str = "spark"):
        """Log a trained model."""
        if MLFLOW_AVAILABLE and self._run:
            if model_type == "spark":
                mlflow.spark.log_model(model, artifact_path)
            else:
                mlflow.pyfunc.log_model(artifact_path, python_model=model)
            print(f"[MLflow] Model logged: {artifact_path}")
        else:
            model_dir = os.path.join(ML_ARTIFACTS_DIR, artifact_path)
            try:
                model.write().overwrite().save(model_dir)
                print(f"[Tracker] Model saved: {model_dir}")
            except Exception as e:
                print(f"[Tracker] Model save failed: {e}")

    def log_artifact(self, filepath: str):
        """Log a file artifact."""
        if MLFLOW_AVAILABLE and self._run:
            mlflow.log_artifact(filepath)
            print(f"[MLflow] Artifact logged: {filepath}")

    def log_dict(self, data: dict, filename: str):
        """Log a dictionary as a JSON artifact."""
        filepath = os.path.join(ML_ARTIFACTS_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        self.log_artifact(filepath)

    def log_dataframe_profile(self, df, name: str):
        """Log basic DataFrame statistics."""
        stats = {
            "name": name,
            "row_count": df.count(),
            "column_count": len(df.columns),
            "columns": df.columns,
        }
        self.log_dict(stats, f"{name}_profile.json")

    def log_feature_importance(self, model, feature_names: list):
        """Log feature importance from a tree-based model."""
        try:
            importances = model.featureImportances.toArray()
            fi_data = sorted(
                [(name, float(imp)) for name, imp in zip(feature_names, importances)],
                key=lambda x: x[1], reverse=True
            )
            fi_dict = {name: imp for name, imp in fi_data[:20]}
            self.log_dict(
                {"top_features": fi_dict, "total_features": len(feature_names)},
                f"feature_importance_{self.run_name}.json"
            )
        except Exception as e:
            print(f"[Tracker] Feature importance logging failed: {e}")

    def _save_fallback(self):
        """Save tracking data as JSON when MLflow is not available."""
        run_data = {
            "experiment": self.experiment_name,
            "run_name": self.run_name,
            "params": self._params,
            "metrics": self._metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        filepath = os.path.join(self._fallback_dir, f"{self.run_name}.json")
        with open(filepath, "w") as f:
            json.dump(run_data, f, indent=2)
        print(f"[Tracker] Results saved: {filepath}")


class ExperimentComparison:
    """Compare metrics across multiple experiment runs."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    def get_all_runs(self) -> list:
        """Get all runs for an experiment."""
        if MLFLOW_AVAILABLE:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = mlflow.search_runs(experiment.experiment_id)
                return runs.to_dict("records")
            return []
        else:
            # Load from JSON fallback
            fallback_dir = os.path.join(ML_ARTIFACTS_DIR, "tracking", self.experiment_name)
            if not os.path.exists(fallback_dir):
                return []
            runs = []
            for f in os.listdir(fallback_dir):
                if f.endswith(".json"):
                    with open(os.path.join(fallback_dir, f)) as fh:
                        runs.append(json.load(fh))
            return runs

    def compare(self, metric_names: list = None) -> dict:
        """Compare runs by specified metrics."""
        runs = self.get_all_runs()
        if not runs:
            print(f"[Tracker] No runs found for experiment: {self.experiment_name}")
            return {}

        comparison = {}
        for run in runs:
            name = run.get("run_name", run.get("tags.mlflow.runName", "unknown"))
            metrics = run.get("metrics", {})
            if metric_names:
                metrics = {k: v for k, v in metrics.items() if k in metric_names}
            comparison[name] = metrics

        return comparison

    def print_comparison(self, metric_names: list = None):
        """Print formatted comparison table."""
        comparison = self.compare(metric_names)
        if not comparison:
            return

        all_metrics = set()
        for metrics in comparison.values():
            all_metrics.update(metrics.keys())

        all_metrics = sorted(all_metrics)

        # Header
        print(f"\n{'Metric':<25}", end="")
        for run_name in comparison:
            print(f" {run_name:>15}", end="")
        print()
        print("-" * (25 + 16 * len(comparison)))

        # Rows
        for metric in all_metrics:
            print(f"  {metric:<23}", end="")
            for run_name in comparison:
                val = comparison[run_name].get(metric, "N/A")
                if isinstance(val, float):
                    print(f" {val:>15.4f}", end="")
                else:
                    print(f" {str(val):>15}", end="")
            print()
