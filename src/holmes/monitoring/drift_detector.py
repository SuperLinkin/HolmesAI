"""Model drift detection and monitoring."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from loguru import logger
from collections import deque
import json
from pathlib import Path


class DriftDetector:
    """
    Detect model drift through rolling performance metrics.

    Triggers alert if accuracy drops > 3%.
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_threshold: float = 0.03,
        metrics_file: str = "data/metrics_history.jsonl"
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Size of rolling window for metrics
            alert_threshold: Threshold for drift alert (3% default)
            metrics_file: File to store metrics history
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.metrics_file = Path(metrics_file)

        # Rolling windows for predictions
        self.predictions_window = deque(maxlen=window_size)
        self.ground_truth_window = deque(maxlen=window_size)

        # Baseline metrics
        self.baseline_f1 = None
        self.baseline_precision = None
        self.baseline_recall = None

        # Create metrics file
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.metrics_file.exists():
            self.metrics_file.touch()

    def set_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Set baseline metrics from initial evaluation.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        """
        self.baseline_f1 = f1_score(y_true, y_pred, average="macro")
        self.baseline_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        self.baseline_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        logger.info(
            f"Baseline metrics set: F1={self.baseline_f1:.4f}, "
            f"Precision={self.baseline_precision:.4f}, "
            f"Recall={self.baseline_recall:.4f}"
        )

        self._log_metrics({
            "type": "baseline",
            "f1": self.baseline_f1,
            "precision": self.baseline_precision,
            "recall": self.baseline_recall,
            "timestamp": datetime.now().isoformat()
        })

    def add_prediction(
        self,
        y_true: str,
        y_pred: str
    ):
        """
        Add a new prediction to the rolling window.

        Args:
            y_true: Ground truth label
            y_pred: Predicted label
        """
        self.predictions_window.append(y_pred)
        self.ground_truth_window.append(y_true)

    def add_batch(
        self,
        y_true: List[str],
        y_pred: List[str]
    ):
        """
        Add batch of predictions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        """
        for true_label, pred_label in zip(y_true, y_pred):
            self.add_prediction(true_label, pred_label)

    def compute_current_metrics(self) -> Dict[str, float]:
        """
        Compute metrics on current rolling window.

        Returns:
            Dictionary with current metrics
        """
        if len(self.predictions_window) < 10:
            logger.warning("Insufficient data for metrics computation")
            return {}

        y_true = list(self.ground_truth_window)
        y_pred = list(self.predictions_window)

        current_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        current_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        current_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "f1": current_f1,
            "precision": current_precision,
            "recall": current_recall,
            "n_samples": len(y_true)
        }

    def detect_drift(self) -> Dict:
        """
        Detect if model drift has occurred.

        Returns:
            Dictionary with drift status and details
        """
        if not self.baseline_f1:
            return {
                "drift_detected": False,
                "reason": "baseline_not_set"
            }

        current_metrics = self.compute_current_metrics()

        if not current_metrics:
            return {
                "drift_detected": False,
                "reason": "insufficient_data"
            }

        # Calculate drift
        f1_drift = self.baseline_f1 - current_metrics["f1"]
        precision_drift = self.baseline_precision - current_metrics["precision"]
        recall_drift = self.baseline_recall - current_metrics["recall"]

        # Check if drift exceeds threshold
        drift_detected = abs(f1_drift) > self.alert_threshold

        result = {
            "drift_detected": drift_detected,
            "baseline_f1": self.baseline_f1,
            "current_f1": current_metrics["f1"],
            "f1_drift": f1_drift,
            "precision_drift": precision_drift,
            "recall_drift": recall_drift,
            "alert_threshold": self.alert_threshold,
            "timestamp": datetime.now().isoformat()
        }

        if drift_detected:
            logger.warning(
                f"⚠️ DRIFT DETECTED: F1 dropped by {f1_drift:.4f} "
                f"({f1_drift/self.baseline_f1*100:.2f}%)"
            )

        # Log metrics
        self._log_metrics({
            "type": "drift_check",
            **result
        })

        return result

    def _log_metrics(self, metrics: Dict):
        """Log metrics to file."""
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def get_metrics_history(
        self,
        since: datetime = None,
        limit: int = None
    ) -> List[Dict]:
        """
        Retrieve metrics history.

        Args:
            since: Only return metrics after this date
            limit: Maximum records to return

        Returns:
            List of metrics dictionaries
        """
        history = []

        if not self.metrics_file.exists():
            return history

        with open(self.metrics_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)

                    # Filter by date
                    if since and "timestamp" in entry:
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time < since:
                            continue

                    history.append(entry)

                    # Limit results
                    if limit and len(history) >= limit:
                        break

        return history

    def generate_report(self) -> Dict:
        """
        Generate comprehensive drift report.

        Returns:
            Dictionary with drift analysis
        """
        current_metrics = self.compute_current_metrics()
        drift_status = self.detect_drift()

        # Get recent history
        recent_history = self.get_metrics_history(
            since=datetime.now() - timedelta(days=30)
        )

        # Calculate trends
        drift_checks = [
            h for h in recent_history
            if h.get("type") == "drift_check"
        ]

        avg_f1 = (
            sum(d["current_f1"] for d in drift_checks) / len(drift_checks)
            if drift_checks else 0
        )

        report = {
            "current_metrics": current_metrics,
            "baseline_metrics": {
                "f1": self.baseline_f1,
                "precision": self.baseline_precision,
                "recall": self.baseline_recall
            },
            "drift_status": drift_status,
            "rolling_window_size": len(self.predictions_window),
            "recent_checks": len(drift_checks),
            "avg_f1_30d": avg_f1,
            "timestamp": datetime.now().isoformat()
        }

        return report


class PerformanceMonitor:
    """Monitor model performance in production."""

    def __init__(
        self,
        drift_detector: DriftDetector = None
    ):
        """
        Initialize performance monitor.

        Args:
            drift_detector: Drift detector instance
        """
        self.drift_detector = drift_detector or DriftDetector()
        self.prediction_count = 0
        self.low_confidence_count = 0
        self.category_distribution = {}

    def log_prediction(
        self,
        category: str,
        confidence: float,
        ground_truth: str = None
    ):
        """
        Log a prediction for monitoring.

        Args:
            category: Predicted category
            confidence: Prediction confidence
            ground_truth: True category (if available)
        """
        self.prediction_count += 1

        # Track low confidence
        if confidence < 0.7:
            self.low_confidence_count += 1

        # Track category distribution
        if category not in self.category_distribution:
            self.category_distribution[category] = 0
        self.category_distribution[category] += 1

        # Add to drift detector if ground truth available
        if ground_truth and self.drift_detector:
            self.drift_detector.add_prediction(ground_truth, category)

    def get_statistics(self) -> Dict:
        """
        Get monitoring statistics.

        Returns:
            Dictionary with monitoring stats
        """
        low_confidence_rate = (
            self.low_confidence_count / self.prediction_count
            if self.prediction_count > 0
            else 0
        )

        return {
            "total_predictions": self.prediction_count,
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_rate": low_confidence_rate,
            "category_distribution": self.category_distribution,
            "timestamp": datetime.now().isoformat()
        }

    def reset_statistics(self):
        """Reset monitoring statistics."""
        self.prediction_count = 0
        self.low_confidence_count = 0
        self.category_distribution = {}

        logger.info("Monitoring statistics reset")
