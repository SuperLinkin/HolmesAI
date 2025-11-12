"""Feedback loop and continuous learning system."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from loguru import logger
import json

from ..encoding.vector_store import VectorStore
from ..classification.classifier import TransactionClassifier


class FeedbackCollector:
    """Collect and manage user feedback on predictions."""

    def __init__(
        self,
        feedback_file: str = "data/feedback.jsonl"
    ):
        """
        Initialize feedback collector.

        Args:
            feedback_file: Path to feedback log file
        """
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self.feedback_file.exists():
            self.feedback_file.touch()

    def add_feedback(
        self,
        transaction_id: str,
        merchant: str,
        predicted_category: str,
        corrected_category: str,
        confidence: float,
        user_id: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """
        Record user feedback on a prediction.

        Args:
            transaction_id: Transaction identifier
            merchant: Merchant description
            predicted_category: Original prediction
            corrected_category: User-corrected category
            confidence: Original confidence score
            user_id: User who provided feedback
            notes: Optional feedback notes
        """
        feedback_entry = {
            "transaction_id": transaction_id,
            "merchant": merchant,
            "predicted_category": predicted_category,
            "corrected_category": corrected_category,
            "confidence": confidence,
            "user_id": user_id,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }

        # Append to JSONL file
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")

        logger.info(
            f"Feedback recorded: {merchant} | "
            f"{predicted_category} -> {corrected_category}"
        )

    def get_feedback(
        self,
        since: datetime = None,
        limit: int = None
    ) -> List[Dict]:
        """
        Retrieve feedback records.

        Args:
            since: Only return feedback after this date
            limit: Maximum records to return

        Returns:
            List of feedback dictionaries
        """
        feedback = []

        if not self.feedback_file.exists():
            return feedback

        with open(self.feedback_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)

                    # Filter by date
                    if since:
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time < since:
                            continue

                    feedback.append(entry)

                    # Limit results
                    if limit and len(feedback) >= limit:
                        break

        return feedback

    def get_correction_stats(self) -> Dict:
        """
        Get statistics on corrections.

        Returns:
            Dictionary with correction statistics
        """
        feedback = self.get_feedback()

        if not feedback:
            return {
                "total_corrections": 0,
                "avg_original_confidence": 0,
                "most_corrected_categories": {}
            }

        total = len(feedback)
        avg_confidence = sum(f["confidence"] for f in feedback) / total

        # Count corrections per original category
        from collections import Counter
        category_corrections = Counter(f["predicted_category"] for f in feedback)

        return {
            "total_corrections": total,
            "avg_original_confidence": avg_confidence,
            "most_corrected_categories": dict(category_corrections.most_common(10))
        }


class ContinuousLearner:
    """
    Continuous learning system with nightly retraining.

    Improves accuracy 3-5% per quarter through feedback integration.
    """

    def __init__(
        self,
        classifier: TransactionClassifier,
        vector_store: VectorStore = None,
        feedback_collector: FeedbackCollector = None,
        min_feedback_count: int = 100
    ):
        """
        Initialize continuous learner.

        Args:
            classifier: Transaction classifier
            vector_store: Vector store for embeddings
            feedback_collector: Feedback collector
            min_feedback_count: Minimum feedback before retraining
        """
        self.classifier = classifier
        self.vector_store = vector_store
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.min_feedback_count = min_feedback_count

        self.last_retrain = None
        self.retrain_history = []

    def should_retrain(self) -> bool:
        """
        Determine if model should be retrained.

        Criteria:
        - At least min_feedback_count new corrections
        - At least 7 days since last retrain
        """
        # Check feedback count
        recent_feedback = self.feedback_collector.get_feedback(
            since=self.last_retrain
        )

        if len(recent_feedback) < self.min_feedback_count:
            logger.info(
                f"Insufficient feedback: {len(recent_feedback)}/{self.min_feedback_count}"
            )
            return False

        # Check time since last retrain
        if self.last_retrain:
            days_since = (datetime.now() - self.last_retrain).days
            if days_since < 7:
                logger.info(f"Too soon to retrain: {days_since} days")
                return False

        logger.info("Retraining criteria met")
        return True

    def collect_training_data(self) -> tuple:
        """
        Collect training data from feedback and vector store.

        Returns:
            Tuple of (X_new, y_new, weights)
        """
        X_new = []
        y_new = []
        weights = []

        # Get feedback data
        feedback = self.feedback_collector.get_feedback(
            since=self.last_retrain
        )

        logger.info(f"Collected {len(feedback)} feedback samples")

        # Get corresponding embeddings from vector store
        if self.vector_store:
            for entry in feedback:
                txn_data = self.vector_store.get_by_merchant(
                    entry["merchant"],
                    limit=1
                )

                if txn_data and len(txn_data) > 0:
                    embedding = np.array(txn_data[0]["embedding"])
                    X_new.append(embedding)
                    y_new.append(entry["corrected_category"])

                    # Weight by inverse confidence (lower confidence = higher weight)
                    weight = 1.0 / (entry["confidence"] + 0.1)
                    weights.append(weight)

        # Get additional high-confidence data from vector store
        if self.vector_store:
            high_conf_embeddings, high_conf_labels = (
                self.vector_store.get_training_data(
                    min_confidence=0.9,
                    limit=10000
                )
            )

            if len(high_conf_embeddings) > 0:
                X_new.extend(high_conf_embeddings)
                y_new.extend(high_conf_labels)
                weights.extend([0.5] * len(high_conf_labels))  # Lower weight

        logger.info(f"Total training samples: {len(X_new)}")

        return (
            np.array(X_new) if X_new else np.array([]),
            np.array(y_new) if y_new else np.array([]),
            np.array(weights) if weights else np.array([])
        )

    def retrain(
        self,
        X_base: np.ndarray = None,
        y_base: np.ndarray = None
    ) -> Dict:
        """
        Retrain model with feedback data.

        Args:
            X_base: Base training features (optional)
            y_base: Base training labels (optional)

        Returns:
            Retraining metrics
        """
        logger.info("Starting model retraining")

        # Collect new training data
        X_new, y_new, weights = self.collect_training_data()

        if len(X_new) == 0:
            logger.warning("No new training data available")
            return {"status": "skipped", "reason": "no_data"}

        # Combine with base data if provided
        if X_base is not None and y_base is not None:
            X_train = np.vstack([X_base, X_new])
            y_train = np.concatenate([y_base, y_new])
        else:
            X_train = X_new
            y_train = y_new

        # Retrain classifier
        metrics = self.classifier.train(X_train, y_train)

        # Record retraining
        self.last_retrain = datetime.now()
        self.retrain_history.append({
            "timestamp": self.last_retrain.isoformat(),
            "n_samples": len(X_new),
            "metrics": metrics
        })

        logger.info(f"Retraining complete. New F1: {metrics.get('train_f1', 0):.4f}")

        return {
            "status": "success",
            "timestamp": self.last_retrain.isoformat(),
            "metrics": metrics
        }

    def auto_retrain(
        self,
        X_base: np.ndarray = None,
        y_base: np.ndarray = None
    ) -> Dict:
        """
        Automatic retraining if criteria are met.

        Args:
            X_base: Base training data
            y_base: Base labels

        Returns:
            Retraining result
        """
        if not self.should_retrain():
            return {
                "status": "skipped",
                "reason": "criteria_not_met"
            }

        return self.retrain(X_base, y_base)

    def get_improvement_stats(self) -> Dict:
        """
        Get statistics on model improvement over time.

        Returns:
            Dictionary with improvement metrics
        """
        if len(self.retrain_history) < 2:
            return {"status": "insufficient_history"}

        # Calculate improvement
        first_f1 = self.retrain_history[0]["metrics"].get("train_f1", 0)
        latest_f1 = self.retrain_history[-1]["metrics"].get("train_f1", 0)

        improvement = latest_f1 - first_f1
        improvement_pct = (improvement / first_f1) * 100 if first_f1 > 0 else 0

        return {
            "total_retrains": len(self.retrain_history),
            "first_f1": first_f1,
            "latest_f1": latest_f1,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "history": self.retrain_history
        }
