"""Training script for Holmes AI transaction classifier."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime
from sklearn.model_selection import train_test_split

from holmes.ingestion.parsers import parse_file
from holmes.preprocessing.cleaner import MerchantCleaner
from holmes.preprocessing.enrichment import FeatureExtractor
from holmes.encoding.semantic_encoder import SemanticEncoder
from holmes.classification.classifier import TransactionClassifier
from holmes.monitoring.drift_detector import DriftDetector


def load_data(data_path: str) -> tuple:
    """
    Load and parse training data.

    Args:
        data_path: Path to CSV/JSON file with labeled transactions

    Returns:
        Tuple of (transactions, labels)
    """
    logger.info(f"Loading data from {data_path}")

    batch = parse_file(data_path)

    transactions = []
    labels = []

    for txn in batch.transactions:
        if txn.category:  # Only include labeled data
            transactions.append(txn)
            labels.append(txn.category)

    logger.info(f"Loaded {len(transactions)} labeled transactions")

    return transactions, labels


def preprocess_transactions(transactions: list) -> tuple:
    """
    Preprocess transactions: clean, encode, extract features.

    Args:
        transactions: List of Transaction objects

    Returns:
        Tuple of (embeddings, numeric_features, labels)
    """
    logger.info("Preprocessing transactions...")

    cleaner = MerchantCleaner()
    encoder = SemanticEncoder()
    feature_extractor = FeatureExtractor()

    # Clean and encode merchants
    cleaned_merchants = []
    for txn in transactions:
        cleaned = cleaner.clean(txn.merchant)
        cleaned_merchants.append(cleaned["cleaned"])

    logger.info("Generating embeddings...")
    embeddings = encoder.encode(cleaned_merchants, show_progress=True)

    # Extract numeric features
    logger.info("Extracting numeric features...")
    numeric_features = []
    for txn in transactions:
        features = feature_extractor.extract({
            "merchant": txn.merchant,
            "amount": txn.amount,
            "date": txn.date,
            "location": txn.location
        })
        numeric_features.append(features)

    numeric_features = np.array(numeric_features)

    logger.info(
        f"Embeddings shape: {embeddings.shape}, "
        f"Numeric features shape: {numeric_features.shape}"
    )

    return embeddings, numeric_features


def train_model(
    data_path: str,
    output_path: str = "models/classifier.pkl",
    test_size: float = 0.15,
    val_size: float = 0.15
):
    """
    Train the Holmes AI classifier.

    Args:
        data_path: Path to training data
        output_path: Path to save trained model
        test_size: Test set proportion
        val_size: Validation set proportion
    """
    logger.info("=" * 80)
    logger.info("Holmes AI Training Pipeline")
    logger.info("=" * 80)

    # Load data
    transactions, labels = load_data(data_path)

    if len(transactions) < 100:
        logger.error("Insufficient training data. Need at least 100 labeled samples.")
        return

    # Preprocess
    embeddings, numeric_features = preprocess_transactions(transactions)

    # Combine features
    X = np.hstack([embeddings, numeric_features])
    y = np.array(labels)

    logger.info(f"Combined feature shape: {X.shape}")
    logger.info(f"Label distribution:\n{pd.Series(y).value_counts()}")

    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Initialize and train classifier
    logger.info("Training LightGBM classifier...")

    classifier = TransactionClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31
    )

    # Feature names
    feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
    feature_names += [
        "merchant_frequency", "amount_band", "amount_log",
        "hour", "day_of_week", "is_weekend", "time_of_day", "has_location"
    ]

    metrics = classifier.train(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_names
    )

    logger.info(f"Training metrics: {metrics}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = classifier.evaluate(X_test, y_test)

    logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")

    # Feature importance
    logger.info("\nTop 20 Feature Importance:")
    for feature, importance in classifier.get_feature_importance(top_k=20):
        logger.info(f"  {feature}: {importance:.4f}")

    # Set up drift detector baseline
    logger.info("Setting up drift detection baseline...")
    drift_detector = DriftDetector()

    y_test_pred_encoded = classifier.model.predict(X_test)
    y_test_encoded = classifier.label_encoder.transform(y_test)

    drift_detector.set_baseline(y_test_encoded, y_test_pred_encoded)

    # Save model
    logger.info(f"Saving model to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save(output_path)

    # Save drift detector
    import joblib
    drift_path = Path(output_path).parent / "drift_detector.pkl"
    joblib.dump(drift_detector, drift_path)

    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_categories": len(np.unique(y)),
        "categories": list(np.unique(y)),
        "test_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }

    metadata_path = Path(output_path).parent / "model_metadata.json"
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info("=" * 80)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Holmes AI classifier")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (CSV or JSON)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/classifier.pkl",
        help="Path to save trained model"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set proportion (default: 0.15)"
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation set proportion (default: 0.15)"
    )

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        output_path=args.output,
        test_size=args.test_size,
        val_size=args.val_size
    )


if __name__ == "__main__":
    main()
