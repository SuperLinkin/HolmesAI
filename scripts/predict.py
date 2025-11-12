"""Prediction script for Holmes AI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd
from loguru import logger

from holmes.ingestion.parsers import parse_file
from holmes.preprocessing.cleaner import MerchantCleaner
from holmes.preprocessing.enrichment import FeatureExtractor
from holmes.encoding.semantic_encoder import SemanticEncoder
from holmes.classification.classifier import TransactionClassifier


def predict(
    input_path: str,
    model_path: str,
    output_path: str = None,
    confidence_threshold: float = 0.7
):
    """
    Predict categories for transactions.

    Args:
        input_path: Path to input data (CSV/JSON)
        model_path: Path to trained model
        output_path: Path to save predictions (optional)
        confidence_threshold: Threshold for low-confidence warnings
    """
    logger.info("Loading model...")
    classifier = TransactionClassifier()
    classifier.load(model_path)

    logger.info("Loading data...")
    batch = parse_file(input_path)

    logger.info("Preprocessing...")
    cleaner = MerchantCleaner()
    encoder = SemanticEncoder()
    feature_extractor = FeatureExtractor()

    # Preprocess all transactions
    cleaned_merchants = []
    for txn in batch.transactions:
        cleaned = cleaner.clean(txn.merchant)
        cleaned_merchants.append(cleaned["cleaned"])

    embeddings = encoder.encode(cleaned_merchants, show_progress=True)

    numeric_features = []
    for txn in batch.transactions:
        features = feature_extractor.extract({
            "merchant": txn.merchant,
            "amount": txn.amount,
            "date": txn.date,
            "location": txn.location
        })
        numeric_features.append(features)

    numeric_features = np.array(numeric_features)

    # Combine features
    X = np.hstack([embeddings, numeric_features])

    logger.info("Predicting...")
    predictions = classifier.predict(X, return_confidence=True)

    # Build results dataframe
    results = []
    low_confidence_count = 0

    for txn, pred in zip(batch.transactions, predictions):
        result = {
            "transaction_id": txn.transaction_id,
            "merchant": txn.merchant,
            "amount": txn.amount,
            "date": txn.date.isoformat(),
            "predicted_category": pred["category"],
            "confidence": pred["confidence"]
        }

        if pred["confidence"] < confidence_threshold:
            low_confidence_count += 1
            result["needs_review"] = True
        else:
            result["needs_review"] = False

        results.append(result)

    df = pd.DataFrame(results)

    # Print summary
    logger.info("=" * 80)
    logger.info(f"Predictions complete: {len(results)} transactions")
    logger.info(f"Low confidence (<{confidence_threshold}): {low_confidence_count}")
    logger.info("\nCategory distribution:")
    logger.info(df["predicted_category"].value_counts())
    logger.info("=" * 80)

    # Save results
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

    return df


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(description="Predict transaction categories")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data (CSV or JSON)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/classifier.pkl",
        help="Path to trained model"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (CSV)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for warnings"
    )

    args = parser.parse_args()

    predict(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        confidence_threshold=args.threshold
    )


if __name__ == "__main__":
    main()
