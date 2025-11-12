"""Basic usage examples for Holmes AI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime
import numpy as np

from holmes.preprocessing.cleaner import MerchantCleaner
from holmes.preprocessing.enrichment import FeatureExtractor
from holmes.encoding.semantic_encoder import SemanticEncoder
from holmes.classification.classifier import TransactionClassifier


def example_1_clean_merchant():
    """Example 1: Clean and normalize merchant descriptions."""
    print("=" * 80)
    print("Example 1: Cleaning Merchant Descriptions")
    print("=" * 80)

    cleaner = MerchantCleaner()

    merchants = [
        "STARBUCKS STORE #12345 SEATTLE WA",
        "SQ *COFFEE SHOP 415-555-1234",
        "PAYPAL *AMAZON.COM",
        "UBER   *RIDE 800-555-0000"
    ]

    for merchant in merchants:
        result = cleaner.clean(merchant)
        print(f"\nOriginal: {merchant}")
        print(f"Cleaned:  {result['cleaned']}")
        print(f"Location: {result['location']}")
        print(f"Tokens:   {result['tokens']}")


def example_2_extract_features():
    """Example 2: Extract contextual features from transactions."""
    print("\n" + "=" * 80)
    print("Example 2: Extracting Features")
    print("=" * 80)

    extractor = FeatureExtractor()

    transaction = {
        "merchant": "STARBUCKS STORE #12345",
        "amount": 5.75,
        "date": datetime(2025, 11, 12, 8, 30),
        "location": "Seattle, WA"
    }

    features = extractor.compute_features(**transaction)

    print("\nTransaction:")
    print(f"  Merchant: {transaction['merchant']}")
    print(f"  Amount: ${transaction['amount']}")
    print(f"  Date: {transaction['date']}")

    print("\nExtracted Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value}")


def example_3_semantic_encoding():
    """Example 3: Generate semantic embeddings."""
    print("\n" + "=" * 80)
    print("Example 3: Semantic Encoding")
    print("=" * 80)

    encoder = SemanticEncoder()

    merchants = [
        "STARBUCKS",
        "COSTA COFFEE",
        "CAFE COFFEE DAY",
        "CHEVRON GAS STATION",
        "SHELL FUEL"
    ]

    print("\nGenerating embeddings...")
    embeddings = encoder.encode(merchants)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Compute similarity
    print("\nSimilarity Scores (STARBUCKS vs others):")
    for i, merchant in enumerate(merchants[1:], 1):
        similarity = encoder.compute_similarity(embeddings[0], embeddings[i])
        print(f"  {merchant}: {similarity:.4f}")


def example_4_full_pipeline():
    """Example 4: Complete categorization pipeline."""
    print("\n" + "=" * 80)
    print("Example 4: Full Pipeline (requires trained model)")
    print("=" * 80)

    # Initialize components
    cleaner = MerchantCleaner()
    encoder = SemanticEncoder()
    extractor = FeatureExtractor()

    # Sample transaction
    transaction = {
        "transaction_id": "TXN123",
        "merchant": "STARBUCKS STORE #12345 SEATTLE WA",
        "amount": 5.75,
        "date": datetime(2025, 11, 12, 8, 30),
        "location": "Seattle, WA"
    }

    print("\nInput Transaction:")
    print(f"  ID: {transaction['transaction_id']}")
    print(f"  Merchant: {transaction['merchant']}")
    print(f"  Amount: ${transaction['amount']}")

    # Step 1: Clean merchant
    cleaned = cleaner.clean(transaction["merchant"])
    print(f"\nCleaned Merchant: {cleaned['cleaned']}")

    # Step 2: Generate embedding
    embedding = encoder.encode(cleaned["cleaned"])
    print(f"Embedding shape: {embedding.shape}")

    # Step 3: Extract features
    features = extractor.extract(transaction)
    feature_array = np.array([
        features["merchant_frequency"],
        features["amount_band"],
        features["amount_log"],
        features["hour"],
        features["day_of_week"],
        features["is_weekend"],
        features["time_of_day"],
        features["has_location"],
    ])
    print(f"Feature array shape: {feature_array.shape}")

    # Step 4: Combine features
    combined = np.concatenate([embedding.flatten(), feature_array])
    print(f"Combined feature shape: {combined.shape}")

    print("\n✓ Pipeline complete. Ready for classification.")
    print("  (Load trained model with classifier.load('models/classifier.pkl'))")


def example_5_similarity_search():
    """Example 5: Find similar merchants."""
    print("\n" + "=" * 80)
    print("Example 5: Similarity Search")
    print("=" * 80)

    encoder = SemanticEncoder()

    query = "COFFEE SHOP"
    candidates = [
        "STARBUCKS CAFE",
        "DUNKIN DONUTS",
        "PEET'S COFFEE",
        "CHEVRON GAS",
        "MICROSOFT STORE",
        "COSTA COFFEE",
        "LOCAL COFFEE BAR"
    ]

    print(f"\nQuery: {query}")
    print("\nTop 5 Similar Merchants:")

    results = encoder.find_similar(query, candidates, top_k=5)

    for i, (merchant, score) in enumerate(results, 1):
        print(f"  {i}. {merchant}: {score:.4f}")


if __name__ == "__main__":
    example_1_clean_merchant()
    example_2_extract_features()
    example_3_semantic_encoding()
    example_4_full_pipeline()
    example_5_similarity_search()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
