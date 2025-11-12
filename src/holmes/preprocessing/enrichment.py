"""Token enrichment with contextual features."""

from typing import Dict, List
from datetime import datetime
from collections import Counter
from loguru import logger
import numpy as np


class TokenEnricher:
    """Enrich transactions with contextual features."""

    def __init__(self):
        self.merchant_frequency = Counter()
        self.category_history = {}

    def compute_features(
        self,
        merchant: str,
        amount: float,
        date: datetime,
        location: str = None
    ) -> Dict[str, float]:
        """
        Compute enrichment features for a transaction.

        Features:
        - merchant_frequency: How often this merchant appears
        - amount_band: Categorical amount range
        - time_of_day: Hour bucket (morning/afternoon/evening/night)
        - day_of_week: Day of week (0-6)
        - is_weekend: Weekend flag
        - amount_log: Log-transformed amount
        - has_location: Location flag
        """
        features = {}

        # Merchant frequency (if tracking enabled)
        merchant_clean = merchant.upper().strip()
        self.merchant_frequency[merchant_clean] += 1
        features["merchant_frequency"] = self.merchant_frequency[merchant_clean]

        # Amount bands
        features["amount_band"] = self._get_amount_band(amount)
        features["amount_log"] = np.log1p(abs(amount))

        # Temporal features
        features["hour"] = date.hour
        features["day_of_week"] = date.weekday()
        features["is_weekend"] = 1 if date.weekday() >= 5 else 0
        features["time_of_day"] = self._get_time_of_day(date.hour)

        # Location flag
        features["has_location"] = 1 if location else 0

        return features

    def _get_amount_band(self, amount: float) -> int:
        """
        Categorize amount into bands.

        0: < $10
        1: $10-50
        2: $50-100
        3: $100-500
        4: $500-1000
        5: $1000+
        """
        abs_amount = abs(amount)

        if abs_amount < 10:
            return 0
        elif abs_amount < 50:
            return 1
        elif abs_amount < 100:
            return 2
        elif abs_amount < 500:
            return 3
        elif abs_amount < 1000:
            return 4
        else:
            return 5

    def _get_time_of_day(self, hour: int) -> int:
        """
        Categorize hour into time of day.

        0: Night (0-5)
        1: Morning (6-11)
        2: Afternoon (12-17)
        3: Evening (18-23)
        """
        if 0 <= hour < 6:
            return 0
        elif 6 <= hour < 12:
            return 1
        elif 12 <= hour < 18:
            return 2
        else:
            return 3

    def enrich_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Enrich a batch of transactions.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of transactions with added 'features' field
        """
        enriched = []

        for txn in transactions:
            features = self.compute_features(
                merchant=txn.get("merchant", ""),
                amount=txn.get("amount", 0.0),
                date=txn.get("date", datetime.now()),
                location=txn.get("location")
            )

            enriched_txn = txn.copy()
            enriched_txn["features"] = features
            enriched.append(enriched_txn)

        logger.info(f"Enriched {len(enriched)} transactions")
        return enriched

    def update_frequency(self, merchant: str):
        """Update merchant frequency counter."""
        merchant_clean = merchant.upper().strip()
        self.merchant_frequency[merchant_clean] += 1

    def get_top_merchants(self, n: int = 100) -> List[tuple]:
        """Get top N most frequent merchants."""
        return self.merchant_frequency.most_common(n)


class FeatureExtractor:
    """Extract feature vectors for ML models."""

    def __init__(self):
        self.enricher = TokenEnricher()

    def extract(self, transaction: Dict) -> np.ndarray:
        """
        Extract feature vector from transaction.

        Returns:
            NumPy array of numeric features
        """
        features = self.enricher.compute_features(
            merchant=transaction.get("merchant", ""),
            amount=transaction.get("amount", 0.0),
            date=transaction.get("date", datetime.now()),
            location=transaction.get("location")
        )

        # Convert to ordered feature vector
        feature_vector = [
            features["merchant_frequency"],
            features["amount_band"],
            features["amount_log"],
            features["hour"],
            features["day_of_week"],
            features["is_weekend"],
            features["time_of_day"],
            features["has_location"],
        ]

        return np.array(feature_vector, dtype=np.float32)
