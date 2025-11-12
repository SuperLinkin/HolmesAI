"""Example API client for Holmes AI."""

import requests
import json
from datetime import datetime


class HolmesClient:
    """Simple client for Holmes AI API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client."""
        self.base_url = base_url.rstrip("/")

    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def categorize(
        self,
        transaction_id: str,
        merchant: str,
        amount: float,
        date: datetime,
        channel: str = None,
        location: str = None
    ):
        """Categorize a single transaction."""
        payload = {
            "transaction_id": transaction_id,
            "merchant": merchant,
            "amount": amount,
            "date": date.isoformat(),
            "channel": channel,
            "location": location
        }

        response = requests.post(
            f"{self.base_url}/categorize",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def categorize_batch(self, transactions: list):
        """Categorize multiple transactions."""
        payload = {"transactions": transactions}

        response = requests.post(
            f"{self.base_url}/categorize/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def submit_feedback(
        self,
        transaction_id: str,
        merchant: str,
        predicted_category: str,
        corrected_category: str,
        confidence: float,
        user_id: str = None,
        notes: str = None
    ):
        """Submit feedback on a prediction."""
        payload = {
            "transaction_id": transaction_id,
            "merchant": merchant,
            "predicted_category": predicted_category,
            "corrected_category": corrected_category,
            "confidence": confidence,
            "user_id": user_id,
            "notes": notes
        }

        response = requests.post(
            f"{self.base_url}/feedback",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_model_info(self):
        """Get model information."""
        response = requests.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

    def get_taxonomy(self):
        """Get category taxonomy."""
        response = requests.get(f"{self.base_url}/taxonomy")
        response.raise_for_status()
        return response.json()


def example_usage():
    """Example usage of Holmes AI client."""
    print("Holmes AI API Client Examples")
    print("=" * 80)

    client = HolmesClient()

    # 1. Health check
    print("\n1. Health Check")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Note: Make sure the API server is running!")
        return

    # 2. Categorize single transaction
    print("\n2. Categorize Single Transaction")
    try:
        result = client.categorize(
            transaction_id="TXN001",
            merchant="STARBUCKS STORE #12345",
            amount=5.75,
            date=datetime(2025, 11, 12, 8, 30),
            channel="in-store",
            location="Seattle, WA"
        )
        print(f"   Merchant: {result['merchant']}")
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Batch categorization
    print("\n3. Batch Categorization")
    try:
        transactions = [
            {
                "transaction_id": "TXN002",
                "merchant": "CHEVRON GAS STATION",
                "amount": 45.50,
                "date": datetime(2025, 11, 12, 9, 15).isoformat(),
                "channel": "in-store",
                "location": "Seattle, WA"
            },
            {
                "transaction_id": "TXN003",
                "merchant": "AMAZON WEB SERVICES",
                "amount": 125.00,
                "date": datetime(2025, 11, 12, 0, 0).isoformat(),
                "channel": "online"
            }
        ]

        results = client.categorize_batch(transactions)
        print(f"   Processed {len(results)} transactions")
        for result in results:
            print(f"   - {result['merchant']}: {result['category']} ({result['confidence']:.2f})")
    except Exception as e:
        print(f"   Error: {e}")

    # 4. Get taxonomy
    print("\n4. Get Taxonomy")
    try:
        taxonomy = client.get_taxonomy()
        print(f"   Total categories: {taxonomy['metadata']['total_categories']}")
        print("   Categories:")
        for cat in taxonomy['categories'][:5]:
            print(f"   - {cat['name']}")
    except Exception as e:
        print(f"   Error: {e}")

    # 5. Submit feedback
    print("\n5. Submit Feedback")
    try:
        feedback_result = client.submit_feedback(
            transaction_id="TXN001",
            merchant="STARBUCKS",
            predicted_category="Dining & Food",
            corrected_category="Office Supplies",
            confidence=0.65,
            notes="This was office supplies purchase"
        )
        print(f"   Status: {feedback_result['status']}")
        print(f"   Message: {feedback_result['message']}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 80)
    print("Examples complete!")


if __name__ == "__main__":
    example_usage()
