"""Text cleaning and normalization for merchant descriptions."""

import re
from typing import Dict, List
from loguru import logger


class MerchantCleaner:
    """Clean and normalize merchant descriptions."""

    # Common patterns to remove
    NOISE_PATTERNS = [
        r"\b\d{3,}\b",  # Long numbers (store IDs, etc.)
        r"#\d+",  # Hash numbers
        r"\*+",  # Asterisks
        r"\s+\d{2}/\d{2}",  # Dates
        r"\b[A-Z]{2,3}\s*\d+",  # State/Country codes with numbers
        r"SQ\s*\*",  # Square payment prefix
        r"TST\s*\*",  # Toast payment prefix
        r"SP\s*\*",  # Stripe payment prefix
        r"PAYPAL\s*\*",  # PayPal prefix
    ]

    # Location patterns
    LOCATION_PATTERNS = [
        r",\s*[A-Z]{2}\b",  # State codes (e.g., ", CA")
        r"\b[A-Z]{2}\s+\d{5}\b",  # State + ZIP
        r"\s+USA\b",  # USA suffix
    ]

    # Common merchant prefixes/suffixes to remove
    STRIP_TERMS = [
        "inc", "llc", "ltd", "corp", "corporation",
        "co", "company", "store", "shop", "market"
    ]

    def __init__(self):
        self.compiled_noise = [re.compile(pattern, re.IGNORECASE) for pattern in self.NOISE_PATTERNS]
        self.compiled_location = [re.compile(pattern) for pattern in self.LOCATION_PATTERNS]

    def clean(self, merchant: str) -> Dict[str, str]:
        """
        Clean merchant description and extract components.

        Returns:
            Dictionary with 'cleaned', 'location', and 'tokens'
        """
        original = merchant
        text = merchant.upper()

        # Extract location
        location = None
        for pattern in self.compiled_location:
            match = pattern.search(text)
            if match:
                location = match.group(0).strip()
                text = pattern.sub("", text)
                break

        # Remove noise patterns
        for pattern in self.compiled_noise:
            text = pattern.sub(" ", text)

        # Remove special characters except spaces and hyphens
        text = re.sub(r"[^A-Z0-9\s\-]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove common suffixes
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in self.STRIP_TERMS]

        cleaned = " ".join(filtered_words) if filtered_words else text

        logger.debug(f"Cleaned: '{original}' -> '{cleaned}'")

        return {
            "cleaned": cleaned,
            "location": location,
            "tokens": cleaned.split()
        }


class TextNormalizer:
    """Additional text normalization utilities."""

    @staticmethod
    def remove_duplicate_tokens(tokens: List[str]) -> List[str]:
        """Remove duplicate consecutive tokens."""
        if not tokens:
            return tokens

        result = [tokens[0]]
        for token in tokens[1:]:
            if token != result[-1]:
                result.append(token)

        return result

    @staticmethod
    def expand_abbreviations(text: str) -> str:
        """Expand common abbreviations."""
        abbreviations = {
            r"\bRESTAURANT\b": "RESTAURANT",
            r"\bRSTRNT\b": "RESTAURANT",
            r"\bMKT\b": "MARKET",
            r"\bSTR\b": "STORE",
            r"\bCTR\b": "CENTER",
            r"\bDEPT\b": "DEPARTMENT",
            r"\bSVC\b": "SERVICE",
            r"\bINTL\b": "INTERNATIONAL",
        }

        for abbr, full in abbreviations.items():
            text = re.sub(abbr, full, text, flags=re.IGNORECASE)

        return text

    @staticmethod
    def handle_camel_case(text: str) -> str:
        """Split camelCase into separate words."""
        # Insert space before capital letters
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        return text
