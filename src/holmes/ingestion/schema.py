"""Canonical transaction schema for Holmes AI."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class Transaction(BaseModel):
    """Canonical transaction schema."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    merchant: str = Field(..., description="Merchant name or description")
    amount: float = Field(..., description="Transaction amount")
    date: datetime = Field(..., description="Transaction date")
    channel: Optional[str] = Field(None, description="Transaction channel (online, in-store, etc.)")
    location: Optional[str] = Field(None, description="Geographic location")
    currency: str = Field(default="USD", description="Transaction currency")
    category: Optional[str] = Field(None, description="Assigned category")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification confidence")

    # Metadata
    account_id: Optional[str] = Field(None, description="Account identifier")
    card_last_four: Optional[str] = Field(None, description="Last 4 digits of card")
    mcc_code: Optional[str] = Field(None, description="Merchant Category Code")

    @validator("merchant")
    def clean_merchant(cls, v):
        """Clean and normalize merchant name."""
        if v:
            return v.strip()
        return v

    @validator("amount")
    def validate_amount(cls, v):
        """Ensure amount is valid."""
        if v == 0:
            raise ValueError("Transaction amount cannot be zero")
        return round(v, 2)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN123456",
                "merchant": "STARBUCKS STORE #12345",
                "amount": 5.75,
                "date": "2025-11-12T08:30:00",
                "channel": "in-store",
                "location": "Seattle, WA",
                "currency": "USD"
            }
        }


class TransactionBatch(BaseModel):
    """Batch of transactions for processing."""

    transactions: list[Transaction]
    batch_id: str
    source: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    def __len__(self):
        return len(self.transactions)
