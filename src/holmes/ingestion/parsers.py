"""Data parsers for CSV and JSON transaction files."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union
from loguru import logger
import pandas as pd

from .schema import Transaction, TransactionBatch


class BaseParser:
    """Base parser class."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def parse(self) -> TransactionBatch:
        """Parse file and return transaction batch."""
        raise NotImplementedError


class CSVParser(BaseParser):
    """Parse CSV transaction files."""

    # Common field mappings from ERP/bank exports
    FIELD_MAPPINGS = {
        "transaction_id": ["id", "transaction_id", "txn_id", "reference"],
        "merchant": ["merchant", "description", "merchant_name", "payee"],
        "amount": ["amount", "value", "transaction_amount", "debit", "credit"],
        "date": ["date", "transaction_date", "posted_date", "value_date"],
        "channel": ["channel", "type", "transaction_type"],
        "location": ["location", "city", "merchant_location"],
        "currency": ["currency", "ccy"],
        "account_id": ["account_id", "account", "account_number"],
        "mcc_code": ["mcc", "mcc_code", "merchant_category_code"],
    }

    def __init__(self, file_path: Union[str, Path], delimiter: str = ","):
        super().__init__(file_path)
        self.delimiter = delimiter

    def _map_field(self, headers: list[str], canonical_field: str) -> str:
        """Map CSV header to canonical field name."""
        possible_names = self.FIELD_MAPPINGS.get(canonical_field, [])
        headers_lower = [h.lower() for h in headers]

        for possible_name in possible_names:
            if possible_name.lower() in headers_lower:
                idx = headers_lower.index(possible_name.lower())
                return headers[idx]

        return None

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from various formats."""
        date_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date: {date_str}")

    def parse(self) -> TransactionBatch:
        """Parse CSV file."""
        logger.info(f"Parsing CSV file: {self.file_path}")

        df = pd.read_csv(self.file_path, delimiter=self.delimiter)
        headers = df.columns.tolist()

        # Build field mapping
        field_map = {}
        for canonical_field in self.FIELD_MAPPINGS.keys():
            mapped_field = self._map_field(headers, canonical_field)
            if mapped_field:
                field_map[canonical_field] = mapped_field

        logger.info(f"Field mapping: {field_map}")

        transactions = []
        for idx, row in df.iterrows():
            try:
                # Extract required fields
                txn_id = str(row.get(field_map.get("transaction_id", ""), f"TXN_{idx}"))
                merchant = str(row.get(field_map.get("merchant", ""), "UNKNOWN"))
                amount = float(row.get(field_map.get("amount", ""), 0.0))

                date_str = str(row.get(field_map.get("date", ""), datetime.now().isoformat()))
                try:
                    txn_date = self._parse_date(date_str)
                except:
                    txn_date = datetime.now()

                # Extract optional fields
                channel = row.get(field_map.get("channel")) if field_map.get("channel") else None
                location = row.get(field_map.get("location")) if field_map.get("location") else None
                currency = row.get(field_map.get("currency"), "USD")
                account_id = row.get(field_map.get("account_id")) if field_map.get("account_id") else None
                mcc_code = row.get(field_map.get("mcc_code")) if field_map.get("mcc_code") else None

                transaction = Transaction(
                    transaction_id=txn_id,
                    merchant=merchant,
                    amount=amount,
                    date=txn_date,
                    channel=channel,
                    location=location,
                    currency=currency,
                    account_id=account_id,
                    mcc_code=mcc_code,
                )

                transactions.append(transaction)

            except Exception as e:
                logger.warning(f"Failed to parse row {idx}: {e}")
                continue

        logger.info(f"Successfully parsed {len(transactions)} transactions")

        return TransactionBatch(
            transactions=transactions,
            batch_id=f"BATCH_{self.file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source=str(self.file_path),
        )


class JSONParser(BaseParser):
    """Parse JSON transaction files."""

    def parse(self) -> TransactionBatch:
        """Parse JSON file."""
        logger.info(f"Parsing JSON file: {self.file_path}")

        with open(self.file_path, "r") as f:
            data = json.load(f)

        # Handle both single object and array formats
        if isinstance(data, dict):
            if "transactions" in data:
                raw_transactions = data["transactions"]
            else:
                raw_transactions = [data]
        else:
            raw_transactions = data

        transactions = []
        for idx, txn_data in enumerate(raw_transactions):
            try:
                # Parse date if string
                if isinstance(txn_data.get("date"), str):
                    txn_data["date"] = datetime.fromisoformat(
                        txn_data["date"].replace("Z", "+00:00")
                    )

                transaction = Transaction(**txn_data)
                transactions.append(transaction)

            except Exception as e:
                logger.warning(f"Failed to parse transaction {idx}: {e}")
                continue

        logger.info(f"Successfully parsed {len(transactions)} transactions")

        return TransactionBatch(
            transactions=transactions,
            batch_id=f"BATCH_{self.file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source=str(self.file_path),
        )


def parse_file(file_path: Union[str, Path]) -> TransactionBatch:
    """Auto-detect file type and parse."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        parser = CSVParser(file_path)
    elif file_path.suffix.lower() == ".json":
        parser = JSONParser(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    return parser.parse()
