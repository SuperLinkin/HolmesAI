"""Supabase pgvector integration for embeddings storage and similarity search."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from supabase import create_client, Client
from loguru import logger
import os
from datetime import datetime


class VectorStore:
    """
    Supabase pgvector store for transaction embeddings.

    Enables:
    - Embedding storage with metadata
    - Similarity search for merchant matching
    - Re-training data collection
    """

    def __init__(
        self,
        supabase_url: str = None,
        supabase_key: str = None,
        table_name: str = "transaction_embeddings"
    ):
        """
        Initialize vector store.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            table_name: Name of embeddings table
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.table_name = table_name

        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not provided. Vector store disabled.")
            self.client = None
            return

        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"Connected to Supabase. Table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.client = None

    def create_table(self):
        """
        Create embeddings table with pgvector extension.

        SQL Schema:
        CREATE TABLE transaction_embeddings (
            id BIGSERIAL PRIMARY KEY,
            transaction_id TEXT UNIQUE NOT NULL,
            merchant TEXT NOT NULL,
            embedding VECTOR(384),
            category TEXT,
            confidence FLOAT,
            amount FLOAT,
            date TIMESTAMP,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX ON transaction_embeddings USING ivfflat (embedding vector_cosine_ops);
        """
        logger.info("Create table SQL provided in docstring. Execute manually in Supabase.")
        pass

    def store_embedding(
        self,
        transaction_id: str,
        merchant: str,
        embedding: np.ndarray,
        category: str = None,
        confidence: float = None,
        amount: float = None,
        date: datetime = None,
        metadata: Dict = None
    ) -> bool:
        """
        Store transaction embedding.

        Args:
            transaction_id: Unique transaction ID
            merchant: Merchant description
            embedding: Embedding vector
            category: Assigned category
            confidence: Classification confidence
            amount: Transaction amount
            date: Transaction date
            metadata: Additional metadata

        Returns:
            Success boolean
        """
        if not self.client:
            logger.warning("Vector store not initialized")
            return False

        try:
            # Convert embedding to list
            embedding_list = embedding.flatten().tolist()

            data = {
                "transaction_id": transaction_id,
                "merchant": merchant,
                "embedding": embedding_list,
                "category": category,
                "confidence": confidence,
                "amount": amount,
                "date": date.isoformat() if date else None,
                "metadata": metadata or {}
            }

            result = self.client.table(self.table_name).insert(data).execute()

            logger.debug(f"Stored embedding for transaction {transaction_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False

    def store_batch(
        self,
        transactions: List[Dict],
        embeddings: np.ndarray
    ) -> int:
        """
        Store batch of embeddings.

        Args:
            transactions: List of transaction dictionaries
            embeddings: Array of embeddings (n_transactions, embedding_dim)

        Returns:
            Number of successfully stored embeddings
        """
        if not self.client:
            logger.warning("Vector store not initialized")
            return 0

        if len(transactions) != len(embeddings):
            raise ValueError("Number of transactions must match number of embeddings")

        success_count = 0

        for txn, emb in zip(transactions, embeddings):
            success = self.store_embedding(
                transaction_id=txn.get("transaction_id"),
                merchant=txn.get("merchant"),
                embedding=emb,
                category=txn.get("category"),
                confidence=txn.get("confidence"),
                amount=txn.get("amount"),
                date=txn.get("date"),
                metadata=txn.get("metadata")
            )

            if success:
                success_count += 1

        logger.info(f"Stored {success_count}/{len(transactions)} embeddings")
        return success_count

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        category_filter: str = None,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find similar transactions using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            category_filter: Filter by category
            threshold: Minimum similarity threshold

        Returns:
            List of similar transactions with metadata
        """
        if not self.client:
            logger.warning("Vector store not initialized")
            return []

        try:
            # Note: Actual similarity search requires RPC call to Postgres function
            # This is a simplified version. You'll need to create a Postgres function:
            # CREATE OR REPLACE FUNCTION match_transactions(query_embedding vector, match_threshold float, match_count int)
            # RETURNS TABLE (id bigint, transaction_id text, merchant text, category text, similarity float)

            logger.warning("Similarity search requires custom Postgres function. See documentation.")
            return []

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_by_merchant(
        self,
        merchant: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Retrieve embeddings for a specific merchant.

        Args:
            merchant: Merchant name
            limit: Maximum results

        Returns:
            List of transaction records
        """
        if not self.client:
            return []

        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .ilike("merchant", f"%{merchant}%")
                .limit(limit)
                .execute()
            )

            return result.data

        except Exception as e:
            logger.error(f"Failed to query by merchant: {e}")
            return []

    def get_low_confidence(
        self,
        threshold: float = 0.7,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Retrieve transactions with low classification confidence.

        Args:
            threshold: Confidence threshold
            limit: Maximum results

        Returns:
            List of low-confidence transactions
        """
        if not self.client:
            return []

        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .lt("confidence", threshold)
                .limit(limit)
                .execute()
            )

            logger.info(f"Found {len(result.data)} low-confidence transactions")
            return result.data

        except Exception as e:
            logger.error(f"Failed to query low-confidence: {e}")
            return []

    def update_category(
        self,
        transaction_id: str,
        new_category: str,
        confidence: float = 1.0
    ) -> bool:
        """
        Update category for a transaction (feedback loop).

        Args:
            transaction_id: Transaction ID
            new_category: Corrected category
            confidence: New confidence score

        Returns:
            Success boolean
        """
        if not self.client:
            return False

        try:
            result = (
                self.client.table(self.table_name)
                .update({
                    "category": new_category,
                    "confidence": confidence
                })
                .eq("transaction_id", transaction_id)
                .execute()
            )

            logger.info(f"Updated category for {transaction_id} to {new_category}")
            return True

        except Exception as e:
            logger.error(f"Failed to update category: {e}")
            return False

    def get_training_data(
        self,
        min_confidence: float = 0.7,
        limit: int = 100000
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve high-quality data for re-training.

        Args:
            min_confidence: Minimum confidence threshold
            limit: Maximum records

        Returns:
            Tuple of (embeddings, categories)
        """
        if not self.client:
            logger.warning("Vector store not initialized")
            return np.array([]), []

        try:
            result = (
                self.client.table(self.table_name)
                .select("embedding, category")
                .gte("confidence", min_confidence)
                .not_.is_("category", "null")
                .limit(limit)
                .execute()
            )

            embeddings = []
            categories = []

            for row in result.data:
                embeddings.append(row["embedding"])
                categories.append(row["category"])

            logger.info(f"Retrieved {len(embeddings)} training samples")

            return np.array(embeddings), categories

        except Exception as e:
            logger.error(f"Failed to retrieve training data: {e}")
            return np.array([]), []

    def delete_old_records(self, days: int = 365) -> int:
        """
        Delete records older than specified days.

        Args:
            days: Number of days to retain

        Returns:
            Number of deleted records
        """
        if not self.client:
            return 0

        try:
            # Calculate cutoff date
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=days)

            result = (
                self.client.table(self.table_name)
                .delete()
                .lt("created_at", cutoff.isoformat())
                .execute()
            )

            logger.info(f"Deleted records older than {days} days")
            return len(result.data) if result.data else 0

        except Exception as e:
            logger.error(f"Failed to delete old records: {e}")
            return 0
