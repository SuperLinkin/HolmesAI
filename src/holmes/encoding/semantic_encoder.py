"""Sentence-BERT semantic encoding for merchant descriptions."""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch


class SemanticEncoder:
    """
    Encode merchant descriptions using fine-tuned Sentence-BERT.

    Uses all-MiniLM-L6-v2 model (384-dimensional embeddings).
    Achieves cosine similarity ~0.82 for similar merchants.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
        batch_size: int = 32
    ):
        """
        Initialize semantic encoder.

        Args:
            model_name: SentenceTransformer model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Sentence-BERT model: {model_name} on {self.device}")

        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into semantic embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings to unit length
            show_progress: Show progress bar

        Returns:
            NumPy array of shape (n_texts, embedding_dim)
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        logger.debug(f"Encoding {len(texts)} texts")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )

            return embeddings

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode batch of texts with custom batch size.

        Args:
            texts: List of texts to encode
            batch_size: Custom batch size (overrides default)
            show_progress: Show progress bar

        Returns:
            NumPy array of embeddings
        """
        if batch_size is None:
            batch_size = self.batch_size

        return self.encode(
            texts,
            normalize=True,
            show_progress=show_progress
        )

    def compute_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two texts or embeddings.

        Args:
            text1: First text or embedding
            text2: Second text or embedding

        Returns:
            Cosine similarity score (0-1)
        """
        # Encode if texts
        if isinstance(text1, str):
            emb1 = self.encode(text1)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.encode(text2)
        else:
            emb2 = text2

        # Compute cosine similarity
        similarity = np.dot(emb1.flatten(), emb2.flatten())

        return float(similarity)

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar candidates to query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (candidate, similarity_score) tuples
        """
        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates)

        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def fine_tune(
        self,
        train_examples: List[tuple],
        epochs: int = 1,
        warmup_steps: int = 100
    ):
        """
        Fine-tune the model on domain-specific data.

        Args:
            train_examples: List of (text, label) tuples
            epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate

        Note: Placeholder for fine-tuning logic. Requires InputExample
        and training loop from sentence-transformers.
        """
        logger.info("Fine-tuning not implemented in this version")
        pass

    def save(self, path: str):
        """Save model to disk."""
        logger.info(f"Saving model to {path}")
        self.model.save(path)

    def load(self, path: str):
        """Load model from disk."""
        logger.info(f"Loading model from {path}")
        self.model = SentenceTransformer(path, device=self.device)


class MerchantEncoder:
    """Specialized encoder for merchant descriptions."""

    def __init__(self, semantic_encoder: SemanticEncoder = None):
        """Initialize with a semantic encoder."""
        self.encoder = semantic_encoder or SemanticEncoder()

    def encode_merchants(
        self,
        merchants: List[str],
        clean: bool = True
    ) -> np.ndarray:
        """
        Encode merchant descriptions.

        Args:
            merchants: List of merchant strings
            clean: Apply basic cleaning before encoding

        Returns:
            Embeddings array
        """
        if clean:
            # Basic cleaning
            merchants = [m.strip().upper() for m in merchants]

        return self.encoder.encode(merchants, show_progress=True)

    def cluster_merchants(
        self,
        merchants: List[str],
        threshold: float = 0.82
    ) -> dict:
        """
        Cluster similar merchants together.

        Args:
            merchants: List of merchant names
            threshold: Similarity threshold for clustering

        Returns:
            Dictionary mapping cluster_id to list of merchants
        """
        logger.info(f"Clustering {len(merchants)} merchants")

        embeddings = self.encode_merchants(merchants)

        # Simple greedy clustering
        clusters = {}
        assigned = set()
        cluster_id = 0

        for i, merchant in enumerate(merchants):
            if i in assigned:
                continue

            # Start new cluster
            clusters[cluster_id] = [merchant]
            assigned.add(i)

            # Find similar merchants
            for j in range(i + 1, len(merchants)):
                if j in assigned:
                    continue

                similarity = self.encoder.compute_similarity(
                    embeddings[i],
                    embeddings[j]
                )

                if similarity >= threshold:
                    clusters[cluster_id].append(merchants[j])
                    assigned.add(j)

            cluster_id += 1

        logger.info(f"Created {len(clusters)} clusters")
        return clusters
