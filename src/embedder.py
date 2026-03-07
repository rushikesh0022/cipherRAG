# src/embedder.py
# ═══════════════════════════════════════════════════════════
#  Generate embeddings from text chunks
# ═══════════════════════════════════════════════════════════

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import EmbeddingConfig


class Embedder:
    """
    Converts text chunks into dense vector embeddings.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default).
    Output: L2-normalized float32 vectors of dimension 384.
    """

    def __init__(self, config: EmbeddingConfig = None):
        if config is None:
            config = EmbeddingConfig()

        self.config = config
        print(f"  🧠 Loading embedding model: {config.model_name} ...")
        self.model = SentenceTransformer(config.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  ✅ Model loaded. Embedding dimension: {self.dimension}")

    def embed_documents(self, chunks: List[str]) -> np.ndarray:
        """
        Embed a list of document chunks.

        Returns: np.ndarray of shape (n_chunks, dimension), L2-normalized.
        """
        print(f"  📐 Embedding {len(chunks)} document chunks ...")
        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=True,
            batch_size=32,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"  ✅ Document embeddings: shape {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns: np.ndarray of shape (dimension,), L2-normalized.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize,
        )
        return np.array(embedding[0], dtype=np.float32)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Embed multiple queries.

        Returns: np.ndarray of shape (n_queries, dimension), L2-normalized.
        """
        embeddings = self.model.encode(
            queries,
            normalize_embeddings=self.config.normalize,
        )
        return np.array(embeddings, dtype=np.float32)
