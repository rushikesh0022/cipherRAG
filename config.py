# config.py
# ═══════════════════════════════════════════════════════════
#  Central configuration for the Private RAG system
# ═══════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChunkConfig:
    """How to split documents into chunks."""
    chunk_size: int = 20          # words per chunk
    chunk_overlap: int = 5         # overlapping words between chunks
    min_chunk_length: int = 10     # discard chunks shorter than this


@dataclass
class EmbeddingConfig:
    """Which embedding model to use."""
    model_name: str = "all-MiniLM-L6-v2"
    normalize: bool = True         # L2-normalize embeddings


@dataclass
class QuantizationConfig:
    """Int8 quantization settings."""
    scale: int = 120               # max absolute int value (not 127, leaves headroom)
    # 120 avoids overflow edge cases while using most of the int8 range


@dataclass
class BFVConfig:
    """BFV homomorphic encryption parameters."""
    poly_modulus_degree: int = 8192    # security: ~128 bits
    plain_mod_bits: int = 25           # plaintext modulus bit size
    # plain_modulus must be > 2 * max_possible_dot_product
    # max dot = dim * scale^2 = 384 * 120 * 120 = 5,529,600
    # 2^25 = 33,554,432 > 2 * 5,529,600 = 11,059,200  ✓


@dataclass
class RAGConfig:
    """Overall RAG pipeline configuration."""
    top_k: int = 3                     # number of chunks to retrieve
    relevance_threshold: float = 0.02  # minimum normalized top score to accept a match
    no_match_message: str = (
        "Sorry, the query you asked is not present in the document."
    )
    search_mode: str = "ct_ct"         # "ct_ct" (full privacy) or "ct_pt" (query privacy)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    bfv: BFVConfig = field(default_factory=BFVConfig)


# Default configuration
DEFAULT_CONFIG = RAGConfig()
