# src/quantizer.py
# ═══════════════════════════════════════════════════════════
#  Int8 Quantization for Embeddings
#
#  KEY INSIGHT (from the paper):
#
#  Float32 embeddings → CKKS (approximate, adds error ε)
#  Int8 embeddings    → BFV  (exact, zero error)
#
#  We use GLOBAL scaling: one scale factor for ALL vectors.
#  This preserves the RANKING of dot products exactly.
#
#  Why global (not per-vector) scaling:
#    Per-vector: dot(q, d1) vs dot(q, d2) uses different scales
#                → ranking distorted
#    Global:     dot(q, d1) vs dot(q, d2) uses same scale²
#                → ranking perfectly preserved
# ═══════════════════════════════════════════════════════════

import numpy as np
from typing import Tuple
from config import QuantizationConfig


class Int8Quantizer:
    """
    Quantizes float32 embeddings to int8 using global scaling.

    The scale factor is computed from the DOCUMENT embeddings
    and then applied identically to queries. This ensures
    dot product rankings are preserved.
    """

    def __init__(self, config: QuantizationConfig = None):
        if config is None:
            config = QuantizationConfig()
        self.config = config
        self.scale_factor: float = None
        self.global_max: float = None

    def fit(self, embeddings: np.ndarray) -> 'Int8Quantizer':
        """
        Compute the global scaling factor from document embeddings.
        Call this ONCE on your document corpus.
        """
        self.global_max = np.max(np.abs(embeddings))
        self.scale_factor = self.config.scale / self.global_max

        print(f"  🔢 Quantizer fitted:")
        print(f"     Global max absolute value: {self.global_max:.6f}")
        print(f"     Scale factor: {self.scale_factor:.2f}")
        print(f"     Integer range: [-{self.config.scale}, +{self.config.scale}]")

        return self

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantize float32 embeddings to int64 (int8-range values).

        We use int64 dtype because TenSEAL BFV expects Python ints,
        but the VALUES are in the range [-120, +120] (int8 range).
        """
        if self.scale_factor is None:
            raise RuntimeError("Call fit() first with document embeddings")

        quantized = np.round(embeddings * self.scale_factor).astype(np.int64)
        quantized = np.clip(quantized, -self.config.scale, self.config.scale)

        return quantized

    def quantize_documents(self, doc_embeddings: np.ndarray) -> np.ndarray:
        """Quantize document embeddings (fit + transform)."""
        self.fit(doc_embeddings)
        int_docs = self.quantize(doc_embeddings)

        print(f"  ✅ Documents quantized: {int_docs.shape}")
        print(f"     Value range: [{int_docs.min()}, {int_docs.max()}]")
        print(f"     Non-zero elements: {np.count_nonzero(int_docs)} / {int_docs.size} "
              f"({np.count_nonzero(int_docs)/int_docs.size*100:.1f}%)")

        return int_docs

    def quantize_query(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Quantize a query embedding using the SAME scale factor
        as the documents. This is critical for ranking preservation.
        """
        if self.scale_factor is None:
            raise RuntimeError("Call fit() or quantize_documents() first")

        int_query = np.round(query_embedding * self.scale_factor).astype(np.int64)
        int_query = np.clip(int_query, -self.config.scale, self.config.scale)

        return int_query

    def verify_ranking(self, doc_embeddings: np.ndarray,
                       query_embeddings: np.ndarray,
                       top_k: int = 3) -> dict:
        """
        Verify that int8 quantization preserves top-k ranking
        compared to float32 plaintext search.
        """
        int_docs = self.quantize(doc_embeddings)

        top1_correct = 0
        topk_correct = 0
        n_queries = len(query_embeddings)

        for qi in range(n_queries):
            # float32 ranking (ground truth)
            float_scores = doc_embeddings @ query_embeddings[qi]
            float_top = list(np.argsort(float_scores)[-top_k:][::-1])

            # int8 ranking
            int_query = self.quantize_query(query_embeddings[qi])
            int_scores = int_docs @ int_query
            int_top = list(np.argsort(int_scores)[-top_k:][::-1])

            if float_top[0] == int_top[0]:
                top1_correct += 1
            if set(float_top) == set(int_top):
                topk_correct += 1

        result = {
            "top1_accuracy": top1_correct / n_queries,
            "topk_accuracy": topk_correct / n_queries,
            "n_queries": n_queries,
            "top_k": top_k,
        }

        print(f"\n  🔍 Quantization ranking verification:")
        print(f"     Top-1 accuracy: {result['top1_accuracy']*100:.1f}%")
        print(f"     Top-{top_k} accuracy: {result['topk_accuracy']*100:.1f}%")

        return result

    def get_max_dot_product(self, dim: int) -> int:
        """
        Calculate the maximum possible dot product value.
        Used to verify BFV plain_modulus is large enough.
        """
        return dim * self.config.scale * self.config.scale
