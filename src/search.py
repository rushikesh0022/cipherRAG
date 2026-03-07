# src/search.py
# ═══════════════════════════════════════════════════════════
#  Three Search Implementations:
#
#  1. PlaintextSearch    — no encryption (speed baseline)
#  2. CKKSSearch         — CKKS float32 (standard encrypted RAG)
#  3. BFVSearch          — BFV int8 (YOUR method)
#
#  All three return the same result format for fair comparison.
# ═══════════════════════════════════════════════════════════

import time
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
from src.encryption import BFVEngine, CKKSEngine


@dataclass
class SearchResult:
    """Unified result format for all three search methods."""
    method: str = ""
    query_text: str = ""
    top_k_indices: List[int] = field(default_factory=list)
    top_k_scores: List[float] = field(default_factory=list)
    all_scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    crypto_error: float = 0.0  # max |encrypted_score - plaintext_score|


# ═══════════════════════════════════════════════════════════
#  METHOD 1: PLAINTEXT SEARCH (no encryption — baseline)
# ═══════════════════════════════════════════════════════════

class PlaintextSearch:
    """
    Standard vector similarity search with NO encryption.

    This is the SPEED BASELINE.
    Obviously not private — server sees everything.
    Used to measure the cost of encryption.
    """

    def __init__(self):
        self.doc_embeddings: np.ndarray = None
        self.n_docs: int = 0

    def index_documents(self, doc_embeddings: np.ndarray):
        """Store document embeddings in plaintext."""
        print(f"\n  📂 [Plaintext] Indexing {len(doc_embeddings)} documents "
              f"(NO encryption)")
        self.doc_embeddings = doc_embeddings
        self.n_docs = len(doc_embeddings)

    def search(self, query_embedding: np.ndarray,
               top_k: int = 3) -> SearchResult:
        """
        Standard dot-product search.
        Server sees query AND documents in plaintext.
        """
        t0 = time.time()

        scores = (self.doc_embeddings @ query_embedding).tolist()

        latency = (time.time() - t0) * 1000

        sorted_idx = list(np.argsort(scores)[::-1])
        top_idx = sorted_idx[:top_k]
        top_scores = [scores[i] for i in top_idx]

        return SearchResult(
            method="plaintext",
            top_k_indices=top_idx,
            top_k_scores=top_scores,
            all_scores=scores,
            latency_ms=latency,
            crypto_error=0.0,
        )

    def search_batch(self, query_embeddings: np.ndarray,
                     query_texts: List[str],
                     top_k: int = 3) -> List[SearchResult]:
        """Run multiple plaintext searches."""
        results = []
        for qi in range(len(query_embeddings)):
            r = self.search(query_embeddings[qi], top_k)
            r.query_text = query_texts[qi] if qi < len(query_texts) else ""
            results.append(r)
        return results


# ═══════════════════════════════════════════════════════════
#  METHOD 2: CKKS FLOAT32 SEARCH (standard encrypted RAG)
# ═══════════════════════════════════════════════════════════

class CKKSSearch:
    """
    CKKS-encrypted vector search on float32 embeddings.

    THIS IS THE "ENEMY" — the standard approach your paper improves on.

    Problems with this approach:
      1. CKKS adds approximation error ε to every dot product
      2. Need score-gap theorem to prove ranking is correct
      3. Slower than BFV (more complex arithmetic)
      4. Larger ciphertexts
    """

    def __init__(self, engine: CKKSEngine):
        self.engine = engine
        self.enc_docs: List[bytes] = []
        self.n_docs: int = 0

    def index_documents(self, float_embeddings: np.ndarray):
        """Encrypt all document vectors with CKKS."""
        self.n_docs = len(float_embeddings)
        self.enc_docs = self.engine.encrypt_documents(float_embeddings)

    def search(self, query_embedding: np.ndarray,
               plaintext_scores: List[float] = None,
               top_k: int = 3) -> SearchResult:
        """
        Encrypted search: E(q) · E(d) = E(q·d) with CKKS.

        Steps:
          1. Client encrypts query
          2. Server computes blind dot products (approximate)
          3. Client decrypts scores
          4. Client selects top-k

        plaintext_scores: if provided, used to compute crypto error
        """
        t0 = time.time()

        # client encrypts query
        enc_q = self.engine.encrypt_vector(query_embedding)

        # server: blind dot products
        enc_score_list = []
        for doc_bytes in self.enc_docs:
            score_bytes = self.engine.server_dot_product(enc_q, doc_bytes)
            enc_score_list.append(score_bytes)

        # client: decrypt scores
        scores = []
        for s_bytes in enc_score_list:
            scores.append(self.engine.decrypt_score(s_bytes))

        latency = (time.time() - t0) * 1000

        # compute crypto error if ground truth provided
        max_error = 0.0
        if plaintext_scores is not None:
            errors = [abs(plaintext_scores[i] - scores[i])
                      for i in range(len(scores))]
            max_error = max(errors)

        # select top-k
        sorted_idx = list(np.argsort(scores)[::-1])
        top_idx = sorted_idx[:top_k]
        top_scores = [scores[i] for i in top_idx]

        return SearchResult(
            method="CKKS-float32",
            top_k_indices=top_idx,
            top_k_scores=top_scores,
            all_scores=scores,
            latency_ms=latency,
            crypto_error=max_error,
        )

    def search_batch(self, query_embeddings: np.ndarray,
                     doc_embeddings_plain: np.ndarray,
                     query_texts: List[str],
                     top_k: int = 3) -> List[SearchResult]:
        """Run multiple CKKS-encrypted searches."""
        print(f"\n  🔍 [CKKS] Running {len(query_embeddings)} encrypted queries ...")
        results = []
        for qi in range(len(query_embeddings)):
            # compute plaintext scores for error measurement
            plain_scores = (doc_embeddings_plain @ query_embeddings[qi]).tolist()

            r = self.search(
                query_embeddings[qi],
                plaintext_scores=plain_scores,
                top_k=top_k,
            )
            r.query_text = query_texts[qi] if qi < len(query_texts) else ""
            results.append(r)

            print(f"     [{qi+1}/{len(query_embeddings)}] "
                  f"'{r.query_text[:35]}' "
                  f"err={r.crypto_error:.2e} "
                  f"lat={r.latency_ms:.0f}ms", end='\r')

        avg_lat = np.mean([r.latency_ms for r in results])
        print(f"\n  ✅ [CKKS] Done. Avg latency: {avg_lat:.0f}ms/query")
        return results


# ═══════════════════════════════════════════════════════════
#  METHOD 3: BFV INT8 SEARCH (YOUR METHOD)
# ═══════════════════════════════════════════════════════════

class BFVSearch:
    """
    BFV-encrypted vector search on int8-quantized embeddings.

    THIS IS YOUR METHOD — the novel contribution.

    Why this is better:
      1. BFV gives EXACT integer arithmetic (zero crypto error)
      2. Int8 quantization preserves ranking (verified empirically)
      3. Faster than CKKS
      4. Smaller ciphertexts
      5. Score-gap theorem trivially satisfied (ε = 0)
    """

    def __init__(self, engine: BFVEngine):
        self.engine = engine
        self.enc_docs: List[bytes] = []
        self.int_docs: np.ndarray = None
        self.n_docs: int = 0

    def index_documents(self, int_docs: np.ndarray,
                        mode: str = "ct_ct"):
        """
        Index int8-quantized documents.
        ct_ct: encrypt everything (full privacy)
        ct_pt: keep docs plaintext (query privacy only)
        """
        self.int_docs = int_docs
        self.n_docs = len(int_docs)

        if mode == "ct_ct":
            self.enc_docs = self.engine.encrypt_documents(int_docs)
        else:
            print(f"\n  📂 [BFV ct×pt] Storing {len(int_docs)} docs "
                  f"in plaintext")
            self.enc_docs = []

    def search(self, int_query: np.ndarray,
               mode: str = "ct_ct",
               int_plaintext_scores: List[int] = None,
               top_k: int = 3) -> SearchResult:
        """
        Encrypted search with BFV.

        ct×ct: E(q) · E(d) = E(q·d)  — full privacy
        ct×pt: E(q) · d    = E(q·d)  — query privacy only
        """
        t0 = time.time()

        # client encrypts query
        enc_q = self.engine.encrypt_vector(int_query)

        # server: blind dot products
        enc_score_list = []
        for i in range(self.n_docs):
            if mode == "ct_ct":
                score_bytes = self.engine.server_dot_product_ct_ct(
                    enc_q, self.enc_docs[i]
                )
            else:
                score_bytes = self.engine.server_dot_product_ct_pt(
                    enc_q, self.int_docs[i]
                )
            enc_score_list.append(score_bytes)

        # client: decrypt scores
        scores = []
        for s_bytes in enc_score_list:
            scores.append(self.engine.decrypt_score(s_bytes))

        latency = (time.time() - t0) * 1000

        # crypto error vs plaintext int8
        max_error = 0.0
        if int_plaintext_scores is not None:
            errors = [abs(int(int_plaintext_scores[i]) - scores[i])
                      for i in range(len(scores))]
            max_error = max(errors)

        # select top-k
        sorted_idx = list(np.argsort(scores)[::-1])
        top_idx = sorted_idx[:top_k]
        top_scores = [float(scores[i]) for i in top_idx]

        return SearchResult(
            method=f"BFV-int8-{mode}",
            top_k_indices=top_idx,
            top_k_scores=top_scores,
            all_scores=[float(s) for s in scores],
            latency_ms=latency,
            crypto_error=float(max_error),
        )

    def search_batch(self, int_queries: np.ndarray,
                     int_docs: np.ndarray,
                     query_texts: List[str],
                     mode: str = "ct_ct",
                     top_k: int = 3) -> List[SearchResult]:
        """Run multiple BFV-encrypted searches."""
        print(f"\n  🔍 [BFV {mode}] Running {len(int_queries)} "
              f"encrypted queries ...")
        results = []
        for qi in range(len(int_queries)):
            # plaintext int8 scores for error measurement
            plain_int_scores = (int_docs @ int_queries[qi]).tolist()

            r = self.search(
                int_queries[qi],
                mode=mode,
                int_plaintext_scores=plain_int_scores,
                top_k=top_k,
            )
            r.query_text = query_texts[qi] if qi < len(query_texts) else ""
            results.append(r)

            print(f"     [{qi+1}/{len(int_queries)}] "
                  f"'{r.query_text[:35]}' "
                  f"err={r.crypto_error:.0f} "
                  f"lat={r.latency_ms:.0f}ms", end='\r')

        avg_lat = np.mean([r.latency_ms for r in results])
        print(f"\n  ✅ [BFV {mode}] Done. Avg latency: {avg_lat:.0f}ms/query")
        return results
