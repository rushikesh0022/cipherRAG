# src/rag_pipeline.py
# ═══════════════════════════════════════════════════════════
#  End-to-End Private RAG Pipeline
#
#  PDF → Chunks → Embeddings → Int8 → BFV Encrypt → Search
#
#  This is the main class that ties everything together.
# ═══════════════════════════════════════════════════════════

import time
import numpy as np
from typing import List, Dict, Optional

from config import RAGConfig, DEFAULT_CONFIG
from src.chunker import load_document, create_sample_corpus
from src.embedder import Embedder
from src.quantizer import Int8Quantizer
from src.encryption import BFVEngine
from src.search import BFVSearch, SearchResult


class PrivateRAGPipeline:
    """
    Complete privacy-preserving RAG pipeline.

    Architecture:
      CLIENT                          SERVER (untrusted)
      ──────                          ──────────────────
      PDF → chunks → embed
        → quantize (int8)
        → encrypt (BFV)   ─────────→  store E(docs)

      query → embed
        → quantize (int8)
        → encrypt (BFV)   ─────────→  compute E(q)·E(d) for all d
                           ←─────────  return E(scores)
      decrypt scores
      select top-k
      feed to local LLM
                                       Server never sees plaintext.
    """

    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = DEFAULT_CONFIG

        self.config = config
        self.chunks: List[str] = []
        self.doc_embeddings: np.ndarray = None
        self.int_docs: np.ndarray = None
        self.int_queries: np.ndarray = None

        # components
        self.embedder: Optional[Embedder] = None
        self.quantizer: Optional[Int8Quantizer] = None
        self.engine: Optional[BFVEngine] = None
        self.searcher: Optional[BFVSearch] = None

        # state tracking
        self.is_indexed = False
        self.timings: Dict[str, float] = {}

    def _normalize_int_score(self, raw_score: float) -> float:
        """
        Map an int dot-product score into [-1, 1] so one threshold works across settings.
        """
        if self.quantizer is None or self.doc_embeddings is None:
            return 0.0

        max_possible = self.quantizer.get_max_dot_product(self.doc_embeddings.shape[1])
        if max_possible <= 0:
            return 0.0

        return float(raw_score) / float(max_possible)

    def _build_relevance_info(self, top_score: float) -> Dict[str, float]:
        normalized = self._normalize_int_score(top_score)
        has_match = normalized >= self.config.relevance_threshold
        return {
            "top_score": float(top_score),
            "top_score_normalized": normalized,
            "relevance_threshold": float(self.config.relevance_threshold),
            "has_match": has_match,
        }

    def ingest_document(self, path: str) -> 'PrivateRAGPipeline':
        """
        Step 1: Load and chunk a document.
        """
        print("\n" + "=" * 60)
        print("  STEP 1: DOCUMENT INGESTION")
        print("=" * 60)

        t0 = time.time()
        self.chunks = load_document(path, self.config.chunk)
        self.timings['ingestion'] = time.time() - t0

        return self

    def ingest_chunks(self, chunks: List[str]) -> 'PrivateRAGPipeline':
        """
        Step 1 (alternative): Use pre-made chunks.
        """
        print("\n" + "=" * 60)
        print("  STEP 1: LOADING CHUNKS")
        print("=" * 60)

        self.chunks = chunks
        print(f"  📝 Loaded {len(chunks)} chunks")
        self.timings['ingestion'] = 0.0

        return self

    def generate_embeddings(self) -> 'PrivateRAGPipeline':
        """
        Step 2: Generate float32 embeddings for all chunks.
        """
        print("\n" + "=" * 60)
        print("  STEP 2: EMBEDDING GENERATION")
        print("=" * 60)

        t0 = time.time()
        self.embedder = Embedder(self.config.embedding)
        self.doc_embeddings = self.embedder.embed_documents(self.chunks)
        self.timings['embedding'] = time.time() - t0

        return self

    def quantize(self) -> 'PrivateRAGPipeline':
        """
        Step 3: Quantize float32 → int8.
        """
        print("\n" + "=" * 60)
        print("  STEP 3: INT8 QUANTIZATION")
        print("=" * 60)

        t0 = time.time()
        self.quantizer = Int8Quantizer(self.config.quantization)
        self.int_docs = self.quantizer.quantize_documents(self.doc_embeddings)
        self.timings['quantization'] = time.time() - t0

        # verify max dot product fits in BFV plaintext space
        max_dot = self.quantizer.get_max_dot_product(self.doc_embeddings.shape[1])
        print(f"\n  📊 Max possible dot product: {max_dot:,}")
        print(f"     BFV plain_mod needs to be > {max_dot * 2:,}")

        return self

    def encrypt_and_index(self) -> 'PrivateRAGPipeline':
        """
        Step 4: Set up BFV encryption and index documents.
        """
        print("\n" + "=" * 60)
        print("  STEP 4: BFV ENCRYPTION & INDEXING")
        print("=" * 60)

        t0 = time.time()

        # create encryption engine
        self.engine = BFVEngine(self.config.bfv)
        self.engine.setup()

        # create searcher and index documents
        self.searcher = BFVSearch(self.engine)
        self.searcher.index_documents(
            self.int_docs,
            mode=self.config.search_mode,
        )

        self.timings['encryption'] = time.time() - t0
        self.is_indexed = True

        return self

    def verify(self, query_embeddings: np.ndarray = None) -> Dict:
        """
        Step 4.5 (optional): Verify correctness.

        Checks that:
          1. Int8 quantization preserves ranking
          2. BFV encryption gives exact int8 results
        """
        print("\n" + "=" * 60)
        print("  VERIFICATION")
        print("=" * 60)

        results = {}

        # verify quantization
        if query_embeddings is not None:
            quant_result = self.quantizer.verify_ranking(
                self.doc_embeddings, query_embeddings,
                top_k=self.config.top_k,
            )
            results['quantization'] = quant_result

        # verify BFV exactness — done INLINE (no dependency on engine method)
        if (self.config.search_mode == "ct_ct"
                and self.searcher is not None
                and len(self.searcher.enc_docs) > 0):

            print(f"\n  🔬 Verifying BFV exactness ...")

            import tenseal as ts

            test_query = self.int_docs[0]

            # plaintext int8 dot products
            plain_scores = (self.int_docs @ test_query).tolist()

            # encrypted dot products
            enc_q = self.engine.encrypt_vector(test_query)
            bfv_scores = []

            for doc_bytes in self.searcher.enc_docs:
                score_bytes = self.engine.server_dot_product_ct_ct(
                    enc_q, doc_bytes
                )
                bfv_scores.append(self.engine.decrypt_score(score_bytes))

            # compare
            errors = [abs(int(plain_scores[i]) - bfv_scores[i])
                      for i in range(len(plain_scores))]
            max_error = max(errors)
            mean_error = sum(errors) / len(errors)

            bfv_result = {
                "max_error": max_error,
                "mean_error": mean_error,
                "is_exact": max_error == 0,
                "n_vectors": len(plain_scores),
            }

            if max_error == 0:
                print(f"  ✅ PERFECT: BFV gives EXACTLY the same dot products")
                print(f"     as plaintext int8 computation. Zero crypto error.")
                print(f"     Verified across all {len(plain_scores)} document vectors.")
            else:
                print(f"  ⚠️  Max error: {max_error}")
                print(f"     Mean error: {mean_error:.6f}")

            results['bfv_exactness'] = bfv_result

        return results

    def search(self, query: str, top_k: int = None) -> Dict:
        """
        Step 5: Search with a natural language query.

        This is the main user-facing function.
        """
        if not self.is_indexed:
            raise RuntimeError("Call encrypt_and_index() first")

        if top_k is None:
            top_k = self.config.top_k

        # embed the query
        query_embedding = self.embedder.embed_query(query)

        # quantize the query (using same scale as documents)
        int_query = self.quantizer.quantize_query(query_embedding)

        # encrypted search
        result: SearchResult = self.searcher.search(
            int_query,
            top_k=top_k,
            mode=self.config.search_mode,
        )
        result.query_text = query

        # build response
        retrieved_chunks = []
        for idx in result.top_k_indices:
            retrieved_chunks.append({
                "index": idx,
                "text": self.chunks[idx],
                "score": result.all_scores[idx],
            })

        top_score = result.top_k_scores[0] if result.top_k_scores else 0.0
        relevance = self._build_relevance_info(top_score)

        message = "Relevant chunks found."
        if not relevance["has_match"]:
            message = self.config.no_match_message

        return {
            "query": query,
            "mode": self.config.search_mode,
            "top_k": top_k,
            "results": retrieved_chunks,
            "latency_ms": result.latency_ms,
            "has_match": relevance["has_match"],
            "message": message,
            "top_score": relevance["top_score"],
            "top_score_normalized": relevance["top_score_normalized"],
            "relevance_threshold": relevance["relevance_threshold"],
        }

    def search_batch(self, queries: List[str],
                     top_k: int = None) -> List[Dict]:
        """
        Search with multiple queries. Returns list of results.
        """
        if not self.is_indexed:
            raise RuntimeError("Call encrypt_and_index() first")

        if top_k is None:
            top_k = self.config.top_k

        # embed all queries
        query_embeddings = self.embedder.embed_queries(queries)

        # quantize all queries
        int_queries = np.array([
            self.quantizer.quantize_query(qe) for qe in query_embeddings
        ])

        # run encrypted search
        search_results: List[SearchResult] = self.searcher.search_batch(
            int_queries,
            self.int_docs,
            queries,
            mode=self.config.search_mode,
            top_k=top_k,
        )

        # build responses
        all_results = []
        for i, sr in enumerate(search_results):
            retrieved = []
            for idx in sr.top_k_indices:
                retrieved.append({
                    "index": idx,
                    "text": self.chunks[idx],
                    "score": sr.all_scores[idx],
                })

            top_score = sr.top_k_scores[0] if sr.top_k_scores else 0.0
            relevance = self._build_relevance_info(top_score)

            message = "Relevant chunks found."
            if not relevance["has_match"]:
                message = self.config.no_match_message

            all_results.append({
                "query": queries[i],
                "mode": self.config.search_mode,
                "top_k": top_k,
                "results": retrieved,
                "latency_ms": sr.latency_ms,
                "has_match": relevance["has_match"],
                "message": message,
                "top_score": relevance["top_score"],
                "top_score_normalized": relevance["top_score_normalized"],
                "relevance_threshold": relevance["relevance_threshold"],
            })

        return all_results

    def print_summary(self):
        """Print a summary of the pipeline state and timings."""
        print("\n" + "=" * 60)
        print("  PIPELINE SUMMARY")
        print("=" * 60)
        print(f"""
  Document chunks:     {len(self.chunks)}
  Embedding dimension: {self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 'N/A'}
  Quantization scale:  {self.quantizer.config.scale if self.quantizer else 'N/A'}
  Encryption scheme:   BFV
  Search mode:         {self.config.search_mode}
  Top-k:               {self.config.top_k}
  Indexed:             {'✅' if self.is_indexed else '❌'}

  TIMINGS:
  ────────""")
        for step, t in self.timings.items():
            print(f"    {step:<20} {t:.2f}s")
        total = sum(self.timings.values())
        print(f"    {'TOTAL':<20} {total:.2f}s")
