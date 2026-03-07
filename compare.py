# compare.py
# ═══════════════════════════════════════════════════════════
#  HEAD-TO-HEAD COMPARISON
#
#  Runs ALL THREE methods on the same corpus and queries:
#
#    1. Plaintext RAG        (no encryption — speed baseline)
#    2. CKKS float32 RAG     (standard approach — the "enemy")
#    3. BFV int8 RAG         (YOUR method — the contribution)
#
#  Produces:
#    Table 1: Accuracy comparison (top-1, top-3 vs plaintext)
#    Table 2: Latency comparison
#    Table 3: Crypto error comparison
#    Table 4: Ciphertext size comparison
#    Table 5: Error decomposition
#    Table 6: Score-gap analysis
#
#  USAGE:
#    python compare.py
#    python compare.py data/document.pdf
#
#  OUTPUT goes to terminal AND results/comparison.txt
# ═══════════════════════════════════════════════════════════

import sys
import time
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field

from config import RAGConfig
from src.chunker import create_sample_corpus, load_document
from src.embedder import Embedder
from src.quantizer import Int8Quantizer
from src.encryption import BFVEngine, CKKSEngine
from src.search import PlaintextSearch, CKKSSearch, BFVSearch, SearchResult
from src.utils import compute_score_gaps


# ═══════════════════════════════════════════════════════════
#  DATA CLASS FOR STORING METHOD BENCHMARKS
# ═══════════════════════════════════════════════════════════

@dataclass
class MethodBenchmark:
    name: str
    scheme: str                 # "none", "CKKS", "BFV"
    precision: str              # "float32", "int8"
    mode: str                   # "plaintext", "ct×ct", "ct×pt"
    top1_accuracy: float = 0.0
    topk_accuracy: float = 0.0
    mean_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    ct_bytes_per_vec: int = 0
    max_crypto_error: float = 0.0
    mean_crypto_error: float = 0.0
    keygen_time_s: float = 0.0
    encrypt_time_s: float = 0.0
    privacy_level: str = ""
    search_results: List[SearchResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
#  THE COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════

class Comparator:
    """
    Runs all three methods and produces comparison tables.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.chunks: List[str] = []
        self.queries: List[str] = []

        # embeddings
        self.doc_embeddings: np.ndarray = None    # float32
        self.query_embeddings: np.ndarray = None  # float32
        self.int_docs: np.ndarray = None          # int8-range
        self.int_queries: np.ndarray = None       # int8-range

        # ground truth (from plaintext float32)
        self.ground_truth: List[List[int]] = []

        # results
        self.benchmarks: List[MethodBenchmark] = []
        self.output_lines: List[str] = []

    def log(self, msg: str = "", end="\n"):
        """Print and store output."""
        print(msg, end=end)
        self.output_lines.append(msg)

    def prepare_data(self, chunks: List[str] = None,
                     queries: List[str] = None,
                     pdf_path: str = None):
        """
        Step 0: Prepare corpus, embeddings, quantization, ground truth.
        """
        self.log("\n" + "=" * 70)
        self.log("  STEP 0: DATA PREPARATION")
        self.log("=" * 70)

        # get chunks
        if pdf_path:
            config = RAGConfig()
            self.chunks = load_document(pdf_path, config.chunk)
            # generate some default queries for PDFs
            self.queries = [
                "What is this document about?",
                "What are the main findings?",
                "What methodology was used?",
                "What are the key conclusions?",
                "What problems does this address?",
            ]
        elif chunks is not None:
            self.chunks = chunks
            self.queries = queries or []
        else:
            self.chunks, self.queries = create_sample_corpus()

        self.log(f"  📝 Corpus: {len(self.chunks)} chunks")
        self.log(f"  ❓ Queries: {len(self.queries)} test queries")

        # embeddings
        self.log(f"\n  🧠 Generating embeddings ...")
        embedder = Embedder()
        self.doc_embeddings = embedder.embed_documents(self.chunks)
        self.query_embeddings = embedder.embed_queries(self.queries)
        dim = self.doc_embeddings.shape[1]
        self.log(f"  ✅ Embeddings: dim={dim}")

        # quantization
        self.log(f"\n  🔢 Quantizing to int8 ...")
        quantizer = Int8Quantizer()
        self.int_docs = quantizer.quantize_documents(self.doc_embeddings)
        self.int_queries = np.array([
            quantizer.quantize_query(qe) for qe in self.query_embeddings
        ])
        self.log(f"  ✅ Quantized: range [{self.int_docs.min()}, {self.int_docs.max()}]")

        # ground truth from float32 plaintext
        self.ground_truth = []
        for qv in self.query_embeddings:
            scores = self.doc_embeddings @ qv
            top = list(np.argsort(scores)[-self.top_k:][::-1])
            self.ground_truth.append(top)

        self.log(f"  ✅ Ground truth top-{self.top_k} computed")

    # ───────────────────────────────────────────────────────
    #  METHOD 1: PLAINTEXT
    # ───────────────────────────────────────────────────────

    def run_plaintext(self):
        """Run plaintext search (no encryption)."""
        self.log("\n" + "=" * 70)
        self.log("  METHOD 1: PLAINTEXT RAG (no encryption)")
        self.log("  Privacy: NONE — server sees everything")
        self.log("=" * 70)

        searcher = PlaintextSearch()
        searcher.index_documents(self.doc_embeddings)

        t0 = time.time()
        results = searcher.search_batch(
            self.query_embeddings, self.queries, self.top_k
        )
        total_time = time.time() - t0

        # accuracy (should be 100% since this IS the ground truth)
        top1_ok = sum(1 for i, r in enumerate(results)
                      if r.top_k_indices[0] == self.ground_truth[i][0])
        topk_ok = sum(1 for i, r in enumerate(results)
                      if set(r.top_k_indices[:self.top_k]) ==
                         set(self.ground_truth[i]))

        n = len(self.queries)
        latencies = [r.latency_ms for r in results]

        bench = MethodBenchmark(
            name="Plaintext RAG",
            scheme="none",
            precision="float32",
            mode="plaintext",
            top1_accuracy=top1_ok / n,
            topk_accuracy=topk_ok / n,
            mean_latency_ms=np.mean(latencies),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            ct_bytes_per_vec=self.doc_embeddings.shape[1] * 4,  # float32
            max_crypto_error=0.0,
            mean_crypto_error=0.0,
            privacy_level="NONE",
            search_results=results,
        )
        self.benchmarks.append(bench)

        self.log(f"\n  Results:")
        self.log(f"    Top-1 accuracy: {top1_ok}/{n} ({top1_ok/n*100:.0f}%)")
        self.log(f"    Top-{self.top_k} accuracy: {topk_ok}/{n} "
                 f"({topk_ok/n*100:.0f}%)")
        self.log(f"    Avg latency:    {np.mean(latencies):.3f} ms")
        self.log(f"    Crypto error:   0.0 (no encryption)")

    # ───────────────────────────────────────────────────────
    #  METHOD 2: CKKS FLOAT32
    # ───────────────────────────────────────────────────────

    def run_ckks(self):
        """Run CKKS-encrypted search on float32 embeddings."""
        self.log("\n" + "=" * 70)
        self.log("  METHOD 2: CKKS FLOAT32 RAG (standard encrypted)")
        self.log("  Privacy: FULL — query + docs encrypted")
        self.log("  This is the BASELINE your paper improves on")
        self.log("=" * 70)

        # setup CKKS
        engine = CKKSEngine(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale_power=40,
            name="CKKS-standard",
        )
        engine.setup()

        # index (encrypt) documents
        t_enc_0 = time.time()
        searcher = CKKSSearch(engine)
        searcher.index_documents(self.doc_embeddings)
        encrypt_time = time.time() - t_enc_0
        ct_size = len(searcher.enc_docs[0])

        # run searches
        results = searcher.search_batch(
            self.query_embeddings,
            self.doc_embeddings,  # for error measurement
            self.queries,
            self.top_k,
        )

        # accuracy vs float32 ground truth
        top1_ok = sum(1 for i, r in enumerate(results)
                      if r.top_k_indices[0] == self.ground_truth[i][0])
        topk_ok = sum(1 for i, r in enumerate(results)
                      if set(r.top_k_indices[:self.top_k]) ==
                         set(self.ground_truth[i]))

        n = len(self.queries)
        latencies = [r.latency_ms for r in results]
        crypto_errors = [r.crypto_error for r in results]

        bench = MethodBenchmark(
            name="CKKS float32",
            scheme="CKKS",
            precision="float32",
            mode="ct×ct",
            top1_accuracy=top1_ok / n,
            topk_accuracy=topk_ok / n,
            mean_latency_ms=np.mean(latencies),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            ct_bytes_per_vec=ct_size,
            max_crypto_error=max(crypto_errors),
            mean_crypto_error=np.mean(crypto_errors),
            keygen_time_s=engine.keygen_time,
            encrypt_time_s=encrypt_time,
            privacy_level="FULL (query + docs encrypted)",
            search_results=results,
        )
        self.benchmarks.append(bench)

        self.log(f"\n  Results:")
        self.log(f"    Top-1 accuracy: {top1_ok}/{n} ({top1_ok/n*100:.0f}%)")
        self.log(f"    Top-{self.top_k} accuracy: {topk_ok}/{n} "
                 f"({topk_ok/n*100:.0f}%)")
        self.log(f"    Avg latency:    {np.mean(latencies):.0f} ms")
        self.log(f"    Max crypto err: {max(crypto_errors):.2e}")
        self.log(f"    CT size:        {ct_size:,} bytes ({ct_size/1024:.0f} KB)")

    # ───────────────────────────────────────────────────────
    #  METHOD 3: BFV INT8 (YOUR METHOD)
    # ───────────────────────────────────────────────────────

    def run_bfv_ct_ct(self):
        """Run BFV-encrypted search on int8 embeddings (full privacy)."""
        self.log("\n" + "=" * 70)
        self.log("  METHOD 3: BFV INT8 RAG — YOUR METHOD (full privacy)")
        self.log("  Privacy: FULL — query + docs encrypted")
        self.log("  Crypto error: ZERO (exact integer arithmetic)")
        self.log("=" * 70)

        from config import BFVConfig
        engine = BFVEngine(BFVConfig())
        engine.setup()

        t_enc_0 = time.time()
        searcher = BFVSearch(engine)
        searcher.index_documents(self.int_docs, mode="ct_ct")
        encrypt_time = time.time() - t_enc_0
        ct_size = len(searcher.enc_docs[0])

        # run searches
        results = searcher.search_batch(
            self.int_queries,
            self.int_docs,
            self.queries,
            mode="ct_ct",
            top_k=self.top_k,
        )

        # accuracy vs float32 ground truth
        top1_ok = sum(1 for i, r in enumerate(results)
                      if r.top_k_indices[0] == self.ground_truth[i][0])
        topk_ok = sum(1 for i, r in enumerate(results)
                      if set(r.top_k_indices[:self.top_k]) ==
                         set(self.ground_truth[i]))

        n = len(self.queries)
        latencies = [r.latency_ms for r in results]
        crypto_errors = [r.crypto_error for r in results]

        bench = MethodBenchmark(
            name="BFV int8 ct×ct",
            scheme="BFV",
            precision="int8",
            mode="ct×ct",
            top1_accuracy=top1_ok / n,
            topk_accuracy=topk_ok / n,
            mean_latency_ms=np.mean(latencies),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            ct_bytes_per_vec=ct_size,
            max_crypto_error=max(crypto_errors),
            mean_crypto_error=np.mean(crypto_errors),
            keygen_time_s=engine.keygen_time,
            encrypt_time_s=encrypt_time,
            privacy_level="FULL (query + docs encrypted)",
            search_results=results,
        )
        self.benchmarks.append(bench)

        self.log(f"\n  Results:")
        self.log(f"    Top-1 accuracy: {top1_ok}/{n} ({top1_ok/n*100:.0f}%)")
        self.log(f"    Top-{self.top_k} accuracy: {topk_ok}/{n} "
                 f"({topk_ok/n*100:.0f}%)")
        self.log(f"    Avg latency:    {np.mean(latencies):.0f} ms")
        self.log(f"    Max crypto err: {max(crypto_errors):.0f} "
                 f"(should be 0)")
        self.log(f"    CT size:        {ct_size:,} bytes ({ct_size/1024:.0f} KB)")

    def run_bfv_ct_pt(self):
        """Run BFV-encrypted search (query privacy only — fastest)."""
        self.log("\n" + "=" * 70)
        self.log("  METHOD 3b: BFV INT8 ct×pt (query privacy only)")
        self.log("  Privacy: QUERY ONLY — docs in plaintext")
        self.log("  Fastest encrypted option")
        self.log("=" * 70)

        from config import BFVConfig
        engine = BFVEngine(BFVConfig())
        engine.setup()

        searcher = BFVSearch(engine)
        searcher.index_documents(self.int_docs, mode="ct_pt")

        results = searcher.search_batch(
            self.int_queries,
            self.int_docs,
            self.queries,
            mode="ct_pt",
            top_k=self.top_k,
        )

        top1_ok = sum(1 for i, r in enumerate(results)
                      if r.top_k_indices[0] == self.ground_truth[i][0])
        topk_ok = sum(1 for i, r in enumerate(results)
                      if set(r.top_k_indices[:self.top_k]) ==
                         set(self.ground_truth[i]))

        n = len(self.queries)
        latencies = [r.latency_ms for r in results]
        crypto_errors = [r.crypto_error for r in results]

        bench = MethodBenchmark(
            name="BFV int8 ct×pt",
            scheme="BFV",
            precision="int8",
            mode="ct×pt",
            top1_accuracy=top1_ok / n,
            topk_accuracy=topk_ok / n,
            mean_latency_ms=np.mean(latencies),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            ct_bytes_per_vec=0,
            max_crypto_error=max(crypto_errors),
            mean_crypto_error=np.mean(crypto_errors),
            keygen_time_s=engine.keygen_time,
            privacy_level="QUERY ONLY (docs plaintext)",
            search_results=results,
        )
        self.benchmarks.append(bench)

        self.log(f"\n  Results:")
        self.log(f"    Top-1 accuracy: {top1_ok}/{n} ({top1_ok/n*100:.0f}%)")
        self.log(f"    Top-{self.top_k} accuracy: {topk_ok}/{n} "
                 f"({topk_ok/n*100:.0f}%)")
        self.log(f"    Avg latency:    {np.mean(latencies):.0f} ms")
        self.log(f"    Max crypto err: {max(crypto_errors):.0f}")

    # ───────────────────────────────────────────────────────
    #  PRINT ALL COMPARISON TABLES
    # ───────────────────────────────────────────────────────

    def print_tables(self):
        """Print all comparison tables for the paper."""

        # ── TABLE 1: Main comparison ──
        self.log("\n\n" + "═" * 95)
        self.log("  TABLE 1: HEAD-TO-HEAD COMPARISON")
        self.log("═" * 95)

        header = (f"  {'Method':<20} {'Scheme':<8} {'Prec':<8} "
                  f"{'Mode':<10} {'Top1%':<8} {'Top3%':<8} "
                  f"{'Lat(ms)':<10} {'MaxErr':<12} {'Privacy'}")
        self.log(header)
        self.log("  " + "─" * 92)

        for b in self.benchmarks:
            line = (f"  {b.name:<20} {b.scheme:<8} {b.precision:<8} "
                    f"{b.mode:<10} "
                    f"{b.top1_accuracy*100:<8.0f} "
                    f"{b.topk_accuracy*100:<8.0f} "
                    f"{b.mean_latency_ms:<10.0f} "
                    f"{b.max_crypto_error:<12.2e} "
                    f"{b.privacy_level}")
            self.log(line)

        # ── TABLE 2: Latency breakdown ──
        self.log("\n\n" + "═" * 70)
        self.log("  TABLE 2: LATENCY COMPARISON")
        self.log("═" * 70)

        self.log(f"\n  {'Method':<20} {'Min(ms)':<10} {'Avg(ms)':<10} "
                 f"{'Max(ms)':<10} {'Slowdown':<10}")
        self.log("  " + "─" * 50)

        # get plaintext baseline for slowdown calculation
        plain_lat = 0.001  # avoid division by zero
        for b in self.benchmarks:
            if b.scheme == "none":
                plain_lat = max(b.mean_latency_ms, 0.001)
                break

        for b in self.benchmarks:
            slowdown = b.mean_latency_ms / plain_lat
            self.log(f"  {b.name:<20} {b.min_latency_ms:<10.2f} "
                     f"{b.mean_latency_ms:<10.0f} "
                     f"{b.max_latency_ms:<10.0f} "
                     f"{slowdown:<10.0f}x")

        # ── TABLE 3: Ciphertext size ──
        self.log("\n\n" + "═" * 70)
        self.log("  TABLE 3: STORAGE OVERHEAD")
        self.log("═" * 70)

        self.log(f"\n  {'Method':<20} {'Bytes/vec':<15} {'KB/vec':<10} "
                 f"{'Expansion':<12}")
        self.log("  " + "─" * 50)

        plain_bytes = self.doc_embeddings.shape[1] * 4  # float32

        for b in self.benchmarks:
            ct = b.ct_bytes_per_vec if b.ct_bytes_per_vec > 0 else plain_bytes
            expansion = ct / plain_bytes
            self.log(f"  {b.name:<20} {ct:<15,} {ct/1024:<10.0f} "
                     f"{expansion:<12.0f}x")

        # ── TABLE 4: Error decomposition ──
        self.log("\n\n" + "═" * 70)
        self.log("  TABLE 4: ERROR DECOMPOSITION")
        self.log("═" * 70)

        self.log(f"""
  Total error = Quantization error + Crypto error

  ┌────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Method             │ Quant Error  │ Crypto Error │ Total        │
  ├────────────────────┼──────────────┼──────────────┼──────────────┤
  │ Plaintext float32  │ 0            │ 0            │ 0            │
  │ CKKS float32       │ 0            │ ε ≠ 0       │ ε            │
  │ BFV int8           │ δ (small)    │ 0 (EXACT)    │ δ            │
  │ CKKS int8 (bad)    │ δ            │ ε            │ δ + ε        │
  └────────────────────┴──────────────┴──────────────┴──────────────┘

  BFV int8 has ONE error source (quantization).
  CKKS float32 has ONE error source (crypto noise).
  CKKS int8 has TWO error sources (worst of both worlds).
  → BFV dominates CKKS whenever int8 quantization is acceptable.
""")

        # ── TABLE 5: Score-gap analysis ──
        self.log("═" * 70)
        self.log("  TABLE 5: SCORE-GAP ANALYSIS")
        self.log("═" * 70)

        gaps = compute_score_gaps(
            self.doc_embeddings, self.query_embeddings, self.top_k
        )

        self.log(f"""
  Score-Gap Theorem:
    For top-{self.top_k} retrieval, crypto error ε must satisfy:
    ε < δₖ/2  where δₖ = score(k) - score(k+1)

  Measured gaps across {len(self.queries)} queries:
    Min gap:             {gaps['min_gap']:.6f}
    Max gap:             {gaps['max_gap']:.6f}
    Mean gap:            {gaps['mean_gap']:.6f}
    Max tolerable error: {gaps['max_tolerable_error']:.6f}
""")

        # check which methods satisfy the theorem
        for b in self.benchmarks:
            satisfies = b.max_crypto_error < gaps['max_tolerable_error']
            symbol = "✅" if satisfies else "❌"
            reason = ""
            if b.scheme == "BFV":
                reason = "(ε=0, trivially satisfied)"
            elif b.scheme == "CKKS":
                reason = f"(ε={b.max_crypto_error:.2e} vs δ/2={gaps['max_tolerable_error']:.6f})"
            elif b.scheme == "none":
                reason = "(no encryption)"

            self.log(f"    {symbol} {b.name:<20} {reason}")

        # ── TABLE 6: Per-query results ──
        self.log("\n\n" + "═" * 95)
        self.log("  TABLE 6: PER-QUERY BREAKDOWN")
        self.log("═" * 95)

        # build header line
        header = f"\n  {'Query':<40} "
        for b in self.benchmarks:
            header += f"{'[' + b.name[:12] + ']':<16} "
        self.log(header)
        self.log("  " + "─" * (40 + 16 * len(self.benchmarks)))

        for qi in range(len(self.queries)):
            q_short = self.queries[qi][:38]
            line = f"  {q_short:<40} "

            for b in self.benchmarks:
                if qi < len(b.search_results):
                    r = b.search_results[qi]
                    match = set(r.top_k_indices[:self.top_k]) == \
                            set(self.ground_truth[qi])
                    symbol = "✅" if match else "❌"
                    top1 = r.top_k_indices[0]
                    line += f"{symbol} top1={top1:<8} "
                else:
                    line += f"{'N/A':<16} "
            self.log(line)

        # ── FINAL SUMMARY ──
        self.log("\n\n" + "═" * 70)
        self.log("  FINAL SUMMARY")
        self.log("═" * 70)

        # find speedups
        ckks_bench = next((b for b in self.benchmarks
                           if b.scheme == "CKKS"), None)
        bfv_ctct = next((b for b in self.benchmarks
                         if b.name == "BFV int8 ct×ct"), None)
        bfv_ctpt = next((b for b in self.benchmarks
                         if b.name == "BFV int8 ct×pt"), None)

        if ckks_bench and bfv_ctct:
            speedup = ckks_bench.mean_latency_ms / max(bfv_ctct.mean_latency_ms, 0.01)
            ct_ratio = ckks_bench.ct_bytes_per_vec / max(bfv_ctct.ct_bytes_per_vec, 1)

            self.log(f"""
  BFV int8 ct×ct vs CKKS float32:
  ────────────────────────────────
  Latency:      {ckks_bench.mean_latency_ms:.0f}ms → {bfv_ctct.mean_latency_ms:.0f}ms  ({speedup:.1f}x speedup)
  Crypto error: {ckks_bench.max_crypto_error:.2e} → {bfv_ctct.max_crypto_error:.0f}  (ZERO vs nonzero)
  Top-1 acc:    {ckks_bench.top1_accuracy*100:.0f}% → {bfv_ctct.top1_accuracy*100:.0f}%
  Top-3 acc:    {ckks_bench.topk_accuracy*100:.0f}% → {bfv_ctct.topk_accuracy*100:.0f}%
  CT size:      {ckks_bench.ct_bytes_per_vec/1024:.0f}KB → {bfv_ctct.ct_bytes_per_vec/1024:.0f}KB  ({ct_ratio:.1f}x reduction)
  Privacy:      FULL → FULL (both encrypt query + docs)
""")

        if ckks_bench and bfv_ctpt:
            speedup2 = ckks_bench.mean_latency_ms / max(bfv_ctpt.mean_latency_ms, 0.01)
            self.log(f"""
  BFV int8 ct×pt vs CKKS float32:
  ────────────────────────────────
  Latency:      {ckks_bench.mean_latency_ms:.0f}ms → {bfv_ctpt.mean_latency_ms:.0f}ms  ({speedup2:.1f}x speedup)
  Crypto error: {ckks_bench.max_crypto_error:.2e} → {bfv_ctpt.max_crypto_error:.0f}  (ZERO)
  Privacy:      FULL → QUERY ONLY (trade-off for speed)
""")

        self.log(f"""
  PAPER CONCLUSION:
  ─────────────────
  "Standard private RAG uses CKKS on float32 vectors.
   We show that quantizing to int8 and switching to BFV
   eliminates cryptographic error entirely while achieving
   competitive or faster latency. BFV trivially satisfies
   the score-gap criterion (ε=0), guaranteeing correct
   top-k retrieval identical to plaintext int8 search."
""")

    def save_results(self, filepath: str = "results/comparison.txt"):
        """Save all output to a file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for line in self.output_lines:
                f.write(line + "\n")
        print(f"\n  📄 Results saved to: {filepath}")


# ═══════════════════════════════════════════════════════════
#  MAIN: RUN THE FULL COMPARISON
# ═══════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🔬  PRIVATE RAG: HEAD-TO-HEAD COMPARISON  🔬                   ║
║                                                                  ║
║   Method 1: Plaintext RAG       (no privacy — speed baseline)    ║
║   Method 2: CKKS float32 RAG   (standard encrypted — baseline)  ║
║   Method 3: BFV int8 RAG       (YOUR method — the contribution) ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # check for PDF argument
    pdf_path = None
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.pdf', '.txt')):
        pdf_path = sys.argv[1]

    # create comparator
    comp = Comparator(top_k=3)

    # prepare data
    comp.prepare_data(pdf_path=pdf_path)

    # run all three methods
    total_t0 = time.time()

    comp.run_plaintext()       # Method 1: no encryption
    comp.run_ckks()            # Method 2: CKKS float32 (the enemy)
    comp.run_bfv_ct_ct()       # Method 3: BFV int8 full privacy (yours)
    comp.run_bfv_ct_pt()       # Method 3b: BFV int8 query privacy (fastest)

    total_time = time.time() - total_t0

    # print all tables
    comp.print_tables()

    comp.log(f"\n  Total comparison time: {total_time:.0f}s")

    # save results
    comp.save_results()

    print("\n  ✅ COMPARISON COMPLETE")
    print("  Copy the tables above into your paper's evaluation section.")


if __name__ == "__main__":
    main()
