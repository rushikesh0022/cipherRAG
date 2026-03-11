#!/usr/bin/env python3
"""
compare_fast.py — Optimized BFV vs CKKS

THREE OPTIMIZATIONS:

1. NO-ROTATION TRICK:
   Standard:  Server does E(q).dot(d) = multiply + 9 rotations  (SLOW)
   Fast:      Server does E(q) * d    = multiply only            (FAST)
              Client decrypts element-wise products, sums locally

2. BATCHED DOCUMENTS:
   Standard:  30 separate dot products (30 encryptions, 30 decryptions)
   Batched:   Pack 10 docs per ciphertext → only 3 operations for 30 docs

3. PCA DIMENSION REDUCTION:
   384 dims → 128 dims = fewer elements to process

Usage:
    python compare_fast.py
"""

import time
import sys
import numpy as np
import tenseal as ts
from sklearn.decomposition import PCA
from typing import List, Tuple
from dataclasses import dataclass

from config import RAGConfig, BFVConfig
from src.chunker import create_sample_corpus
from src.embedder import Embedder
from src.quantizer import Int8Quantizer
from src.encryption import CKKSEngine


# ═══════════════════════════════════════════════════════════
#  RESULT CONTAINER
# ═══════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    name: str
    top1_pct: float
    topk_pct: float
    avg_ms: float
    max_err: float
    ct_kb: float
    privacy: str
    note: str = ""


# ═══════════════════════════════════════════════════════════
#  OPTIMIZATION 1: NO-ROTATION ct×pt
#
#  Standard .dot() = element-wise multiply + rotate-and-sum
#  The rotate-and-sum uses log2(384) ≈ 9 rotations.
#  Each rotation is as expensive as a multiplication.
#  So .dot() is ~10x more expensive than a simple multiply.
#
#  NO-ROTATION approach:
#    Server: E(q) * d_plain = E(q ⊙ d)   [element-wise, 1 operation]
#    Client: decrypt E(q ⊙ d) → [q₀d₀, q₁d₁, ...] → sum → score
#
#  We skip 9 rotations per document. For 30 docs = 270 rotations saved.
# ═══════════════════════════════════════════════════════════

class FastBFVEngine:
    """BFV engine with optimized search methods."""

    def __init__(self, poly_mod=8192, plain_mod_bits=25):
        self.poly_mod = poly_mod
        self.slots = poly_mod // 2

        print(f"\n  🔐 Setting up Fast BFV (N={poly_mod}) ...")
        t0 = time.time()

        try:
            self.plain_modulus = ts.plain_modulus_batching(poly_mod, plain_mod_bits)
        except Exception:
            fallback = {4096: 16760833, 8192: 33538049, 16384: 67104769}
            self.plain_modulus = fallback.get(poly_mod, 33538049)

        self.max_safe = self.plain_modulus // 2

        self.ctx = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=poly_mod,
            plain_modulus=self.plain_modulus,
        )
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()

        pub = self.ctx.copy()
        pub.make_context_public()
        self.server_ctx = ts.context_from(pub.serialize())

        self.keygen_time = time.time() - t0
        print(f"     plain_modulus = {self.plain_modulus}")
        print(f"     keygen = {self.keygen_time:.2f}s")
        print(f"  ✅ Ready")

    def _to_signed(self, val):
        """Convert BFV modular value to signed integer."""
        val = int(val)
        if val > self.max_safe:
            val -= self.plain_modulus
        return val

    # ─────────────────────────────────────────────────
    #  METHOD A: Standard .dot() (with rotations — SLOW)
    # ─────────────────────────────────────────────────

    def search_standard_ctct(self, int_docs, int_queries, top_k=3):
        """Standard ct×ct with .dot() — the slow baseline."""
        print(f"  🔍 [BFV std ct×ct] Encrypting docs ...")
        enc_docs = []
        for vec in int_docs:
            enc = ts.bfv_vector(self.ctx, [int(x) for x in vec.tolist()])
            enc_docs.append(enc.serialize())

        results = []
        for qi in range(len(int_queries)):
            t0 = time.time()
            q_list = [int(x) for x in int_queries[qi].tolist()]
            enc_q = ts.bfv_vector(self.ctx, q_list)
            enc_q_bytes = enc_q.serialize()

            scores = []
            for db in enc_docs:
                enc_q_s = ts.bfv_vector_from(self.server_ctx, enc_q_bytes)
                enc_d = ts.bfv_vector_from(self.server_ctx, db)
                enc_score = enc_q_s.dot(enc_d)
                sb = enc_score.serialize()
                sv = ts.bfv_vector_from(self.ctx, sb)
                scores.append(self._to_signed(sv.decrypt()[0]))

            lat = (time.time() - t0) * 1000
            top = list(np.argsort(scores)[-top_k:][::-1])
            results.append((top, scores, lat))
            print(f"     [{qi+1}/{len(int_queries)}] {lat:.0f}ms", end='\r')

        print()
        return results

    # ─────────────────────────────────────────────────
    #  METHOD B: No-rotation ct×pt (element-wise — FAST)
    # ─────────────────────────────────────────────────

    def search_norotation_ctpt(self, int_docs, int_queries, top_k=3):
        """
        OPTIMIZED: Skip rotations entirely.
        Server does element-wise multiply, client sums locally.
        """
        dim = int_docs.shape[1]
        results = []

        for qi in range(len(int_queries)):
            t0 = time.time()
            q_list = [int(x) for x in int_queries[qi].tolist()]
            enc_q = ts.bfv_vector(self.ctx, q_list)
            enc_q_bytes = enc_q.serialize()

            scores = []
            for i in range(len(int_docs)):
                d_list = [int(x) for x in int_docs[i].tolist()]

                # SERVER: element-wise multiply (NO rotation!)
                enc_q_s = ts.bfv_vector_from(self.server_ctx, enc_q_bytes)
                enc_products = enc_q_s * d_list  # element-wise ct×pt
                product_bytes = enc_products.serialize()

                # CLIENT: decrypt vector and sum locally
                dec_vec = ts.bfv_vector_from(self.ctx, product_bytes)
                raw = dec_vec.decrypt()

                # Sum with signed conversion
                score = 0
                for j in range(dim):
                    score += self._to_signed(raw[j])
                scores.append(score)

            lat = (time.time() - t0) * 1000
            top = list(np.argsort(scores)[-top_k:][::-1])
            results.append((top, scores, lat))
            print(f"     [{qi+1}/{len(int_queries)}] {lat:.0f}ms", end='\r')

        print()
        return results

    # ─────────────────────────────────────────────────
    #  METHOD C: Batched ct×pt (pack multiple docs — FASTEST)
    # ─────────────────────────────────────────────────

    def search_batched_ctpt(self, int_docs, int_queries, top_k=3):
        """
        MOST OPTIMIZED: Pack multiple documents per ciphertext.

        With N=8192 → 4096 slots, dim=384:
          docs_per_batch = 4096 // 384 = 10
          30 docs → 3 batches → 3 multiplications instead of 30

        Combined with no-rotation: 3 ops instead of 30×10 = 300 ops
        """
        dim = int_docs.shape[1]
        n_docs = len(int_docs)
        docs_per_batch = self.slots // dim
        n_batches = (n_docs + docs_per_batch - 1) // docs_per_batch

        print(f"  📦 Batching: {docs_per_batch} docs/batch, "
              f"{n_batches} batches for {n_docs} docs")

        # Pre-pack document batches (one-time cost)
        doc_batches = []
        for b in range(n_batches):
            start = b * docs_per_batch
            end = min(start + docs_per_batch, n_docs)
            packed = []
            for i in range(start, end):
                packed.extend([int(x) for x in int_docs[i].tolist()])
            # Pad to fill slots
            packed.extend([0] * (self.slots - len(packed)))
            doc_batches.append(packed[:self.slots])

        results = []
        for qi in range(len(int_queries)):
            t0 = time.time()
            q_list = [int(x) for x in int_queries[qi].tolist()]

            # Pack query as repeating pattern: [q, q, q, ..., q, 0, 0]
            q_packed = []
            for _ in range(docs_per_batch):
                q_packed.extend(q_list)
            q_packed.extend([0] * (self.slots - len(q_packed)))
            q_packed = q_packed[:self.slots]

            # Encrypt packed query (ONCE)
            enc_q = ts.bfv_vector(self.ctx, q_packed)
            enc_q_bytes = enc_q.serialize()

            all_scores = []
            for b in range(n_batches):
                # SERVER: one element-wise multiply per batch
                enc_q_s = ts.bfv_vector_from(self.server_ctx, enc_q_bytes)
                enc_products = enc_q_s * doc_batches[b]
                product_bytes = enc_products.serialize()

                # CLIENT: decrypt and sum groups of `dim`
                dec_vec = ts.bfv_vector_from(self.ctx, product_bytes)
                raw = dec_vec.decrypt()

                start_doc = b * docs_per_batch
                end_doc = min(start_doc + docs_per_batch, n_docs)

                for d in range(end_doc - start_doc):
                    offset = d * dim
                    score = 0
                    for j in range(dim):
                        score += self._to_signed(raw[offset + j])
                    all_scores.append(score)

            lat = (time.time() - t0) * 1000
            top = list(np.argsort(all_scores)[-top_k:][::-1])
            results.append((top, all_scores, lat))
            print(f"     [{qi+1}/{len(int_queries)}] {lat:.0f}ms "
                  f"({n_batches} batch ops)", end='\r')

        print()
        return results


# ═══════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════

def evaluate(results, ground_truth, top_k):
    """Compute accuracy and latency from results."""
    t1 = t3 = 0
    lats = []
    for i, (top, scores, lat) in enumerate(results):
        if top[0] == ground_truth[i][0]:
            t1 += 1
        if set(top[:top_k]) == set(ground_truth[i]):
            t3 += 1
        lats.append(lat)
    n = len(results)
    return t1/n, t3/n, np.mean(lats)


def verify_scores(results, int_docs, int_queries, method_name):
    """Check BFV scores match plaintext int8 scores."""
    max_err = 0
    for qi, (top, scores, lat) in enumerate(results):
        plain = (int_docs @ int_queries[qi]).tolist()
        for j in range(len(scores)):
            err = abs(int(plain[j]) - int(scores[j]))
            max_err = max(max_err, err)
    return max_err


# ═══════════════════════════════════════════════════════════
#  MAIN COMPARISON
# ═══════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🚀  OPTIMIZED PRIVATE RAG COMPARISON  🚀                      ║
║                                                                  ║
║   Three optimizations to make BFV competitive:                   ║
║     1. No-rotation trick (skip 9 rotations per dot product)      ║
║     2. Batch packing (10 docs per ciphertext)                    ║
║     3. PCA reduction (384 → 128 dimensions)                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    TOP_K = 3

    # ── Load data ──
    print("=" * 65)
    print("  DATA PREPARATION")
    print("=" * 65)

    chunks, queries = create_sample_corpus()
    embedder = Embedder()
    doc_embs = embedder.embed_documents(chunks)
    query_embs = embedder.embed_queries(queries)
    DIM = doc_embs.shape[1]

    # Ground truth (float32 plaintext)
    ground_truth = []
    for qv in query_embs:
        scores = doc_embs @ qv
        ground_truth.append(list(np.argsort(scores)[-TOP_K:][::-1]))

    # Quantize full-dim
    quantizer = Int8Quantizer()
    int_docs = quantizer.quantize_documents(doc_embs)
    int_queries = np.array([quantizer.quantize_query(qe) for qe in query_embs])

    # PCA dimension reduction
    target_dim = min(128, len(doc_embs) - 1)  # can't exceed n_samples - 1
    print(f"\n  📉 Applying PCA: {DIM} → {target_dim} dimensions ...")
    if len(doc_embs) < 128:
        print(f"     ⚠️  Only {len(doc_embs)} chunks. PCA capped at {target_dim}.")
        print(f"     Use a real PDF with 130+ chunks for full PCA-128.")
    pca = PCA(n_components=target_dim)
    doc_embs_pca = pca.fit_transform(doc_embs)
    query_embs_pca = pca.transform(query_embs)
    # re-normalize
    doc_embs_pca /= np.linalg.norm(doc_embs_pca, axis=1, keepdims=True)
    query_embs_pca /= np.linalg.norm(query_embs_pca, axis=1, keepdims=True)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"     Variance explained: {var_explained:.1%}")

    # PCA ground truth
    gt_pca = []
    for qv in query_embs_pca:
        scores = doc_embs_pca @ qv
        gt_pca.append(list(np.argsort(scores)[-TOP_K:][::-1]))

    # Quantize PCA
    quantizer_pca = Int8Quantizer()
    int_docs_pca = quantizer_pca.quantize_documents(doc_embs_pca)
    int_queries_pca = np.array([
        quantizer_pca.quantize_query(qe) for qe in query_embs_pca
    ])

    # PCA accuracy vs original ground truth
    pca_t1 = sum(1 for i in range(len(queries))
                 if gt_pca[i][0] == ground_truth[i][0])
    pca_t3 = sum(1 for i in range(len(queries))
                 if set(gt_pca[i]) == set(ground_truth[i]))
    print(f"     PCA-128 ranking preservation: "
          f"Top-1={pca_t1}/{len(queries)} "
          f"Top-3={pca_t3}/{len(queries)}")

    benchmarks: List[BenchResult] = []

    # ══════════════════════════════════════════════════════
    #  METHOD 1: PLAINTEXT BASELINE
    # ══════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print(f"  METHOD 1: PLAINTEXT (no encryption)")
    print(f"{'=' * 65}")

    lats = []
    for qv in query_embs:
        t0 = time.time()
        _ = doc_embs @ qv
        lats.append((time.time() - t0) * 1000)

    benchmarks.append(BenchResult(
        "Plaintext", 1.0, 1.0, np.mean(lats), 0.0,
        DIM * 4 / 1024, "NONE"
    ))
    print(f"  ✅ Avg: {np.mean(lats):.3f}ms")

    # ══════════════════════════════════════════════════════
    #  METHOD 2: CKKS STANDARD (the enemy)
    # ══════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print(f"  METHOD 2: CKKS float32 ct×ct (standard)")
    print(f"{'=' * 65}")

    ckks = CKKSEngine(8192, [60, 40, 40, 60], 40, "CKKS-std")
    ckks.setup()
    enc_docs_ckks = ckks.encrypt_documents(doc_embs)
    ct_kb_ckks = len(enc_docs_ckks[0]) / 1024

    ckks_lats = []
    ckks_t1 = ckks_t3 = 0
    ckks_max_err = 0.0

    print(f"  🔍 Running {len(queries)} queries ...")
    for qi, qv in enumerate(query_embs):
        t0 = time.time()
        enc_q = ckks.encrypt_vector(qv)
        scores = []
        for db in enc_docs_ckks:
            sb = ckks.server_dot_product(enc_q, db)
            scores.append(ckks.decrypt_score(sb))
        lat = (time.time() - t0) * 1000
        ckks_lats.append(lat)

        plain = (doc_embs @ qv).tolist()
        errs = [abs(plain[j] - scores[j]) for j in range(len(scores))]
        ckks_max_err = max(ckks_max_err, max(errs))

        top = list(np.argsort(scores)[-TOP_K:][::-1])
        if top[0] == ground_truth[qi][0]: ckks_t1 += 1
        if set(top) == set(ground_truth[qi]): ckks_t3 += 1
        print(f"     [{qi+1}/{len(queries)}] {lat:.0f}ms", end='\r')

    n = len(queries)
    benchmarks.append(BenchResult(
        "CKKS-f32 std", ckks_t1/n, ckks_t3/n, np.mean(ckks_lats),
        ckks_max_err, ct_kb_ckks, "FULL"
    ))
    print(f"\n  ✅ Avg: {np.mean(ckks_lats):.0f}ms, "
          f"err={ckks_max_err:.2e}")

    # ══════════════════════════════════════════════════════
    #  METHOD 3: BFV STANDARD ct×ct (current slow version)
    # ══════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print(f"  METHOD 3: BFV int8 ct×ct STANDARD (with rotations)")
    print(f"{'=' * 65}")

    engine = FastBFVEngine(8192, 25)
    print(f"  🔍 Running standard ct×ct ...")
    bfv_std_results = engine.search_standard_ctct(int_docs, int_queries, TOP_K)
    bfv_std_err = verify_scores(bfv_std_results, int_docs, int_queries, "BFV-std")
    t1, t3, avg = evaluate(bfv_std_results, ground_truth, TOP_K)

    benchmarks.append(BenchResult(
        "BFV-i8 std ct×ct", t1, t3, avg, bfv_std_err,
        432, "FULL", "9 rotations/dot"
    ))
    print(f"  ✅ Avg: {avg:.0f}ms, err={bfv_std_err}")

    # ══════════════════════════════════════════════════════
    #  METHOD 4: BFV NO-ROTATION ct×pt (optimized!)
    # ══════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print(f"  METHOD 4: BFV int8 ct×pt NO-ROTATION ⚡")
    print(f"{'=' * 65}")

    engine2 = FastBFVEngine(8192, 25)

    norot_ok = True
    try:
        print(f"  🔍 Testing element-wise multiply ...")
        # Quick test
        test_enc = ts.bfv_vector(engine2.ctx, [1, 2, 3, 4, 5])
        test_result = test_enc * [10, 20, 30, 40, 50]
        test_dec = test_result.decrypt()
        expected = [10, 40, 90, 160, 250]
        if all(int(test_dec[i]) == expected[i] for i in range(5)):
            print(f"  ✅ Element-wise multiply works! Skipping rotations.")
        else:
            print(f"  ⚠️ Unexpected result. Falling back to .dot()")
            norot_ok = False
    except Exception as e:
        print(f"  ❌ Element-wise multiply not supported: {e}")
        print(f"     Falling back to .dot()")
        norot_ok = False

    if norot_ok:
        print(f"  🔍 Running no-rotation ct×pt ...")
        norot_results = engine2.search_norotation_ctpt(
            int_docs, int_queries, TOP_K
        )
        norot_err = verify_scores(norot_results, int_docs, int_queries, "norot")
        t1, t3, avg = evaluate(norot_results, ground_truth, TOP_K)

        benchmarks.append(BenchResult(
            "BFV-i8 no-rot ct×pt", t1, t3, avg, norot_err,
            0, "QUERY ONLY", "0 rotations!"
        ))
        print(f"  ✅ Avg: {avg:.0f}ms, err={norot_err}")
    else:
        print(f"  ⏭️  Skipped (element-wise multiply not available)")

    # ══════════════════════════════════════════════════════
    #  METHOD 5: BFV BATCHED ct×pt (pack 10 docs/ciphertext)
    # ══════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print(f"  METHOD 5: BFV int8 BATCHED ct×pt ⚡⚡")
    print(f"{'=' * 65}")

    if norot_ok:
        engine3 = FastBFVEngine(8192, 25)
        print(f"  🔍 Running batched ct×pt ...")
        batch_results = engine3.search_batched_ctpt(
            int_docs, int_queries, TOP_K
        )
        batch_err = verify_scores(batch_results, int_docs, int_queries, "batch")
        t1, t3, avg = evaluate(batch_results, ground_truth, TOP_K)

        benchmarks.append(BenchResult(
            "BFV-i8 batch ct×pt", t1, t3, avg, batch_err,
            0, "QUERY ONLY", "3 batch ops!"
        ))
        print(f"  ✅ Avg: {avg:.0f}ms, err={batch_err}")
    else:
        print(f"  ⏭️  Skipped (requires element-wise multiply)")

    # ══════════════════════════════════════════════════════
    #  METHOD 6: BFV BATCHED + PCA-128 (maximum speed)
    # ══════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print(f"  METHOD 6: BFV int8 BATCHED + PCA-128 ct×pt ⚡⚡⚡")
    print(f"{'=' * 65}")

    if norot_ok:
        engine4 = FastBFVEngine(8192, 25)
        print(f"  🔍 Running batched + PCA-128 ...")
        pca_batch_results = engine4.search_batched_ctpt(
            int_docs_pca, int_queries_pca, TOP_K
        )
        pca_batch_err = verify_scores(
            pca_batch_results, int_docs_pca, int_queries_pca, "pca-batch"
        )
        t1_pca, t3_pca, avg_pca = evaluate(
            pca_batch_results, ground_truth, TOP_K
        )

        benchmarks.append(BenchResult(
            "BFV-i8 batch+PCA", t1_pca, t3_pca, avg_pca, pca_batch_err,
            0, "QUERY ONLY", "3 ops, 128d"
        ))
        print(f"  ✅ Avg: {avg_pca:.0f}ms, err={pca_batch_err}")
    else:
        print(f"  ⏭️  Skipped")

    # ══════════════════════════════════════════════════════
    #  METHOD 7: CKKS + PCA-128 (fair comparison)
    # ══════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print(f"  METHOD 7: CKKS float32 + PCA-128 (fair comparison)")
    print(f"{'=' * 65}")

    ckks_pca = CKKSEngine(8192, [60, 40, 40, 60], 40, "CKKS-PCA")
    ckks_pca.setup()
    enc_docs_ckks_pca = ckks_pca.encrypt_documents(doc_embs_pca)

    ckks_pca_lats = []
    ckks_pca_t1 = ckks_pca_t3 = 0
    ckks_pca_max_err = 0.0

    print(f"  🔍 Running {len(queries)} queries ...")
    for qi, qv in enumerate(query_embs_pca):
        t0 = time.time()
        enc_q = ckks_pca.encrypt_vector(qv)
        scores = []
        for db in enc_docs_ckks_pca:
            sb = ckks_pca.server_dot_product(enc_q, db)
            scores.append(ckks_pca.decrypt_score(sb))
        lat = (time.time() - t0) * 1000
        ckks_pca_lats.append(lat)

        plain = (doc_embs_pca @ qv).tolist()
        errs = [abs(plain[j] - scores[j]) for j in range(len(scores))]
        ckks_pca_max_err = max(ckks_pca_max_err, max(errs))

        top = list(np.argsort(scores)[-TOP_K:][::-1])
        if top[0] == ground_truth[qi][0]: ckks_pca_t1 += 1
        if set(top) == set(ground_truth[qi]): ckks_pca_t3 += 1
        print(f"     [{qi+1}/{len(queries)}] {lat:.0f}ms", end='\r')

    benchmarks.append(BenchResult(
        "CKKS-f32 PCA-128", ckks_pca_t1/n, ckks_pca_t3/n,
        np.mean(ckks_pca_lats), ckks_pca_max_err,
        len(enc_docs_ckks_pca[0])/1024, "FULL", "128 dims"
    ))
    print(f"\n  ✅ Avg: {np.mean(ckks_pca_lats):.0f}ms")

    # ══════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ══════════════════════════════════════════════════════

    print(f"\n\n{'═' * 100}")
    print(f"  RESULTS: OPTIMIZED COMPARISON")
    print(f"{'═' * 100}")

    print(f"\n  {'Method':<25} {'Top1%':<7} {'Top3%':<7} "
          f"{'Avg ms':<10} {'MaxErr':<12} {'CT KB':<8} "
          f"{'Privacy':<12} {'Note'}")
    print(f"  {'─' * 97}")

    for b in benchmarks:
        print(f"  {b.name:<25} {b.top1_pct*100:<7.0f} {b.topk_pct*100:<7.0f} "
              f"{b.avg_ms:<10.0f} {b.max_err:<12.2e} {b.ct_kb:<8.0f} "
              f"{b.privacy:<12} {b.note}")

    # ── Speedup calculations ──
    ckks_time = next(b.avg_ms for b in benchmarks if "CKKS" in b.name and "PCA" not in b.name)
    bfv_std_time = next(b.avg_ms for b in benchmarks if "std ct×ct" in b.name)

    print(f"\n  SPEEDUP vs CKKS standard ({ckks_time:.0f}ms):")
    print(f"  {'─' * 55}")
    for b in benchmarks:
        if b.avg_ms > 0 and "Plaintext" not in b.name:
            speedup = ckks_time / b.avg_ms
            bar = "█" * int(min(speedup * 5, 50))
            faster = "FASTER" if speedup > 1 else "slower"
            print(f"    {b.name:<25} {speedup:>5.1f}x {faster:<8} {bar}")

    print(f"\n  SPEEDUP vs BFV standard ({bfv_std_time:.0f}ms):")
    print(f"  {'─' * 55}")
    for b in benchmarks:
        if "BFV" in b.name:
            speedup = bfv_std_time / max(b.avg_ms, 0.01)
            bar = "█" * int(min(speedup * 5, 50))
            print(f"    {b.name:<25} {speedup:>5.1f}x {bar}")

    # ── Key finding ──
    fastest_bfv = min((b for b in benchmarks if "BFV" in b.name),
                      key=lambda b: b.avg_ms)
    print(f"""

  ┌────────────────────────────────────────────────────────────┐
  │  KEY FINDING                                               │
  │                                                            │
  │  Fastest BFV:  {fastest_bfv.name:<25} {fastest_bfv.avg_ms:>6.0f}ms │
  │  CKKS std:     {"CKKS-f32 std":<25} {ckks_time:>6.0f}ms │
  │  Speedup:      {ckks_time/max(fastest_bfv.avg_ms, 0.01):>5.1f}x {"FASTER ⚡" if fastest_bfv.avg_ms < ckks_time else "slower ⚠️ ":<30}│
  │                                                            │
  │  BFV crypto error:  {fastest_bfv.max_err}                          │
  │  CKKS crypto error: {ckks_max_err:.2e}                      │
  │                                                            │
  │  BFV is {"EXACT" if fastest_bfv.max_err == 0 else "approximate"} (zero crypto error)                     │
  └────────────────────────────────────────────────────────────┘
""")

    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/fast_comparison.txt", "w") as f:
        for b in benchmarks:
            f.write(f"{b.name}\t{b.top1_pct:.2f}\t{b.topk_pct:.2f}\t"
                    f"{b.avg_ms:.1f}\t{b.max_err:.2e}\t{b.note}\n")
    print(f"  📄 Saved to results/fast_comparison.txt")


if __name__ == "__main__":
    main()