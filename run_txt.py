#!/usr/bin/env python3
"""
run_txt.py — Run cipherRAG on your own .txt files

Usage:
    python3 run_txt.py data/my_document.txt data/my_queries.txt
    python3 run_txt.py data/my_document.txt
    python3 run_txt.py data/my_document.txt data/my_queries.txt --interactive
"""

import sys
import os
import time
import numpy as np

from config import RAGConfig, BFVConfig
from src.chunker import chunk_text
from src.embedder import Embedder
from src.quantizer import Int8Quantizer
from src.encryption import BFVEngine, CKKSEngine
from src.search import PlaintextSearch, CKKSSearch, BFVSearch


def load_txt_document(filepath):
    """Load a text file and split into chunks."""
    print(f"\n  📄 Loading: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"     Characters: {len(text):,}")
    print(f"     Words: {len(text.split()):,}")

    # Split into chunks
    config = RAGConfig().chunk
    chunks = chunk_text(text, config)

    # If chunks are too few (short document), split by paragraphs instead
    if len(chunks) < 5:
        print(f"     ⚠️  Only {len(chunks)} chunks with word-splitting.")
        print(f"     Trying paragraph splitting instead ...")
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        # filter out very short paragraphs
        paragraphs = [p for p in paragraphs if len(p) > 50]
        if len(paragraphs) > len(chunks):
            chunks = paragraphs
            print(f"     ✅ Got {len(chunks)} paragraph chunks")

    print(f"  ✅ Total chunks: {len(chunks)}")

    # Show first few
    print(f"\n  First 3 chunks:")
    for i, c in enumerate(chunks[:3]):
        preview = c[:80].replace('\n', ' ')
        print(f"    [{i}] \"{preview}...\"")

    return chunks


def load_queries(filepath=None):
    """Load queries from a file (one per line) or use defaults."""
    if filepath and os.path.exists(filepath):
        print(f"\n  ❓ Loading queries from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        print(f"     Loaded {len(queries)} queries")
    else:
        print(f"\n  ❓ Using default queries")
        queries = [
            "What is this document about?",
            "What are the main topics covered?",
            "What are the key findings?",
            "What problems are discussed?",
            "What solutions are proposed?",
        ]

    for i, q in enumerate(queries):
        print(f"    [{i}] \"{q}\"")

    return queries


def run_full_comparison(chunks, queries, relevance_threshold=None):
    """Run Plaintext vs CKKS vs BFV on your data."""

    rag_config = RAGConfig()
    if relevance_threshold is not None:
        rag_config.relevance_threshold = relevance_threshold
    TOP_K = 3

    # ── Embeddings ──
    print(f"\n{'═' * 65}")
    print(f"  STEP 1: EMBEDDINGS")
    print(f"{'═' * 65}")

    embedder = Embedder()
    doc_embs = embedder.embed_documents(chunks)
    query_embs = embedder.embed_queries(queries)
    DIM = doc_embs.shape[1]

    # ── Ground truth ──
    ground_truth = []
    for qv in query_embs:
        scores = doc_embs @ qv
        top = list(np.argsort(scores)[-TOP_K:][::-1])
        ground_truth.append(top)

    # ── Quantize ──
    print(f"\n{'═' * 65}")
    print(f"  STEP 2: INT8 QUANTIZATION")
    print(f"{'═' * 65}")

    quantizer = Int8Quantizer()
    int_docs = quantizer.quantize_documents(doc_embs)
    int_queries = np.array([
        quantizer.quantize_query(qe) for qe in query_embs
    ])
    quantizer.verify_ranking(doc_embs, query_embs, TOP_K)

    # ── Score gap analysis ──
    from src.utils import compute_score_gaps
    gaps = compute_score_gaps(doc_embs, query_embs, TOP_K)

    print(f"\n  Score-Gap Analysis:")
    print(f"    Min gap:  {gaps['min_gap']:.6f}")
    print(f"    Mean gap: {gaps['mean_gap']:.6f}")
    print(f"    Max tolerable CKKS error: {gaps['max_tolerable_error']:.6f}")

    # ══════════════════════════════════════════════════════
    #  METHOD 1: PLAINTEXT
    # ══════════════════════════════════════════════════════
    print(f"\n{'═' * 65}")
    print(f"  METHOD 1: PLAINTEXT (no encryption)")
    print(f"{'═' * 65}")

    plain = PlaintextSearch()
    plain.index_documents(doc_embs)
    plain_results = plain.search_batch(query_embs, queries, TOP_K)
    plain_lats = [r.latency_ms for r in plain_results]
    print(f"  ✅ Avg: {np.mean(plain_lats):.3f}ms")

    # ══════════════════════════════════════════════════════
    #  METHOD 2: CKKS
    # ══════════════════════════════════════════════════════
    print(f"\n{'═' * 65}")
    print(f"  METHOD 2: CKKS float32 (standard encrypted RAG)")
    print(f"{'═' * 65}")

    ckks = CKKSEngine(8192, [60, 40, 40, 60], 40)
    ckks.setup()
    ckks_search = CKKSSearch(ckks)
    ckks_search.index_documents(doc_embs)
    ckks_results = ckks_search.search_batch(
        query_embs, doc_embs, queries, TOP_K
    )
    ckks_lats = [r.latency_ms for r in ckks_results]
    ckks_errs = [r.crypto_error for r in ckks_results]

    # ══════════════════════════════════════════════════════
    #  METHOD 3: BFV (YOUR METHOD)
    # ══════════════════════════════════════════════════════
    print(f"\n{'═' * 65}")
    print(f"  METHOD 3: BFV int8 (YOUR method — cipherRAG)")
    print(f"{'═' * 65}")

    bfv_engine = BFVEngine(BFVConfig())
    bfv_engine.setup()
    bfv_search = BFVSearch(bfv_engine)
    bfv_search.index_documents(int_docs, mode="ct_ct")
    bfv_results = bfv_search.search_batch(
        int_queries, int_docs, queries, mode="ct_ct", top_k=TOP_K
    )
    bfv_lats = [r.latency_ms for r in bfv_results]
    bfv_errs = [r.crypto_error for r in bfv_results]

    # ══════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ══════════════════════════════════════════════════════

    def get_accuracy(results, gt, k):
        n = len(results)
        t1 = sum(1 for i, r in enumerate(results)
                 if r.top_k_indices[0] == gt[i][0])
        tk = sum(1 for i, r in enumerate(results)
                 if set(r.top_k_indices[:k]) == set(gt[i]))
        return t1 / n * 100, tk / n * 100

    p1, pk = get_accuracy(plain_results, ground_truth, TOP_K)
    c1, ck = get_accuracy(ckks_results, ground_truth, TOP_K)
    b1, bk = get_accuracy(bfv_results, ground_truth, TOP_K)

    print(f"\n\n{'═' * 85}")
    print(f"  COMPARISON TABLE")
    print(f"{'═' * 85}")
    print(f"\n  {'Method':<22} {'Top1%':<8} {'Top3%':<8} "
          f"{'Avg(ms)':<10} {'MaxErr':<14} {'Privacy'}")
    print(f"  {'─' * 75}")
    print(f"  {'Plaintext':<22} {p1:<8.0f} {pk:<8.0f} "
          f"{np.mean(plain_lats):<10.3f} {'0':<14} {'NONE'}")
    print(f"  {'CKKS float32':<22} {c1:<8.0f} {ck:<8.0f} "
          f"{np.mean(ckks_lats):<10.0f} {max(ckks_errs):<14.2e} {'FULL'}")
    print(f"  {'BFV int8 (OURS)':<22} {b1:<8.0f} {bk:<8.0f} "
          f"{np.mean(bfv_lats):<10.0f} {max(bfv_errs):<14.0f} {'FULL'}")

    # ══════════════════════════════════════════════════════
    #  PER-QUERY RESULTS
    # ══════════════════════════════════════════════════════

    print(f"\n\n{'═' * 85}")
    print(f"  PER-QUERY RESULTS")
    print(f"{'═' * 85}")

    for qi in range(len(queries)):
        print(f"\n  ❓ Query: \"{queries[qi]}\"")

        # plaintext result
        p_idx = plain_results[qi].top_k_indices[0]
        c_idx = ckks_results[qi].top_k_indices[0]
        b_idx = bfv_results[qi].top_k_indices[0]

        print(f"\n     PLAINTEXT → Chunk {p_idx}:")
        text = chunks[p_idx][:150].replace('\n', ' ')
        print(f"       \"{text}...\"")

        match_c = "✅ MATCH" if c_idx == p_idx else f"❌ Got chunk {c_idx}"
        match_b = "✅ MATCH" if b_idx == p_idx else f"❌ Got chunk {b_idx}"

        print(f"     CKKS      → {match_c}  "
              f"(crypto error: {ckks_results[qi].crypto_error:.2e})")
        print(f"     BFV       → {match_b}  "
              f"(crypto error: {bfv_results[qi].crypto_error})")

        # show all top-3
        print(f"\n     Top-3 comparison:")
        p_top3 = plain_results[qi].top_k_indices[:TOP_K]
        c_top3 = ckks_results[qi].top_k_indices[:TOP_K]
        b_top3 = bfv_results[qi].top_k_indices[:TOP_K]
        print(f"       Plaintext: {p_top3}")
        print(f"       CKKS:      {c_top3}  "
              f"{'✅' if set(c_top3) == set(p_top3) else '❌'}")
        print(f"       BFV:       {b_top3}  "
              f"{'✅' if set(b_top3) == set(p_top3) else '❌'}")

    # ══════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════

    print(f"\n\n{'═' * 65}")
    print(f"  SUMMARY")
    print(f"{'═' * 65}")

    print(f"""
  Document: {len(chunks)} chunks, {DIM}-dimensional embeddings
  Queries:  {len(queries)} test queries
  
  Score-Gap Theorem:
    Max tolerable error: {gaps['max_tolerable_error']:.6f}
    CKKS actual error:   {max(ckks_errs):.2e}  {'✅ within bounds' if max(ckks_errs) < gaps['max_tolerable_error'] else '❌ exceeds bounds'}
    BFV actual error:    {max(bfv_errs)}  ✅ trivially satisfied (exact)
  
  BFV crypto error is ZERO — ranking identical to plaintext int8.
  Any accuracy difference is from quantization, NOT encryption.
    Relevance threshold in use: {rag_config.relevance_threshold:.4f}
""")

    return {
        'chunks': chunks,
        'embedder': embedder,
        'quantizer': quantizer,
        'bfv_engine': bfv_engine,
        'bfv_search': bfv_search,
        'int_docs': int_docs,
        'relevance_threshold': rag_config.relevance_threshold,
        'no_match_message': rag_config.no_match_message,
    }


def interactive_search(state):
    """Ask your own questions to the encrypted index."""

    print(f"\n{'═' * 65}")
    print(f"  🔐 INTERACTIVE ENCRYPTED SEARCH")
    print(f"  Type your questions. Type 'quit' to exit.")
    print(f"{'═' * 65}")

    chunks = state['chunks']
    embedder = state['embedder']
    quantizer = state['quantizer']
    bfv_search = state['bfv_search']
    relevance_threshold = state.get('relevance_threshold', 0.20)
    no_match_message = state.get(
        'no_match_message',
        'Sorry, the query you asked is not present in the document.',
    )
    TOP_K = 3

    while True:
        query = input("\n  🔍 Your query: ").strip()

        if query.lower() in ('quit', 'exit', 'q'):
            print("  👋 Done!")
            break

        if not query:
            continue

        t0 = time.time()

        # embed
        query_emb = embedder.embed_query(query)

        # quantize
        int_query = quantizer.quantize_query(query_emb)

        # encrypted search
        result = bfv_search.search(
            int_query, mode="ct_ct", top_k=TOP_K
        )

        top_score = result.top_k_scores[0] if result.top_k_scores else 0.0
        max_possible = quantizer.get_max_dot_product(len(int_query))
        normalized_top_score = float(top_score) / float(max_possible)
        has_match = normalized_top_score >= relevance_threshold

        total_ms = (time.time() - t0) * 1000

        print(f"\n  🔐 Encrypted search: {total_ms:.0f}ms")
        print(f"  Top score (normalized): {normalized_top_score:.4f} "
              f"| Threshold: {relevance_threshold:.4f}")
        if not has_match:
            print(f"  ⚠️  {no_match_message}")
            continue
        print(f"  Top-{TOP_K} results:\n")

        for j, idx in enumerate(result.top_k_indices):
            score = result.all_scores[idx]
            text = chunks[idx].replace('\n', ' ')

            print(f"    [{j+1}] Score: {score}")
            # word wrap
            words = text.split()
            line = "        "
            for w in words:
                if len(line) + len(w) > 75:
                    print(line)
                    line = "        "
                line += w + " "
            if line.strip():
                print(line)
            print()


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    doc_file = None
    query_file = None
    interactive = False
    threshold_override = None

    for arg in sys.argv[1:]:
        if arg == "--interactive":
            interactive = True
        elif arg.startswith("--threshold="):
            try:
                threshold_override = float(arg.split("=", 1)[1])
            except ValueError:
                print("❌ Invalid value for --threshold. Example: --threshold=0.02")
                sys.exit(1)
        elif arg == "--threshold":
            print("❌ Please pass a value. Example: --threshold=0.02")
            sys.exit(1)
        elif arg == "--help":
            print("""
Usage:
    python3 run_txt.py data/my_document.txt
    python3 run_txt.py data/my_document.txt data/my_queries.txt
    python3 run_txt.py data/my_document.txt --interactive
    python3 run_txt.py data/my_document.txt data/my_queries.txt --interactive
    python3 run_txt.py data/my_document.txt --interactive --threshold=0.02
            """)
            sys.exit(0)
        elif doc_file is None:
            doc_file = arg
        else:
            query_file = arg

    if doc_file is None:
        print("❌ Please provide a text file:")
        print("   python3 run_txt.py data/my_document.txt")
        print("   python3 run_txt.py data/my_document.txt data/my_queries.txt")
        print("   python3 run_txt.py --help")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🔐  cipherRAG — Run on Your Own Documents  🔐             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Load document
    chunks = load_txt_document(doc_file)

    # Load queries
    queries = load_queries(query_file)

    # Run comparison
    state = run_full_comparison(
        chunks,
        queries,
        relevance_threshold=threshold_override,
    )

    # Interactive mode
    if interactive and state:
        interactive_search(state)