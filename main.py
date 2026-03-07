# main.py
# ═══════════════════════════════════════════════════════════
#  Private RAG: BFV Int8 Encrypted Vector Search
#
#  USAGE:
#    python main.py                    # run with sample corpus
#    python main.py data/document.pdf  # run with your PDF
#    python main.py --interactive      # interactive query mode
#
#  WHAT THIS DOES:
#    1. Loads document → chunks
#    2. Generates embeddings (all-MiniLM-L6-v2)
#    3. Quantizes to int8 (preserves ranking)
#    4. Encrypts with BFV (exact integer HE)
#    5. Performs encrypted similarity search
#    6. Returns top-k relevant chunks
#
#  The server NEVER sees plaintext data.
# ═══════════════════════════════════════════════════════════

import sys
import time
import numpy as np

from config import RAGConfig, DEFAULT_CONFIG
from src.chunker import create_sample_corpus, load_document
from src.rag_pipeline import PrivateRAGPipeline
from src.utils import compute_score_gaps


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🔐  PRIVATE RAG: BFV Int8 Encrypted Vector Search  🔐     ║
║                                                              ║
║   Scheme:  BFV (exact integer homomorphic encryption)        ║
║   Vectors: int8-quantized embeddings                         ║
║   Error:   ZERO (exact arithmetic on integers)               ║
║   Trust:   Single untrusted server                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def run_with_sample():
    """Run the full pipeline with the built-in sample corpus."""

    print_banner()

    # ── Configuration ──
    config = RAGConfig()
    config.search_mode = "ct_ct"   # full privacy: both query and docs encrypted
    config.top_k = 3

    # ── Create pipeline ──
    pipeline = PrivateRAGPipeline(config)

    # ── Step 1: Load sample corpus ──
    chunks, queries = create_sample_corpus()
    pipeline.ingest_chunks(chunks)

    # ── Step 2: Generate embeddings ──
    pipeline.generate_embeddings()

    # ── Step 3: Quantize to int8 ──
    pipeline.quantize()

    # ── Score gap analysis (theoretical contribution) ──
    query_embeddings = pipeline.embedder.embed_queries(queries)
    gap_analysis = compute_score_gaps(
        pipeline.doc_embeddings, query_embeddings, k=config.top_k
    )

    print("\n" + "=" * 60)
    print("  SCORE-GAP ANALYSIS (Theoretical Contribution)")
    print("=" * 60)
    print(f"""
  The Score-Gap Theorem:
  ──────────────────────
  For top-{config.top_k} retrieval, encryption error ε must satisfy:
    ε < δₖ/2  where δₖ = score(k) - score(k+1)

  Measured gaps across {len(queries)} queries:
    Min gap:  {gap_analysis['min_gap']:.6f}
    Max gap:  {gap_analysis['max_gap']:.6f}  
    Mean gap: {gap_analysis['mean_gap']:.6f}
    Max tolerable error: {gap_analysis['max_tolerable_error']:.6f}

  BFV crypto error: 0  (exact integer arithmetic)
  → Score-gap theorem is TRIVIALLY SATISFIED for BFV
  → Ranking is ALWAYS correct (identical to plaintext int8)
    """)

    # ── Step 4: Verify int8 ranking preservation ──
    pipeline.quantize()  # re-fit just in case
    verification = pipeline.quantizer.verify_ranking(
        pipeline.doc_embeddings, query_embeddings,
        top_k=config.top_k,
    )

    # ── Step 5: Encrypt and index ──
    pipeline.encrypt_and_index()

    # ── Step 6: Verify BFV exactness ──
    pipeline.verify(query_embeddings)

    # ── Step 7: Run searches ──
    print("\n" + "=" * 60)
    print("  ENCRYPTED SEARCH RESULTS")
    print("=" * 60)

    all_results = pipeline.search_batch(queries, top_k=config.top_k)

    # ── Display results ──
    print(f"\n  {'─' * 70}")
    for i, res in enumerate(all_results):
        print(f"\n  Query {i+1}: \"{res['query']}\"")
        print(f"  Mode: {res['mode']}  |  Latency: {res['latency_ms']:.0f}ms")
        for j, chunk in enumerate(res['results']):
            print(f"    [{j+1}] Score: {chunk['score']:>8}  "
                  f"Chunk {chunk['index']}: "
                  f"\"{chunk['text'][:80]}...\"")

    # ── Accuracy comparison ──
    print("\n" + "=" * 60)
    print("  ACCURACY VERIFICATION")
    print("=" * 60)

    # ground truth from float32 plaintext
    ground_truth = []
    for qv in query_embeddings:
        scores = pipeline.doc_embeddings @ qv
        top = list(np.argsort(scores)[-config.top_k:][::-1])
        ground_truth.append(top)

    # encrypted results
    encrypted_rankings = [
        [r['index'] for r in res['results']]
        for res in all_results
    ]

    top1_ok = sum(1 for gt, enc in zip(ground_truth, encrypted_rankings)
                  if gt[0] == enc[0])
    topk_ok = sum(1 for gt, enc in zip(ground_truth, encrypted_rankings)
                  if set(gt) == set(enc))

    n = len(queries)
    print(f"""
  Comparison: BFV int8 encrypted vs float32 plaintext
  ────────────────────────────────────────────────────
  Top-1 accuracy: {top1_ok}/{n} ({top1_ok/n*100:.0f}%)
  Top-{config.top_k} accuracy: {topk_ok}/{n} ({topk_ok/n*100:.0f}%)

  Any accuracy loss is from int8 QUANTIZATION (not encryption).
  BFV adds ZERO error to the int8 computation.
    """)

    # ── Pipeline summary ──
    pipeline.print_summary()

    return pipeline


def run_with_pdf(pdf_path: str):
    """Run the pipeline with a user-provided PDF."""

    print_banner()

    config = RAGConfig()
    config.search_mode = "ct_ct"
    config.top_k = 3

    pipeline = PrivateRAGPipeline(config)

    # load PDF
    pipeline.ingest_document(pdf_path)
    pipeline.generate_embeddings()
    pipeline.quantize()
    pipeline.encrypt_and_index()

    pipeline.print_summary()

    # interactive search
    print("\n" + "=" * 60)
    print("  INTERACTIVE SEARCH (type 'quit' to exit)")
    print("=" * 60)

    while True:
        query = input("\n  🔍 Enter query: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue

        result = pipeline.search(query)

        print(f"\n  Mode: {result['mode']}  |  Latency: {result['latency_ms']:.0f}ms")
        print(f"  Top-{result['top_k']} results:")
        for j, chunk in enumerate(result['results']):
            print(f"\n    [{j+1}] Score: {chunk['score']}")
            # wrap text nicely
            text = chunk['text']
            lines = [text[i:i+70] for i in range(0, len(text), 70)]
            for line in lines[:3]:
                print(f"        {line}")
            if len(lines) > 3:
                print(f"        ...")

    print("\n  Goodbye! 👋")


def run_interactive():
    """Run with sample corpus in interactive mode."""

    print_banner()

    config = RAGConfig()
    config.search_mode = "ct_ct"
    config.top_k = 3

    pipeline = PrivateRAGPipeline(config)

    chunks, _ = create_sample_corpus()
    pipeline.ingest_chunks(chunks)
    pipeline.generate_embeddings()
    pipeline.quantize()
    pipeline.encrypt_and_index()

    pipeline.print_summary()

    print("\n" + "=" * 60)
    print("  INTERACTIVE ENCRYPTED SEARCH")
    print("  (type 'quit' to exit)")
    print("=" * 60)

    while True:
        query = input("\n  🔍 Enter query: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue

        t0 = time.time()
        result = pipeline.search(query)
        total = (time.time() - t0) * 1000

        print(f"\n  🔐 Encrypted search ({result['mode']}) "
              f"completed in {total:.0f}ms")
        for j, chunk in enumerate(result['results']):
            print(f"\n    [{j+1}] Score: {chunk['score']}")
            text = chunk['text']
            lines = [text[i:i+70] for i in range(0, len(text), 70)]
            for line in lines[:4]:
                print(f"        {line}")
            if len(lines) > 4:
                print(f"        ...")

    print("\n  Goodbye! 👋")


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--interactive":
            run_interactive()
        elif arg.endswith(('.pdf', '.txt', '.md')):
            run_with_pdf(arg)
        else:
            print(f"Usage:")
            print(f"  python main.py                    # sample corpus demo")
            print(f"  python main.py data/document.pdf  # your PDF")
            print(f"  python main.py --interactive      # interactive mode")
    else:
        run_with_sample()
