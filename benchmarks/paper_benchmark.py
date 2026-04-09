import csv
import os
import random
import statistics
import time
from typing import Any, Dict, List

from paper_metrics import (
    jaccard_topk,
    mean_abs_score_error,
    payload_size_kb,
    payload_size_mb,
    precision_at_k,
    recall_at_k,
)
from paper_monitor import ProcessMonitor, get_system_info

RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "paper_benchmark_results.csv")

TOPK_VALUES = [1, 3, 5]
DATASET_SIZES = [100, 1000, 5000]
NUM_REPEATS = 10
SAMPLE_INTERVAL = 0.1

QUERIES = [
    "What is the refund policy for damaged products?",
    "Summarize the password reset requirements.",
    "How should a security incident be reported?",
    "What is the maintenance schedule for the server?",
    "What are the deployment steps for the release pipeline?",
]

HE_PARAMS = {
    "he_scheme": "BFV",
    "poly_modulus_degree": 8192,
    "plain_modulus": 1032193,
    "coeff_mod_bit_sizes": "60,40,40,60",
    "security_level_bits": 128,
    "scale_bits": "",
    "quantization_scale": 100.0,
}

SYSTEM_INFO = get_system_info()


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_dummy_corpus(n: int) -> List[Dict[str, Any]]:
    corpus = []
    for i in range(n):
        corpus.append(
            {
                "chunk_id": f"chunk_{i}",
                "text": (
                    f"This is chunk {i}. It contains policy, maintenance, password, "
                    "incident response, and deployment information."
                ),
            }
        )
    return corpus


def subset_corpus(corpus: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return corpus[:n]


def plaintext_retrieve(query: str, corpus: List[Dict[str, Any]], topk: int = 5) -> Dict[str, Any]:
    """
    ADAPTER: PLAINTEXT
    Replace this with your real plaintext pipeline.
    """
    t0 = time.perf_counter()

    seed = abs(hash(query)) % (10**8)
    rng = random.Random(seed)
    scored = []
    for item in corpus:
        score = rng.random()
        scored.append((item["chunk_id"], score, item["text"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:topk]
    search_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "retrieved_chunk_ids": [x[0] for x in top],
        "scores": {x[0]: x[1] for x in top},
        "context_texts": [x[2] for x in top],
        "raw_query_payload": {"query": query},
        "raw_response_payload": {"topk_ids": [x[0] for x in top]},
        "embedding_ms": 0.0,
        "search_ms": round(search_ms, 2),
    }


def secure_retrieve(query: str, corpus: List[Dict[str, Any]], topk: int = 5) -> Dict[str, Any]:
    """
    ADAPTER: SECURE
    Replace this with your real HE/private pipeline.
    """
    seed = abs(hash(query)) % (10**8)
    rng = random.Random(seed)

    client_embedding_ms = 4.2
    client_encryption_ms = 18.7

    t0 = time.perf_counter()
    scored = []
    for item in corpus:
        base = rng.random()
        noisy = base + rng.uniform(-0.03, 0.03)
        scored.append((item["chunk_id"], noisy, item["text"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:topk]
    server_search_ms = (time.perf_counter() - t0) * 1000.0 + 120.0

    client_decryption_ms = 9.4

    encrypted_query = b"q" * (1024 * 256)
    encrypted_response = b"r" * (1024 * 64)
    eval_keys = b"e" * (1024 * 1024 * 8)

    return {
        "retrieved_chunk_ids": [x[0] for x in top],
        "scores": {x[0]: x[1] for x in top},
        "context_texts": [x[2] for x in top],
        "raw_query_payload": encrypted_query,
        "raw_response_payload": encrypted_response,
        "eval_key_payload": eval_keys,
        "client_embedding_ms": client_embedding_ms,
        "client_encryption_ms": client_encryption_ms,
        "server_search_ms": round(server_search_ms, 2),
        "client_decryption_ms": client_decryption_ms,
        "he_params": HE_PARAMS,
    }


def generate_answer_local(query: str, context_texts: List[str]) -> Dict[str, Any]:
    """
    Replace with your local LLM call.
    """
    _ = context_texts
    t0 = time.perf_counter()
    time.sleep(0.2)
    dt = (time.perf_counter() - t0) * 1000.0

    answer = f"Generated answer for: {query}"
    tokens_generated = max(20, len(answer.split()) * 2)
    return {
        "answer": answer,
        "generation_latency_ms": round(dt, 2),
        "tokens_generated": tokens_generated,
    }


def append_csv(row: Dict[str, Any]):
    ensure_results_dir()
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def benchmark_plain(query: str, corpus: List[Dict[str, Any]], run_id: int) -> Dict[str, Any]:
    monitor = ProcessMonitor(sample_interval=SAMPLE_INTERVAL)

    monitor.start()
    e2e_start = time.perf_counter()

    plain = plaintext_retrieve(query, corpus, topk=max(TOPK_VALUES))
    gen = generate_answer_local(query, plain["context_texts"])

    e2e_ms = (time.perf_counter() - e2e_start) * 1000.0
    monitor.stop()
    mem = monitor.summary()

    tokens_per_sec = gen["tokens_generated"] / max(gen["generation_latency_ms"] / 1000.0, 1e-6)

    row = {
        "run_id": run_id,
        "mode": "plain",
        "dataset_size": len(corpus),
        "query": query,

        "hostname": SYSTEM_INFO["hostname"],
        "os": SYSTEM_INFO["os"],
        "python_version": SYSTEM_INFO["python_version"],
        "cpu_count_logical": SYSTEM_INFO["cpu_count_logical"],
        "cpu_count_physical": SYSTEM_INFO["cpu_count_physical"],
        "total_ram_mb": SYSTEM_INFO["total_ram_mb"],

        "he_scheme": "none",
        "poly_modulus_degree": "",
        "plain_modulus": "",
        "coeff_mod_bit_sizes": "",
        "security_level_bits": "",
        "scale_bits": "",
        "quantization_scale": "",
        "eval_key_payload_mb": 0.0,

        "client_embedding_ms": round(plain.get("embedding_ms", 0.0), 2),
        "client_encryption_ms": 0.0,
        "server_search_ms": round(plain.get("search_ms", 0.0), 2),
        "client_decryption_ms": 0.0,
        "generation_latency_ms": round(gen["generation_latency_ms"], 2),
        "end_to_end_latency_ms": round(e2e_ms, 2),

        "query_payload_kb": round(payload_size_kb(plain.get("raw_query_payload")), 2),
        "response_payload_kb": round(payload_size_kb(plain.get("raw_response_payload")), 2),

        "topk_returned": len(plain["retrieved_chunk_ids"]),
        "retrieved_chunk_ids": "|".join(plain["retrieved_chunk_ids"]),
        "reference_chunk_ids": "|".join(plain["retrieved_chunk_ids"]),

        "recall_at_1": 1.0,
        "recall_at_3": 1.0 if len(plain["retrieved_chunk_ids"]) >= 3 else 0.0,
        "recall_at_5": 1.0 if len(plain["retrieved_chunk_ids"]) >= 5 else 0.0,
        "precision_at_1": 1.0,
        "precision_at_3": 1.0 if len(plain["retrieved_chunk_ids"]) >= 3 else 0.0,
        "precision_at_5": 1.0 if len(plain["retrieved_chunk_ids"]) >= 5 else 0.0,
        "jaccard_at_5": 1.0 if len(plain["retrieved_chunk_ids"]) >= 5 else 0.0,
        "mae_score_top5": 0.0,

        "tokens_generated": gen["tokens_generated"],
        "tokens_per_sec": round(tokens_per_sec, 2),
        "context_chars": sum(len(x) for x in plain["context_texts"]),

        "peak_rss_mb": mem["peak_rss_mb"],
        "avg_rss_mb": mem["avg_rss_mb"],
        "peak_cpu_percent": mem["peak_cpu_percent"],
        "avg_cpu_percent": mem["avg_cpu_percent"],
        "peak_system_mem_used_mb": mem["peak_system_mem_used_mb"],
        "avg_system_mem_percent": mem["avg_system_mem_percent"],
    }

    append_csv(row)
    return row


def benchmark_secure_against_plain(query: str, corpus: List[Dict[str, Any]], run_id: int) -> Dict[str, Any]:
    plain = plaintext_retrieve(query, corpus, topk=max(TOPK_VALUES))
    plain_ids = plain["retrieved_chunk_ids"]
    plain_scores = plain.get("scores", {})

    monitor = ProcessMonitor(sample_interval=SAMPLE_INTERVAL)

    monitor.start()
    e2e_start = time.perf_counter()

    secure = secure_retrieve(query, corpus, topk=max(TOPK_VALUES))
    gen = generate_answer_local(query, secure["context_texts"])

    e2e_ms = (time.perf_counter() - e2e_start) * 1000.0
    monitor.stop()
    mem = monitor.summary()

    secure_ids = secure["retrieved_chunk_ids"]
    secure_scores = secure.get("scores", {})
    he = secure.get("he_params", {})

    r1 = recall_at_k(plain_ids, secure_ids, 1)
    r3 = recall_at_k(plain_ids, secure_ids, 3)
    r5 = recall_at_k(plain_ids, secure_ids, 5)

    p1 = precision_at_k(plain_ids, secure_ids, 1)
    p3 = precision_at_k(plain_ids, secure_ids, 3)
    p5 = precision_at_k(plain_ids, secure_ids, 5)

    jac5 = jaccard_topk(plain_ids, secure_ids, 5)
    mae5 = mean_abs_score_error(plain_scores, secure_scores, topk=5)

    tokens_per_sec = gen["tokens_generated"] / max(gen["generation_latency_ms"] / 1000.0, 1e-6)

    row = {
        "run_id": run_id,
        "mode": "secure_he",
        "dataset_size": len(corpus),
        "query": query,

        "hostname": SYSTEM_INFO["hostname"],
        "os": SYSTEM_INFO["os"],
        "python_version": SYSTEM_INFO["python_version"],
        "cpu_count_logical": SYSTEM_INFO["cpu_count_logical"],
        "cpu_count_physical": SYSTEM_INFO["cpu_count_physical"],
        "total_ram_mb": SYSTEM_INFO["total_ram_mb"],

        "he_scheme": he.get("he_scheme", ""),
        "poly_modulus_degree": he.get("poly_modulus_degree", ""),
        "plain_modulus": he.get("plain_modulus", ""),
        "coeff_mod_bit_sizes": he.get("coeff_mod_bit_sizes", ""),
        "security_level_bits": he.get("security_level_bits", ""),
        "scale_bits": he.get("scale_bits", ""),
        "quantization_scale": he.get("quantization_scale", ""),
        "eval_key_payload_mb": round(payload_size_mb(secure.get("eval_key_payload")), 4),

        "client_embedding_ms": round(secure.get("client_embedding_ms", 0.0), 2),
        "client_encryption_ms": round(secure.get("client_encryption_ms", 0.0), 2),
        "server_search_ms": round(secure.get("server_search_ms", 0.0), 2),
        "client_decryption_ms": round(secure.get("client_decryption_ms", 0.0), 2),
        "generation_latency_ms": round(gen["generation_latency_ms"], 2),
        "end_to_end_latency_ms": round(e2e_ms, 2),

        "query_payload_kb": round(payload_size_kb(secure.get("raw_query_payload")), 2),
        "response_payload_kb": round(payload_size_kb(secure.get("raw_response_payload")), 2),

        "topk_returned": len(secure_ids),
        "retrieved_chunk_ids": "|".join(secure_ids),
        "reference_chunk_ids": "|".join(plain_ids),

        "recall_at_1": round(r1, 4),
        "recall_at_3": round(r3, 4),
        "recall_at_5": round(r5, 4),
        "precision_at_1": round(p1, 4),
        "precision_at_3": round(p3, 4),
        "precision_at_5": round(p5, 4),
        "jaccard_at_5": round(jac5, 4),
        "mae_score_top5": round(mae5, 6),

        "tokens_generated": gen["tokens_generated"],
        "tokens_per_sec": round(tokens_per_sec, 2),
        "context_chars": sum(len(x) for x in secure["context_texts"]),

        "peak_rss_mb": mem["peak_rss_mb"],
        "avg_rss_mb": mem["avg_rss_mb"],
        "peak_cpu_percent": mem["peak_cpu_percent"],
        "avg_cpu_percent": mem["avg_cpu_percent"],
        "peak_system_mem_used_mb": mem["peak_system_mem_used_mb"],
        "avg_system_mem_percent": mem["avg_system_mem_percent"],
    }

    append_csv(row)
    return row


def summarize(rows: List[Dict[str, Any]]):
    print("\n================ PAPER BENCHMARK SUMMARY ================")
    grouped = {}
    for r in rows:
        grouped.setdefault((r["mode"], r["dataset_size"]), []).append(r)

    keys_to_show = [
        "client_encryption_ms",
        "server_search_ms",
        "client_decryption_ms",
        "generation_latency_ms",
        "end_to_end_latency_ms",
        "query_payload_kb",
        "response_payload_kb",
        "eval_key_payload_mb",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "peak_rss_mb",
        "avg_cpu_percent",
    ]

    for (mode, ds), items in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"\nMode={mode}, Dataset={ds}")
        for key in keys_to_show:
            vals = [float(i[key]) for i in items if str(i[key]) != ""]
            if vals:
                print(f"  {key}: mean={statistics.mean(vals):.3f}, std={statistics.pstdev(vals):.3f}")


def main():
    ensure_results_dir()

    master_n = max(DATASET_SIZES)
    master_corpus = generate_dummy_corpus(master_n)

    all_rows = []
    run_id = 1

    for n in DATASET_SIZES:
        corpus_n = subset_corpus(master_corpus, n)
        for query in QUERIES:
            for _ in range(NUM_REPEATS):
                all_rows.append(benchmark_plain(query, corpus_n, run_id))
                run_id += 1

                all_rows.append(benchmark_secure_against_plain(query, corpus_n, run_id))
                run_id += 1

    summarize(all_rows)
    print(f"\nResults written to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
