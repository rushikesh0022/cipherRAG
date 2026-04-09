import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = "results/paper_benchmark_results.csv"
PLOT_DIR = "results/plots_paper"


def ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def save_bar(series, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=220)
    plt.close()


def plot_latency_breakdown(df):
    largest_n = df["dataset_size"].max()
    sub = df[df["dataset_size"] == largest_n]
    grouped = sub.groupby("mode")[[
        "client_embedding_ms",
        "client_encryption_ms",
        "server_search_ms",
        "client_decryption_ms",
        "generation_latency_ms",
    ]].mean()

    plt.figure(figsize=(9, 6))
    bottom = None
    cols = [
        "client_embedding_ms",
        "client_encryption_ms",
        "server_search_ms",
        "client_decryption_ms",
        "generation_latency_ms",
    ]

    for c in cols:
        if bottom is None:
            plt.bar(grouped.index, grouped[c], label=c)
            bottom = grouped[c].values
        else:
            plt.bar(grouped.index, grouped[c], bottom=bottom, label=c)
            bottom = bottom + grouped[c].values

    plt.title(f"Latency Breakdown (N={largest_n})")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "latency_breakdown.png"), dpi=220)
    plt.close()


def plot_accuracy(df):
    largest_n = df["dataset_size"].max()
    sub = df[(df["dataset_size"] == largest_n) & (df["mode"] == "secure_he")]
    means = sub[["recall_at_1", "recall_at_3", "recall_at_5"]].mean()

    plt.figure(figsize=(7, 5))
    means.plot(marker="o")
    plt.title(f"Recall@K vs Plaintext (N={largest_n})")
    plt.ylabel("Recall@K")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "accuracy_recall.png"), dpi=220)
    plt.close()


def plot_scalability(df):
    grouped = df.groupby(["dataset_size", "mode"])["server_search_ms"].mean().unstack()

    plt.figure(figsize=(8, 5))
    for col in grouped.columns:
        plt.plot(grouped.index, grouped[col], marker="o", label=col)

    plt.title("Search Scalability vs Dataset Size")
    plt.xlabel("Dataset Size (N)")
    plt.ylabel("Server Search Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scalability_search_time.png"), dpi=220)
    plt.close()


def plot_resource_footprint(df):
    largest_n = df["dataset_size"].max()
    sub = df[df["dataset_size"] == largest_n]
    metrics = sub.groupby("mode")[[
        "peak_rss_mb",
        "query_payload_kb",
        "response_payload_kb",
        "eval_key_payload_mb",
    ]].mean()

    save_bar(metrics["peak_rss_mb"], f"Peak RAM Usage (N={largest_n})", "MB", "peak_rss.png")
    save_bar(
        metrics["query_payload_kb"],
        f"Query Payload Size (N={largest_n})",
        "KB",
        "query_payload.png",
    )
    save_bar(
        metrics["response_payload_kb"],
        f"Response Payload Size (N={largest_n})",
        "KB",
        "response_payload.png",
    )
    save_bar(
        metrics["eval_key_payload_mb"],
        f"Evaluation Key Size (N={largest_n})",
        "MB",
        "eval_key_payload.png",
    )


def plot_hyperparam_effect(df):
    secure = df[df["mode"] == "secure_he"]
    if secure["poly_modulus_degree"].nunique() <= 1:
        return

    grouped = secure.groupby("poly_modulus_degree")["server_search_ms"].mean()
    plt.figure(figsize=(8, 5))
    grouped.plot(marker="o")
    plt.title("Effect of Polynomial Modulus Degree on Search Latency")
    plt.xlabel("poly_modulus_degree")
    plt.ylabel("Server Search Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "poly_modulus_vs_latency.png"), dpi=220)
    plt.close()


def main():
    ensure_dir()
    df = pd.read_csv(RESULTS_CSV)

    plot_latency_breakdown(df)
    plot_accuracy(df)
    plot_scalability(df)
    plot_resource_footprint(df)
    plot_hyperparam_effect(df)

    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
