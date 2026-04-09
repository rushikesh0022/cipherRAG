import matplotlib.pyplot as plt
import numpy as np
import os

# Create folder for plots
os.makedirs("results/plots", exist_ok=True)

# ─── EXACT DATA FROM YOUR RUN ───
methods = [
    "CKKS Float32\n(Baseline)", 
    "BFV Int8\n(Standard)", 
    "BFV Int8\n(No-Rotation)", 
    "BFV Int8\n(Batched)", 
    "BFV Int8\n(Batch+PCA)"
]
latencies = [1429, 3079, 414, 64, 21]
crypto_errors = [5.62e-06, 0, 0, 0, 0]
speedups_vs_ckks = [1.0, 0.46, 3.45, 22.3, 68.0] # 1429 / latency

# ─── PLOT 1: LATENCY REDUCTION (Log Scale) ───
plt.figure(figsize=(10, 6))
colors = ['#e74c3c', '#95a5a6', '#3498db', '#2980b9', '#2ecc71']

bars = plt.bar(methods, latencies, color=colors)
plt.yscale('log')
plt.ylabel("Latency per Query (ms) - Log Scale", fontsize=12, fontweight='bold')
plt.title("Encrypted Search Latency: CKKS vs Optimized BFV", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of bars
for bar, lat in zip(bars, latencies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15, 
             f"{lat} ms", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("results/plots/figure1_latency.png", dpi=300)
print("✅ Saved Figure 1: Latency Plot")

# ─── PLOT 2: THE OPTIMIZATION PIPELINE (Speedup) ───
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, speedups_vs_ckks, color=colors)
plt.ylabel("Speedup Multiplier (vs CKKS Baseline)", fontsize=12, fontweight='bold')
plt.title("Performance Gain through BFV Optimizations", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, speedup in zip(bars, speedups_vs_ckks):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f"{speedup:.1f}x", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("results/plots/figure2_speedup.png", dpi=300)
print("✅ Saved Figure 2: Speedup Plot")

# ─── PLOT 3: CRYPTOGRAPHIC ERROR COMPARISON ───
plt.figure(figsize=(7, 5))
methods_err = ["CKKS Float32", "BFV Int8 (All Versions)"]
errors = [5.62, 0] # Scaled for display (x 10^-6)

bars = plt.bar(methods_err, errors, color=['#e74c3c', '#2ecc71'], width=0.5)
plt.ylabel("Max Cryptographic Error (x 10⁻⁶)", fontsize=12, fontweight='bold')
plt.title("Cryptographic Approximation Error", fontsize=14, fontweight='bold')

plt.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.1, 
         "5.62e-06", ha='center', va='bottom', fontweight='bold')
plt.text(bars[1].get_x() + bars[1].get_width()/2, 0.1, 
         "EXACT (0.0)", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("results/plots/figure3_error.png", dpi=300)
print("✅ Saved Figure 3: Error Plot")