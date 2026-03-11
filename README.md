# 🔐 Private RAG: Gap-Aware Integer Homomorphic Encryption

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-0.3.14-green.svg)](https://github.com/OpenMined/TenSEAL)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

**Private RAG** is an optimized, privacy-preserving **Retrieval-Augmented Generation (RAG)** architecture. It solves the critical privacy bottleneck in cloud-based AI: **searching a vector database without revealing the query or the documents to the server.**

Standard implementations use **CKKS** (floating-point homomorphic encryption), which is computationally heavy and introduces cryptographic noise. By introducing the **Score-Gap Theorem**, we mathematically bridge floating-point embeddings to the **BFV** (Brakerski-Fan-Vercauteren) exact-integer encryption scheme.

> Combined with **SIMD batching** and **element-wise "no-rotation" evaluation**, our architecture achieves a **68.2x reduction in latency** over the CKKS baseline with **zero cryptographic error**.

---

## 📁 Repository Structure

```
private-rag/
├── requirements.txt         # Python dependencies
├── config.py                # Central settings (chunk size, encryption params)
├── main.py                  # Basic pipeline demo (BFV only)
├── compare_fast.py          # 🔬 CORE RESEARCH: Runs CKKS vs BFV benchmarks
├── run_txt.py               # Utility to run the system on your own text files
├── generate_plots.py        # Generates publication-ready graphs (PNG)
├── presentation.py          # Manim script for the animated video presentation
│
├── src/                     # Core system modules
│   ├── chunker.py           #   Text processing and splitting
│   ├── embedder.py          #   HuggingFace MiniLM floating-point embeddings
│   ├── quantizer.py         #   Global-scaled Int8 quantization logic
│   ├── encryption.py        #   TenSEAL BFV and CKKS engine implementations
│   ├── search.py            #   Encrypted vector search algorithms
│   ├── rag_pipeline.py      #   End-to-end pipeline wrapper
│   └── utils.py             #   Math helpers (Score-Gap calculation)
│
├── data/                    # 📥 Put your custom .txt or .pdf files here
│
└── results/                 # 📤 Output logs and generated plots go here
    └── plots/               #   Auto-generated benchmark figures
```


---

## ⚙️ Step-by-Step Setup

> **Prerequisites:** Python 3.9+  
> Homomorphic encryption relies on C++ bindings, so some system-level tools are required before installing the Python packages.

### 1. Install System Dependencies (macOS)

If you are on a Mac, you must install `cmake` (required by TenSEAL) and rendering tools (required for the Manim video presentation):

```bash
brew install cmake
brew install py3cairo ffmpeg pango pkg-config
```
2. Clone and Setup the Virtual Environment
```
# Clone the repository
git clone https://github.com/yourusername/private-rag.git
cd private-rag

# Create an isolated Python environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```
3. Install Python Packages
```
# Install the core cryptographic and ML libraries
pip install -r requirements.txt

# Install Manim (for the video presentation only)
pip install manim
```
🚀 How to Run the Code
The repository is designed to be easily reproducible. You can run the scientific benchmarks, test it on your own data, or interact with it.

Mode 1: Reproduce the Scientific Benchmarks
Run the head-to-head comparison between Plaintext, CKKS (Standard), and our optimized BFV methods:
```
python3 compare_fast.py
```
Output: Comparison tables printed to console and saved to results/fast_comparison.txt

Mode 2: Test on Your Own Documents
You can test the encryption pipeline on any text file:

Place your text file in the data/ folder (e.g., data/my_document.txt)
Place a list of questions in data/my_queries.txt (one question per line)
Run the script:
```
python3 run_txt.py data/my_document.txt data/my_queries.txt
```
Mode 3: Interactive Encrypted Chat
Chat with your encrypted document in real-time via the terminal:

python3 run_txt.py data/my_document.txt --interactive

📊 Generate Benchmark Plots
To generate publication-ready figures from the benchmark data:
```
python3 generate_plots.py
```
Plots will be saved to results/plots/.
