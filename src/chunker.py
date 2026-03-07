# src/chunker.py
# ═══════════════════════════════════════════════════════════
#  Document ingestion: PDF/text → chunks
# ═══════════════════════════════════════════════════════════

import os
from typing import List, Tuple
from config import ChunkConfig


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    import fitz  # PyMuPDF

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages_text.append(text.strip())
    doc.close()

    full_text = "\n\n".join(pages_text)
    print(f"  📄 Extracted {len(full_text):,} characters from {len(pages_text)} pages")
    return full_text


def chunk_text(text: str, config: ChunkConfig = None) -> List[str]:
    """
    Split text into overlapping chunks.

    Uses word-level splitting with overlap for context continuity.
    Each chunk is a self-contained passage for embedding.
    """
    if config is None:
        config = ChunkConfig()

    # clean the text
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # split into words
    words = text.split()

    if len(words) == 0:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + config.chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        # only keep chunks with enough content
        if len(chunk_text) >= config.min_chunk_length:
            chunks.append(chunk_text)

        # advance by (chunk_size - overlap)
        start += config.chunk_size - config.chunk_overlap

    print(f"  📝 Created {len(chunks)} chunks "
          f"(size={config.chunk_size}, overlap={config.chunk_overlap})")

    return chunks


def load_document(path: str, config: ChunkConfig = None) -> List[str]:
    """
    Load a document (PDF or plain text) and return chunks.
    """
    if config is None:
        config = ChunkConfig()

    if path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(path)
    elif path.lower().endswith(('.txt', '.md')):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  📄 Loaded {len(text):,} characters from {path}")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    chunks = chunk_text(text, config)
    return chunks


def create_sample_corpus() -> Tuple[List[str], List[str]]:
    """
    Create a sample document corpus and test queries for testing
    when no PDF is available.

    Returns: (chunks, queries)
    """
    chunks = [
        "Homomorphic encryption allows computation on encrypted data without "
        "decryption. The CKKS scheme supports approximate arithmetic on real "
        "numbers making it suitable for machine learning applications and "
        "vector search operations.",

        "The Ring Learning With Errors problem provides the security foundation "
        "for lattice-based cryptography. Breaking RLWE is believed to be hard "
        "even for quantum computers with sufficient parameters.",

        "Medical records contain highly sensitive patient information including "
        "diagnoses medications and genetic data. HIPAA regulations impose severe "
        "penalties for unauthorized disclosure of protected health information.",

        "Retrieval Augmented Generation combines document retrieval with language "
        "model generation. The retrieval step finds relevant document chunks "
        "using vector similarity search based on dot products or cosine similarity.",

        "Transport Layer Security protects data during network transmission but "
        "requires decryption at the server endpoint. This creates a vulnerability "
        "known as the decryption point where data is exposed in plaintext.",

        "Financial institutions process millions of transactions daily. Credit "
        "card numbers account balances and transaction histories must be protected "
        "under PCI-DSS and SOX compliance regulations.",

        "The CKKS encoding scheme maps real numbers to polynomial rings using a "
        "scaling factor called the global scale. The scale parameter controls "
        "precision with larger scales providing more decimal digits of accuracy.",

        "Law firms handle privileged attorney-client communications that must "
        "remain strictly confidential. Legal documents including contracts "
        "depositions and case strategies require maximum protection from disclosure.",

        "Vector embeddings represent text as dense numerical arrays in high "
        "dimensional space. Cosine similarity between L2-normalized vectors "
        "equals their dot product enabling efficient semantic search.",

        "Differential privacy adds carefully calibrated random noise to query "
        "results or training data. The privacy parameter epsilon controls the "
        "trade-off between privacy strength and data utility.",

        "Trusted execution environments like Intel SGX create hardware-isolated "
        "secure enclaves for computation. However multiple side-channel attacks "
        "including Spectre and Foreshadow have compromised TEE implementations.",

        "Federated learning distributes model training across multiple devices "
        "keeping data local on each device. Each participant trains on private "
        "data and only shares gradient updates not raw training examples.",

        "The polynomial modulus degree N in CKKS determines both the security "
        "level and the ciphertext slot capacity. N equals 8192 provides "
        "approximately 128 bits of security with 4096 available slots.",

        "Secure multi-party computation protocols allow mutually distrusting "
        "parties to jointly compute agreed-upon functions without revealing "
        "individual inputs to any other participant in the protocol.",

        "Neural network inference on encrypted data was first demonstrated by "
        "CryptoNets. The main challenge is approximating non-linear activation "
        "functions like ReLU as low-degree polynomials for HE compatibility.",

        "Privacy regulations including GDPR grant individuals rights over their "
        "personal data including access rectification erasure and data portability. "
        "Cloud AI services must comply with these requirements.",

        "Ciphertext expansion is a fundamental overhead of homomorphic encryption. "
        "A 384-dimensional float32 vector occupying 1536 bytes expands to "
        "approximately 131072 bytes when encrypted with CKKS parameters.",

        "Locality sensitive hashing creates compact binary codes that approximately "
        "preserve similarity between vectors. Similar vectors map to the same hash "
        "bucket with high probability enabling sublinear nearest neighbor search.",

        "Zero knowledge proofs allow a prover to convince a verifier of a statement "
        "truth without revealing any information beyond the statement validity. "
        "ZK-SNARKs enable succinct non-interactive zero knowledge proofs.",

        "The blind librarian metaphor describes a server that retrieves relevant "
        "information without being able to read document contents. Homomorphic "
        "encrypted vector search makes this computationally possible.",

        "Batch encoding in CKKS packs multiple real values into distinct slots of "
        "a single ciphertext polynomial. This enables SIMD-style parallel "
        "operations on multiple data points simultaneously.",

        "Private information retrieval protocols allow users to fetch database "
        "records without revealing which record was accessed. Computational PIR "
        "achieves this using homomorphic encryption techniques.",

        "Approximate nearest neighbor algorithms including HNSW IVF and ScaNN "
        "trade search exactness for dramatic speed improvements. These are "
        "essential for scaling vector search to billions of documents.",

        "The coefficient modulus chain in CKKS determines the available "
        "multiplicative depth for computations. Each homomorphic multiplication "
        "consumes one chain level limiting total computation complexity.",

        "Cross-encoder reranking models process query-document pairs jointly "
        "through a transformer architecture to produce fine-grained relevance "
        "scores. They are more accurate but much slower than bi-encoder retrieval.",

        "Oblivious RAM protocols hide memory access patterns from an adversarial "
        "server. Without ORAM an attacker observing which encrypted records are "
        "accessed can infer significant information about queries.",

        "Embedding model distillation compresses large transformer encoders into "
        "smaller student models that produce similar vector representations at "
        "a fraction of the computational cost and latency.",

        "Dense passage retrieval using learned embeddings has largely replaced "
        "traditional sparse methods like BM25 and TF-IDF for semantic search "
        "tasks in modern question answering systems.",

        "Relinearization reduces ciphertext size after homomorphic multiplication "
        "by converting a degree-2 ciphertext back to degree-1. This prevents "
        "exponential growth of ciphertext components during computation.",

        "Data sovereignty laws require that certain categories of data including "
        "health financial and government records must be stored and processed "
        "within specific national or regional boundaries.",
    ]

    queries = [
        "How does homomorphic encryption work?",
        "What regulations protect medical patient data?",
        "How is vector similarity search performed?",
        "What is the decryption point vulnerability?",
        "How does CKKS encode real numbers?",
        "What are the risks for legal document privacy?",
        "How does differential privacy add noise?",
        "What is ciphertext expansion overhead?",
        "How does federated learning work?",
        "What is retrieval augmented generation?",
        "What are trusted execution environments?",
        "How do zero knowledge proofs work?",
        "What is the polynomial modulus degree?",
        "How does private information retrieval work?",
        "What is approximate nearest neighbor search?",
    ]

    return chunks, queries
