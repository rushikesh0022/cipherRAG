# src/encryption.py
# ═══════════════════════════════════════════════════════════
#  Homomorphic Encryption Engines
#
#  TWO ENGINES:
#
#  1. BFVEngine  — YOUR method (exact integer HE on int8 vectors)
#  2. CKKSEngine — BASELINE (approximate float HE on float32 vectors)
#
#  The paper's argument:
#    CKKS float32: adds crypto error ε ≠ 0 → needs score-gap theorem
#    BFV int8:     adds crypto error ε = 0 → always correct ranking
# ═══════════════════════════════════════════════════════════

import time
import numpy as np
from typing import List, Optional
import tenseal as ts
from config import BFVConfig


# ═══════════════════════════════════════════════════════════
#  ENGINE 1: BFV (YOUR METHOD — exact integers)
# ═══════════════════════════════════════════════════════════

class BFVEngine:
    """
    BFV Homomorphic Encryption engine for integer vector operations.

    WHY BFV:
      - Designed for EXACT integer arithmetic
      - Zero approximation error on int operations
      - Dot product of int8 vectors is computed PERFECTLY
      - Ranking always matches plaintext int8

    Supports:
      ct×ct: both query and docs encrypted (full privacy)
      ct×pt: query encrypted, docs plaintext (query privacy only)
    """

    def __init__(self, config: BFVConfig = None):
        if config is None:
            config = BFVConfig()
        self.config = config
        self.ctx = None
        self.server_ctx = None
        self.plain_modulus = None
        self.max_safe_value = None
        self.keygen_time = 0.0

    def setup(self) -> 'BFVEngine':
        """Generate encryption keys and create client + server contexts."""
        print(f"\n  🔐 Setting up BFV encryption ...")
        print(f"     poly_modulus_degree = {self.config.poly_modulus_degree}")

        t0 = time.time()

        try:
            self.plain_modulus = ts.plain_modulus_batching(
                self.config.poly_modulus_degree,
                self.config.plain_mod_bits,
            )
        except Exception:
            fallback = {
                4096: 16760833,
                8192: 33538049,
                16384: 67104769,
            }
            self.plain_modulus = fallback.get(
                self.config.poly_modulus_degree, 33538049
            )

        self.max_safe_value = self.plain_modulus // 2

        self.ctx = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=self.config.poly_modulus_degree,
            plain_modulus=self.plain_modulus,
        )
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()

        pub_ctx = self.ctx.copy()
        pub_ctx.make_context_public()
        self.server_ctx = ts.context_from(pub_ctx.serialize())

        self.keygen_time = time.time() - t0

        print(f"     plain_modulus = {self.plain_modulus}")
        print(f"     max safe dot product = ±{self.max_safe_value:,}")
        print(f"     keygen time = {self.keygen_time:.2f}s")
        print(f"  ✅ BFV encryption ready")

        return self

    def encrypt_vector(self, int_vector: np.ndarray) -> bytes:
        """Encrypt a single integer vector. Returns serialized ciphertext."""
        int_list = [int(x) for x in int_vector.tolist()]
        enc = ts.bfv_vector(self.ctx, int_list)
        return enc.serialize()

    def encrypt_documents(self, int_embeddings: np.ndarray) -> List[bytes]:
        """Encrypt all document vectors."""
        print(f"\n  🔒 Encrypting {len(int_embeddings)} document vectors (BFV) ...")
        t0 = time.time()

        encrypted = []
        for i, vec in enumerate(int_embeddings):
            ct_bytes = self.encrypt_vector(vec)
            encrypted.append(ct_bytes)
            if (i + 1) % 10 == 0 or i == len(int_embeddings) - 1:
                print(f"     [{i+1}/{len(int_embeddings)}]", end='\r')

        encrypt_time = time.time() - t0
        ct_size = len(encrypted[0])

        print(f"\n  ✅ BFV documents encrypted:")
        print(f"     Time: {encrypt_time:.1f}s "
              f"({encrypt_time/len(int_embeddings)*1000:.1f} ms/vector)")
        print(f"     Ciphertext size: {ct_size:,} bytes/vector "
              f"({ct_size/1024:.0f} KB)")

        return encrypted

    def decrypt_score(self, score_bytes: bytes) -> int:
        """Decrypt a dot product result with signed conversion."""
        sv = ts.bfv_vector_from(self.ctx, score_bytes)
        val = int(sv.decrypt()[0])
        if val > self.max_safe_value:
            val = val - self.plain_modulus
        return val

    def server_dot_product_ct_ct(self, enc_query_bytes: bytes,
                                  enc_doc_bytes: bytes) -> bytes:
        """SERVER: E(q) · E(d) = E(q·d). Server sees nothing."""
        enc_q = ts.bfv_vector_from(self.server_ctx, enc_query_bytes)
        enc_d = ts.bfv_vector_from(self.server_ctx, enc_doc_bytes)
        enc_score = enc_q.dot(enc_d)
        return enc_score.serialize()

    def server_dot_product_ct_pt(self, enc_query_bytes: bytes,
                                  plain_doc: np.ndarray) -> bytes:
        """SERVER: E(q) · d_plain = E(q·d). Server sees doc but not query."""
        enc_q = ts.bfv_vector_from(self.server_ctx, enc_query_bytes)
        d_list = [int(x) for x in plain_doc.tolist()]
        enc_score = enc_q.dot(d_list)
        return enc_score.serialize()


# ═══════════════════════════════════════════════════════════
#  ENGINE 2: CKKS (BASELINE — approximate floats)
# ═══════════════════════════════════════════════════════════

class CKKSEngine:
    """
    CKKS Homomorphic Encryption engine for float32 vectors.

    THIS IS THE BASELINE / "ENEMY" IN YOUR PAPER.

    WHY CKKS IS WORSE FOR RAG:
      - Designed for approximate float arithmetic
      - Every operation adds noise ε
      - Need score-gap theorem to prove ranking is correct
      - Slower than BFV for equivalent security
      - Larger ciphertexts

    WHY PEOPLE USE CKKS ANYWAY:
      - It's the "default" for ML workloads
      - Works directly on float32 (no quantization step)
      - More papers published using it
    """

    def __init__(self, poly_modulus_degree: int = 8192,
                 coeff_mod_bit_sizes: List[int] = None,
                 scale_power: int = 40,
                 name: str = "CKKS"):
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.name = name
        self.poly_mod = poly_modulus_degree
        self.coeff_bits = coeff_mod_bit_sizes
        self.scale_power = scale_power
        self.ctx = None
        self.server_ctx = None
        self.keygen_time = 0.0

    def setup(self) -> 'CKKSEngine':
        """Generate CKKS encryption keys."""
        print(f"\n  🔐 Setting up CKKS encryption ({self.name}) ...")
        print(f"     poly_modulus_degree = {self.poly_mod}")
        print(f"     coeff_mod_bit_sizes = {self.coeff_bits}")
        print(f"     scale = 2^{self.scale_power}")

        t0 = time.time()

        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_mod,
            coeff_mod_bit_sizes=self.coeff_bits,
        )
        self.ctx.global_scale = 2 ** self.scale_power
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()

        pub = self.ctx.copy()
        pub.make_context_public()
        self.server_ctx = ts.context_from(pub.serialize())

        self.keygen_time = time.time() - t0

        print(f"     keygen time = {self.keygen_time:.2f}s")
        print(f"  ✅ CKKS encryption ready")

        return self

    def encrypt_vector(self, float_vector: np.ndarray) -> bytes:
        """Encrypt a single float32 vector. Returns serialized ciphertext."""
        enc = ts.ckks_vector(self.ctx, float_vector.tolist())
        return enc.serialize()

    def encrypt_documents(self, float_embeddings: np.ndarray) -> List[bytes]:
        """Encrypt all document vectors with CKKS."""
        print(f"\n  🔒 Encrypting {len(float_embeddings)} document vectors (CKKS) ...")
        t0 = time.time()

        encrypted = []
        for i, vec in enumerate(float_embeddings):
            ct_bytes = self.encrypt_vector(vec)
            encrypted.append(ct_bytes)
            if (i + 1) % 10 == 0 or i == len(float_embeddings) - 1:
                print(f"     [{i+1}/{len(float_embeddings)}]", end='\r')

        encrypt_time = time.time() - t0
        ct_size = len(encrypted[0])

        print(f"\n  ✅ CKKS documents encrypted:")
        print(f"     Time: {encrypt_time:.1f}s "
              f"({encrypt_time/len(float_embeddings)*1000:.1f} ms/vector)")
        print(f"     Ciphertext size: {ct_size:,} bytes/vector "
              f"({ct_size/1024:.0f} KB)")

        return encrypted

    def decrypt_score(self, score_bytes: bytes) -> float:
        """Decrypt a CKKS dot product result (approximate float)."""
        sv = ts.ckks_vector_from(self.ctx, score_bytes)
        return sv.decrypt()[0]

    def server_dot_product(self, enc_query_bytes: bytes,
                            enc_doc_bytes: bytes) -> bytes:
        """SERVER: E(q) · E(d) = E(q·d) approximately. Server sees nothing."""
        enc_q = ts.ckks_vector_from(self.server_ctx, enc_query_bytes)
        enc_d = ts.ckks_vector_from(self.server_ctx, enc_doc_bytes)
        enc_score = enc_q.dot(enc_d)
        return enc_score.serialize()
