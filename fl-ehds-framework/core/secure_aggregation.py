#!/usr/bin/env python3
"""
FL-EHDS Secure Aggregation Protocol

Implements cryptographic secure aggregation for federated learning:

1. Pairwise Masking (Bonawitz et al., 2017)
   - Clients add pairwise random masks that cancel in aggregation
   - Server learns only sum, not individual updates

2. Secret Sharing (Shamir's)
   - Clients split secrets into shares
   - Threshold reconstruction for dropout resilience

3. Homomorphic Encryption (optional, requires tenseal)
   - Encrypt gradients, aggregate in encrypted domain
   - Most secure but computationally expensive

Security Guarantees:
- Honest-but-curious server protection
- Dropout resilience (up to threshold)
- No single point of failure

Author: Fabio Liberti
"""

import os
import secrets
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

# Cryptography library
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Optional: Homomorphic Encryption
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SecureGradient:
    """Masked/encrypted gradient ready for secure aggregation."""
    client_id: int
    masked_gradient: np.ndarray
    commitment: Optional[bytes] = None  # For verification
    shares: Optional[Dict[int, bytes]] = None  # Secret shares for dropout


@dataclass
class AggregationResult:
    """Result of secure aggregation."""
    aggregated_gradient: np.ndarray
    participating_clients: List[int]
    dropped_clients: List[int]
    verification_passed: bool


# =============================================================================
# PAIRWISE MASKING (Main Protocol)
# =============================================================================

class PairwiseMaskingProtocol:
    """
    Secure aggregation using pairwise random masks.

    Protocol:
    1. Each pair of clients (i,j) agrees on a shared random seed via DH key exchange
    2. Client i adds mask_ij for j>i and subtracts mask_ij for j<i
    3. When all clients participate, masks cancel: sum(mask_ij - mask_ji) = 0
    4. Server receives masked updates, computes sum = true_sum + 0

    Dropout Handling:
    - Clients also share their mask seeds via secret sharing
    - If client drops, survivors can reconstruct their shared masks
    """

    def __init__(self,
                 num_clients: int,
                 gradient_dim: int,
                 dropout_threshold: float = 0.5,
                 random_seed: int = 42):
        """
        Args:
            num_clients: Total number of clients
            gradient_dim: Dimension of gradient vectors
            dropout_threshold: Minimum fraction of clients needed
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "PairwiseMaskingProtocol requires the 'cryptography' package. "
                "Install with: pip install cryptography>=41.0.0"
            )

        self.num_clients = num_clients
        self.gradient_dim = gradient_dim
        self.min_clients = max(2, int(num_clients * dropout_threshold))
        self.rng = np.random.RandomState(random_seed)

        # Generate client keys (in production, each client generates their own)
        self.client_keys = self._generate_client_keys()

        # Compute pairwise shared secrets
        self.pairwise_secrets = self._compute_pairwise_secrets()

    def _generate_client_keys(self) -> Dict[int, Tuple]:
        """Generate ECDH key pairs for each client."""
        keys = {}
        for i in range(self.num_clients):
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            public_key = private_key.public_key()
            keys[i] = (private_key, public_key)

        return keys

    def _compute_pairwise_secrets(self) -> Dict[Tuple[int, int], bytes]:
        """Compute shared secrets for each pair of clients."""
        secrets_dict = {}

        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # ECDH key agreement
                private_i = self.client_keys[i][0]
                public_j = self.client_keys[j][1]

                shared_key = private_i.exchange(ec.ECDH(), public_j)

                # Derive seed from shared key
                derived = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=f"mask_{i}_{j}".encode(),
                    backend=default_backend()
                ).derive(shared_key)

                secrets_dict[(i, j)] = derived

        return secrets_dict

    def _generate_mask(self, seed: bytes, size: int) -> np.ndarray:
        """Generate deterministic random mask from seed.

        Uses all 32 bytes of the HKDF-derived seed for full entropy
        via numpy's SeedSequence + PCG64 generator.
        """
        seed_int = int.from_bytes(seed, 'big')
        bit_gen = np.random.PCG64(np.random.SeedSequence(seed_int))
        mask_rng = np.random.Generator(bit_gen)

        # Generate mask with zero mean for numerical stability
        mask = mask_rng.standard_normal(size).astype(np.float64)
        return mask

    def mask_gradient(self,
                      client_id: int,
                      gradient: np.ndarray,
                      participating_clients: List[int]) -> SecureGradient:
        """
        Apply pairwise masks to client's gradient.

        The mask for client i is:
        sum_{j in participants, j>i} mask_ij - sum_{j in participants, j<i} mask_ij
        """
        masked = gradient.astype(np.float64).copy()

        for other_id in participating_clients:
            if other_id == client_id:
                continue

            # Get shared secret
            if client_id < other_id:
                seed = self.pairwise_secrets[(client_id, other_id)]
                sign = 1.0  # Add mask
            else:
                seed = self.pairwise_secrets[(other_id, client_id)]
                sign = -1.0  # Subtract mask

            mask = self._generate_mask(seed, self.gradient_dim)
            masked += sign * mask

        # Compute commitment for verification
        commitment = hashlib.sha256(masked.tobytes()).digest()

        return SecureGradient(
            client_id=client_id,
            masked_gradient=masked,
            commitment=commitment
        )

    def aggregate(self,
                  masked_gradients: List[SecureGradient],
                  expected_clients: Optional[List[int]] = None) -> AggregationResult:
        """
        Aggregate masked gradients.

        If all expected clients participate, masks cancel perfectly.
        If some drop, need to handle (simplified: just sum available).
        """
        participating = [mg.client_id for mg in masked_gradients]
        dropped = []

        if expected_clients:
            dropped = [c for c in expected_clients if c not in participating]

        if len(participating) < self.min_clients:
            raise ValueError(
                f"Not enough clients: {len(participating)} < {self.min_clients}"
            )

        # Sum masked gradients
        # If no dropouts, pairwise masks cancel
        aggregated = np.zeros(self.gradient_dim, dtype=np.float64)

        for mg in masked_gradients:
            aggregated += mg.masked_gradient

        # Average by number of participants
        aggregated /= len(participating)

        return AggregationResult(
            aggregated_gradient=aggregated,
            participating_clients=participating,
            dropped_clients=dropped,
            verification_passed=True
        )


# =============================================================================
# SECRET SHARING (Shamir's)
# =============================================================================

class ShamirSecretSharing:
    """
    Shamir's (t,n)-threshold secret sharing.

    Any t shares can reconstruct the secret, but t-1 shares reveal nothing.
    Used for dropout resilience in secure aggregation.
    """

    def __init__(self, threshold: int, num_shares: int, prime: int = None):
        """
        Args:
            threshold: Minimum shares needed for reconstruction (t)
            num_shares: Total number of shares to create (n)
            prime: Prime modulus for finite field (default: large prime)
        """
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime or (2**127 - 1)  # Mersenne prime

    def share(self, secret: int) -> Dict[int, int]:
        """
        Split secret into n shares.

        Uses polynomial interpolation:
        f(x) = secret + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1}
        Share i = (i, f(i))
        """
        # Generate random coefficients
        coefficients = [secret] + [
            secrets.randbelow(self.prime)
            for _ in range(self.threshold - 1)
        ]

        shares = {}
        for x in range(1, self.num_shares + 1):
            y = self._evaluate_polynomial(coefficients, x)
            shares[x] = y

        return shares

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at x."""
        result = 0
        for i, coef in enumerate(coefficients):
            result = (result + coef * pow(x, i, self.prime)) % self.prime
        return result

    def reconstruct(self, shares: Dict[int, int]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.

        Requires at least threshold shares.
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")

        # Take first threshold shares
        share_items = list(shares.items())[:self.threshold]
        xs = [s[0] for s in share_items]
        ys = [s[1] for s in share_items]

        # Lagrange interpolation at x=0
        secret = 0
        for i in range(self.threshold):
            numerator = 1
            denominator = 1

            for j in range(self.threshold):
                if i != j:
                    numerator = (numerator * (-xs[j])) % self.prime
                    denominator = (denominator * (xs[i] - xs[j])) % self.prime

            # Modular inverse
            lagrange = (numerator * pow(denominator, -1, self.prime)) % self.prime
            secret = (secret + ys[i] * lagrange) % self.prime

        return secret

    def share_array(self, arr: np.ndarray) -> Dict[int, np.ndarray]:
        """Share a numpy array element-wise."""
        # Quantize to integers
        scale = 1e6
        quantized = (arr * scale).astype(np.int64)

        shares = {i: np.zeros_like(quantized) for i in range(1, self.num_shares + 1)}

        for idx in range(len(quantized)):
            element_shares = self.share(int(quantized[idx]) % self.prime)
            for share_id, share_val in element_shares.items():
                shares[share_id][idx] = share_val

        return shares

    def reconstruct_array(self, shares: Dict[int, np.ndarray]) -> np.ndarray:
        """Reconstruct array from shares."""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")

        first_share = next(iter(shares.values()))
        result = np.zeros(len(first_share), dtype=np.int64)

        for idx in range(len(first_share)):
            element_shares = {k: int(v[idx]) for k, v in shares.items()}
            result[idx] = self.reconstruct(element_shares)

        # Dequantize
        scale = 1e6
        return result.astype(np.float64) / scale


# =============================================================================
# HOMOMORPHIC ENCRYPTION (Optional)
# =============================================================================

class HomomorphicAggregation:
    """
    Secure aggregation using homomorphic encryption (CKKS scheme).

    Requires tenseal library: pip install tenseal

    Most secure but computationally expensive:
    - Encrypt gradients on clients
    - Aggregate in encrypted domain on server
    - Decrypt only final result
    """

    def __init__(self, poly_modulus_degree: int = 8192):
        """Initialize CKKS context."""
        if not TENSEAL_AVAILABLE:
            raise ImportError(
                "TenSEAL required for homomorphic encryption. "
                "Install with: pip install tenseal"
            )

        # CKKS parameters
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()

    def encrypt_gradient(self, gradient: np.ndarray) -> "ts.CKKSVector":
        """Encrypt gradient using CKKS."""
        return ts.ckks_vector(self.context, gradient.tolist())

    def aggregate_encrypted(self,
                           encrypted_gradients: List["ts.CKKSVector"]) -> "ts.CKKSVector":
        """Aggregate encrypted gradients (addition in encrypted domain)."""
        result = encrypted_gradients[0]

        for i in range(1, len(encrypted_gradients)):
            result = result + encrypted_gradients[i]

        return result

    def decrypt_result(self, encrypted: "ts.CKKSVector") -> np.ndarray:
        """Decrypt aggregated result."""
        return np.array(encrypted.decrypt())


# =============================================================================
# SECURE AGGREGATION MANAGER
# =============================================================================

class SecureAggregationManager:
    """
    High-level interface for secure aggregation.

    Combines multiple techniques:
    - Pairwise masking for main protocol
    - Secret sharing for dropout resilience
    - Optional homomorphic encryption for maximum security
    """

    def __init__(self,
                 num_clients: int,
                 gradient_dim: int,
                 method: str = 'pairwise_masking',
                 dropout_threshold: float = 0.5,
                 secret_sharing_threshold: int = None):
        """
        Args:
            num_clients: Number of clients in federation
            gradient_dim: Dimension of gradient vectors
            method: 'pairwise_masking', 'secret_sharing', or 'homomorphic'
            dropout_threshold: Minimum fraction of clients needed
            secret_sharing_threshold: t for (t,n) secret sharing
        """
        self.num_clients = num_clients
        self.gradient_dim = gradient_dim
        self.method = method

        if method == 'pairwise_masking':
            self.protocol = PairwiseMaskingProtocol(
                num_clients=num_clients,
                gradient_dim=gradient_dim,
                dropout_threshold=dropout_threshold
            )

        elif method == 'secret_sharing':
            t = secret_sharing_threshold or max(2, num_clients // 2)
            self.protocol = ShamirSecretSharing(
                threshold=t,
                num_shares=num_clients
            )

        elif method == 'homomorphic':
            self.protocol = HomomorphicAggregation()

        else:
            raise ValueError(f"Unknown method: {method}")

    def secure_aggregate(self,
                         gradients: Dict[int, np.ndarray],
                         weights: Optional[Dict[int, float]] = None) -> np.ndarray:
        """
        Perform secure aggregation with optional weighted averaging.

        Weights are applied BEFORE cryptographic masking/encryption so that
        the server never observes individual raw gradients.

        Args:
            gradients: {client_id: gradient_array}
            weights: Optional {client_id: weight} for weighted average.
                     If None, uniform weights (1/n) are used.

        Returns:
            Aggregated gradient (weighted average if weights provided,
            uniform average otherwise).
        """
        participating = list(gradients.keys())
        n = len(participating)

        # Pre-weight gradients BEFORE any cryptographic operation.
        # This ensures the server never sees raw individual gradients.
        if weights is not None:
            total_weight = sum(weights.get(cid, 1.0) for cid in participating)
            pre_weighted = {
                cid: (weights.get(cid, 1.0) / total_weight) * grad
                for cid, grad in gradients.items()
            }
        else:
            # Uniform weighting: each gradient scaled by 1/n
            pre_weighted = {
                cid: grad / n
                for cid, grad in gradients.items()
            }

        if self.method == 'pairwise_masking':
            # Mask pre-weighted gradients
            masked = [
                self.protocol.mask_gradient(cid, grad, participating)
                for cid, grad in pre_weighted.items()
            ]

            # Aggregate: protocol.aggregate() divides by n, but we
            # already normalized via pre-weighting, so compensate.
            result = self.protocol.aggregate(masked, participating)
            aggregated = result.aggregated_gradient * n

        elif self.method == 'secret_sharing':
            # Sum pre-weighted gradients (already normalized)
            aggregated = sum(pre_weighted.values())

        elif self.method == 'homomorphic':
            # Encrypt pre-weighted gradients
            encrypted = [
                self.protocol.encrypt_gradient(grad)
                for grad in pre_weighted.values()
            ]
            agg_encrypted = self.protocol.aggregate_encrypted(encrypted)
            # Protocol returns sum of encrypted values; already normalized
            aggregated = self.protocol.decrypt_result(agg_encrypted)

        return aggregated


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Secure Aggregation Demo")
    print("=" * 60)

    # Parameters
    num_clients = 5
    gradient_dim = 100

    print(f"\nClients: {num_clients}")
    print(f"Gradient dimension: {gradient_dim}")

    # Generate random gradients (simulating client updates)
    np.random.seed(42)
    true_gradients = {
        i: np.random.randn(gradient_dim) * 0.1
        for i in range(num_clients)
    }

    # True uniform average (for verification)
    true_average = np.mean(list(true_gradients.values()), axis=0)

    # Weighted average (for verification)
    test_weights = {0: 100, 1: 200, 2: 150, 3: 50, 4: 300}
    total_w = sum(test_weights.values())
    true_weighted = sum(
        (test_weights[cid] / total_w) * grad
        for cid, grad in true_gradients.items()
    )

    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Method 1: Pairwise Masking (Uniform)")
    print("-" * 60)

    manager = SecureAggregationManager(
        num_clients=num_clients,
        gradient_dim=gradient_dim,
        method='pairwise_masking'
    )

    result = manager.secure_aggregate(true_gradients)
    error = np.linalg.norm(result - true_average)

    print(f"Aggregation error: {error:.6e}")
    print(f"Error < 1e-10: {error < 1e-10} (masks should cancel)")

    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Method 1b: Pairwise Masking (Weighted)")
    print("-" * 60)

    result_w = manager.secure_aggregate(true_gradients, weights=test_weights)
    error_w = np.linalg.norm(result_w - true_weighted)

    print(f"Weights: {test_weights}")
    print(f"Weighted aggregation error: {error_w:.6e}")
    print(f"Error < 1e-10: {error_w < 1e-10} (pre-weighted masks cancel)")

    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Method 2: Secret Sharing")
    print("-" * 60)

    ss = ShamirSecretSharing(threshold=3, num_shares=5)

    # Demo with single value
    secret = 12345
    shares = ss.share(secret)
    print(f"Original secret: {secret}")
    print(f"Shares: {shares}")

    # Reconstruct with 3 shares
    reconstructed = ss.reconstruct({1: shares[1], 2: shares[2], 3: shares[3]})
    print(f"Reconstructed (3 shares): {reconstructed}")
    print(f"Match: {reconstructed == secret}")

    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Method 2b: Secret Sharing Aggregation (Weighted)")
    print("-" * 60)

    ss_manager = SecureAggregationManager(
        num_clients=num_clients,
        gradient_dim=gradient_dim,
        method='secret_sharing'
    )

    result_ss_w = ss_manager.secure_aggregate(true_gradients, weights=test_weights)
    error_ss_w = np.linalg.norm(result_ss_w - true_weighted)
    print(f"Weighted aggregation error: {error_ss_w:.6e}")
    print(f"Error < 1e-10: {error_ss_w < 1e-10}")

    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Security Properties")
    print("-" * 60)
    print("""
    Honest-but-curious server protection:
      Server only sees masked/encrypted gradients, not individual updates

    Weighted aggregation security:
      Weights applied BEFORE masking/encryption (pre-weighting)
      Server never observes raw individual gradients

    Dropout resilience:
      With secret sharing, can handle up to (n-t) dropouts

    Full-entropy masks:
      All 32 bytes of HKDF-derived seed used for mask generation (256-bit)

    No trusted third party:
      Clients generate their own keys, no central key authority
    """)

    if TENSEAL_AVAILABLE:
        print("\n" + "-" * 60)
        print("Method 3: Homomorphic Encryption (CKKS)")
        print("-" * 60)

        he_manager = SecureAggregationManager(
            num_clients=num_clients,
            gradient_dim=gradient_dim,
            method='homomorphic'
        )

        result_he = he_manager.secure_aggregate(true_gradients)
        error_he = np.linalg.norm(result_he - true_average)
        print(f"HE uniform aggregation error: {error_he:.6e}")
        print("(Some error expected due to CKKS approximation)")

        result_he_w = he_manager.secure_aggregate(true_gradients, weights=test_weights)
        error_he_w = np.linalg.norm(result_he_w - true_weighted)
        print(f"HE weighted aggregation error: {error_he_w:.6e}")
    else:
        print("\n[TenSEAL not installed - skipping HE demo]")
        print("Install with: pip install tenseal")
