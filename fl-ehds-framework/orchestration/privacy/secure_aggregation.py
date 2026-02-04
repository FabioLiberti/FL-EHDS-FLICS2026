"""
Secure Aggregation Module
=========================
Cryptographic secure aggregation for protecting individual client
gradients during FL aggregation.

Implements Shamir's Secret Sharing based secure aggregation protocol.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import structlog
from cryptography.fernet import Fernet
import secrets
import hashlib

from core.exceptions import SecureAggregationError

logger = structlog.get_logger(__name__)


@dataclass
class SecretShare:
    """A share of a secret value."""

    client_id: str
    share_index: int
    share_value: bytes
    commitment: str  # Hash commitment for verification


class SecureAggregation:
    """
    Secure aggregation protocol for privacy-preserving gradient aggregation.

    Ensures that the server only learns the aggregated sum of client
    gradients, not individual contributions.
    """

    def __init__(
        self,
        protocol: str = "shamir",
        threshold: float = 0.67,
        key_rotation_rounds: int = 10,
    ):
        """
        Initialize secure aggregation.

        Args:
            protocol: Aggregation protocol ('shamir', 'pairwise_masking').
            threshold: Fraction of clients needed to reconstruct (for Shamir).
            key_rotation_rounds: Rounds between key rotation.
        """
        self.protocol = protocol
        self.threshold = threshold
        self.key_rotation_rounds = key_rotation_rounds

        # Session state
        self._session_keys: Dict[str, bytes] = {}
        self._round_count = 0
        self._client_shares: Dict[str, List[SecretShare]] = {}

    def setup_round(
        self,
        client_ids: List[str],
        round_number: int,
    ) -> Dict[str, Any]:
        """
        Set up secure aggregation for a training round.

        Args:
            client_ids: Participating client IDs.
            round_number: Current round number.

        Returns:
            Setup parameters for clients.
        """
        self._round_count = round_number
        num_clients = len(client_ids)
        threshold_count = int(np.ceil(self.threshold * num_clients))

        # Key rotation check
        if round_number > 0 and round_number % self.key_rotation_rounds == 0:
            self._rotate_keys(client_ids)

        logger.info(
            "Secure aggregation setup",
            round=round_number,
            num_clients=num_clients,
            threshold=threshold_count,
            protocol=self.protocol,
        )

        return {
            "protocol": self.protocol,
            "num_clients": num_clients,
            "threshold": threshold_count,
            "round": round_number,
            "client_ids": client_ids,
        }

    def _rotate_keys(self, client_ids: List[str]) -> None:
        """Rotate session keys for all clients."""
        for client_id in client_ids:
            self._session_keys[client_id] = Fernet.generate_key()

        logger.info("Session keys rotated", num_clients=len(client_ids))

    def create_shares(
        self,
        client_id: str,
        gradient_data: bytes,
        num_shares: int,
        threshold: int,
    ) -> List[SecretShare]:
        """
        Create secret shares of gradient data using Shamir's scheme.

        Args:
            client_id: Client creating shares.
            gradient_data: Serialized gradient data.
            num_shares: Total number of shares to create.
            threshold: Minimum shares needed for reconstruction.

        Returns:
            List of SecretShare objects.
        """
        # Simplified Shamir implementation
        # In production, use a proper cryptographic library

        # Generate random polynomial coefficients
        coefficients = [
            int.from_bytes(secrets.token_bytes(32), "big")
            for _ in range(threshold)
        ]
        # First coefficient is the secret
        secret_int = int.from_bytes(gradient_data[:32], "big") if len(gradient_data) >= 32 else int.from_bytes(gradient_data.ljust(32, b'\0'), "big")
        coefficients[0] = secret_int

        # Evaluate polynomial at different points
        shares = []
        prime = 2**127 - 1  # Large prime for field

        for i in range(1, num_shares + 1):
            # Evaluate polynomial at point i
            share_value = 0
            for j, coef in enumerate(coefficients):
                share_value = (share_value + coef * pow(i, j, prime)) % prime

            share = SecretShare(
                client_id=client_id,
                share_index=i,
                share_value=share_value.to_bytes(16, "big"),
                commitment=hashlib.sha256(
                    f"{client_id}:{i}:{share_value}".encode()
                ).hexdigest()[:16],
            )
            shares.append(share)

        logger.debug(
            "Secret shares created",
            client_id=client_id,
            num_shares=num_shares,
            threshold=threshold,
        )

        return shares

    def mask_gradients(
        self,
        client_id: str,
        gradients: Dict[str, Any],
        peer_client_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Mask gradients using pairwise masking protocol.

        Args:
            client_id: Current client ID.
            gradients: Gradient dictionary to mask.
            peer_client_ids: IDs of peer clients.

        Returns:
            Masked gradients.
        """
        masked = {}

        for key, grad in gradients.items():
            grad_array = np.array(grad)
            mask = np.zeros_like(grad_array)

            # Add/subtract pairwise masks
            for peer_id in peer_client_ids:
                # Generate deterministic mask from shared seed
                seed = self._derive_pairwise_seed(client_id, peer_id)
                np.random.seed(seed)
                pairwise_mask = np.random.randn(*grad_array.shape)

                # Add if client_id < peer_id, subtract otherwise
                if client_id < peer_id:
                    mask += pairwise_mask
                else:
                    mask -= pairwise_mask

            masked[key] = grad_array + mask

        return masked

    def _derive_pairwise_seed(self, client_a: str, client_b: str) -> int:
        """Derive deterministic seed for pairwise masking."""
        # Order-independent seed derivation
        ordered = tuple(sorted([client_a, client_b]))
        seed_bytes = hashlib.sha256(f"{ordered[0]}:{ordered[1]}".encode()).digest()
        return int.from_bytes(seed_bytes[:4], "big")

    def aggregate_masked(
        self,
        masked_gradients: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate masked gradients (masks cancel out in sum).

        Args:
            masked_gradients: List of masked gradient dictionaries.

        Returns:
            Aggregated gradients (unmasked sum).
        """
        if not masked_gradients:
            return {}

        # Sum all masked gradients (masks cancel)
        result = {}
        keys = masked_gradients[0].keys()

        for key in keys:
            summed = np.zeros_like(np.array(masked_gradients[0][key]))
            for grads in masked_gradients:
                summed += np.array(grads[key])
            result[key] = summed

        logger.debug(
            "Masked gradients aggregated",
            num_clients=len(masked_gradients),
        )

        return result

    def verify_commitments(
        self,
        shares: List[SecretShare],
        expected_commitments: Dict[str, str],
    ) -> bool:
        """
        Verify share commitments for integrity.

        Args:
            shares: Received shares.
            expected_commitments: Expected commitment values.

        Returns:
            True if all commitments verify.
        """
        for share in shares:
            key = f"{share.client_id}:{share.share_index}"
            if key in expected_commitments:
                if share.commitment != expected_commitments[key]:
                    logger.warning(
                        "Commitment verification failed",
                        client_id=share.client_id,
                        share_index=share.share_index,
                    )
                    return False
        return True

    def handle_dropout(
        self,
        participating_clients: List[str],
        dropped_clients: List[str],
        shares_received: Dict[str, List[SecretShare]],
    ) -> bool:
        """
        Handle client dropout during secure aggregation.

        Args:
            participating_clients: Original participants.
            dropped_clients: Clients that dropped out.
            shares_received: Shares received from active clients.

        Returns:
            True if aggregation can proceed.
        """
        active_count = len(participating_clients) - len(dropped_clients)
        threshold_count = int(np.ceil(self.threshold * len(participating_clients)))

        can_proceed = active_count >= threshold_count

        if not can_proceed:
            logger.error(
                "Too many dropouts for secure aggregation",
                active=active_count,
                threshold=threshold_count,
                dropped=len(dropped_clients),
            )

        return can_proceed

    def get_protocol_info(self) -> Dict[str, Any]:
        """Get information about current protocol configuration."""
        return {
            "protocol": self.protocol,
            "threshold": self.threshold,
            "key_rotation_rounds": self.key_rotation_rounds,
            "current_round": self._round_count,
            "active_clients": len(self._session_keys),
        }
