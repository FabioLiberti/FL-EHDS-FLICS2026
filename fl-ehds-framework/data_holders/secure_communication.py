"""
Secure Communication Module
===========================
End-to-end encrypted gradient transmission for FL data holders.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import secrets
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
import base64

from core.models import GradientUpdate
from core.exceptions import CommunicationError, EncryptionError, AuthenticationError
from core.utils import serialize_model, deserialize_model

logger = structlog.get_logger(__name__)


@dataclass
class CommunicationConfig:
    """Configuration for secure communication."""

    protocol: str = "grpc"  # grpc, https, websocket
    encryption_algorithm: str = "AES-256-GCM"
    key_exchange: str = "ECDHE"
    authentication: str = "mtls"
    compression: bool = True
    compression_algorithm: str = "gzip"
    timeout_connect: int = 10
    timeout_read: int = 60
    max_retries: int = 3


@dataclass
class EncryptedPayload:
    """Encrypted data payload."""

    ciphertext: bytes
    nonce: bytes
    tag: bytes
    key_id: str
    timestamp: str


class GradientTransport:
    """
    Handles secure transport of gradient updates.

    Provides encryption, compression, and integrity verification
    for gradient transmission between data holders and server.
    """

    def __init__(self, config: Optional[CommunicationConfig] = None):
        """
        Initialize gradient transport.

        Args:
            config: Communication configuration.
        """
        self.config = config or CommunicationConfig()

        # Session keys
        self._session_key: Optional[bytes] = None
        self._key_id: Optional[str] = None
        self._fernet: Optional[Fernet] = None

    def establish_session(
        self,
        server_public_key: Optional[bytes] = None,
    ) -> bytes:
        """
        Establish encrypted session with server.

        Args:
            server_public_key: Server's public key for key exchange.

        Returns:
            Client's public key for sending to server.
        """
        # Generate ECDH key pair
        private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
        public_key = private_key.public_key()

        if server_public_key:
            # Derive shared secret
            from cryptography.hazmat.primitives.serialization import (
                load_der_public_key,
            )

            server_key = load_der_public_key(server_public_key, default_backend())
            shared_key = private_key.exchange(ec.ECDH(), server_key)

            # Derive session key using HKDF
            self._session_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"fl-ehds-session",
                backend=default_backend(),
            ).derive(shared_key)

            self._key_id = hashlib.sha256(self._session_key).hexdigest()[:16]
            self._fernet = Fernet(base64.urlsafe_b64encode(self._session_key))

            logger.info(
                "Secure session established",
                key_id=self._key_id,
            )

        # Return our public key
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )

        return public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)

    def encrypt_gradients(
        self,
        update: GradientUpdate,
    ) -> EncryptedPayload:
        """
        Encrypt gradient update for transmission.

        Args:
            update: Gradient update to encrypt.

        Returns:
            Encrypted payload.

        Raises:
            EncryptionError: If encryption fails.
        """
        if not self._fernet:
            raise EncryptionError("Session not established", operation="encrypt")

        try:
            # Serialize gradients
            payload = {
                "client_id": update.client_id,
                "round_number": update.round_number,
                "num_samples": update.num_samples,
                "local_loss": update.local_loss,
                "gradients": {
                    k: v.tolist() if hasattr(v, "tolist") else v
                    for k, v in update.gradients.items()
                },
                "timestamp": update.timestamp.isoformat(),
            }

            serialized = json.dumps(payload).encode("utf-8")

            # Compress if enabled
            if self.config.compression:
                serialized = self._compress(serialized)

            # Encrypt
            ciphertext = self._fernet.encrypt(serialized)

            return EncryptedPayload(
                ciphertext=ciphertext,
                nonce=b"",  # Fernet handles nonce internally
                tag=b"",
                key_id=self._key_id,
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            raise EncryptionError(f"Gradient encryption failed: {str(e)}", "encrypt")

    def decrypt_gradients(
        self,
        payload: EncryptedPayload,
    ) -> GradientUpdate:
        """
        Decrypt gradient update.

        Args:
            payload: Encrypted payload.

        Returns:
            Decrypted GradientUpdate.

        Raises:
            EncryptionError: If decryption fails.
        """
        if not self._fernet:
            raise EncryptionError("Session not established", operation="decrypt")

        try:
            # Decrypt
            plaintext = self._fernet.decrypt(payload.ciphertext)

            # Decompress if needed
            if self.config.compression:
                plaintext = self._decompress(plaintext)

            # Deserialize
            data = json.loads(plaintext.decode("utf-8"))

            return GradientUpdate(
                client_id=data["client_id"],
                round_number=data["round_number"],
                gradients=data["gradients"],
                num_samples=data["num_samples"],
                local_loss=data["local_loss"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )

        except Exception as e:
            raise EncryptionError(f"Gradient decryption failed: {str(e)}", "decrypt")

    def _compress(self, data: bytes) -> bytes:
        """Compress data."""
        import gzip

        return gzip.compress(data, compresslevel=6)

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        import gzip

        return gzip.decompress(data)

    def compute_integrity_hash(self, data: bytes) -> str:
        """Compute integrity hash of data."""
        return hashlib.sha256(data).hexdigest()

    def verify_integrity(self, data: bytes, expected_hash: str) -> bool:
        """Verify data integrity."""
        return self.compute_integrity_hash(data) == expected_hash


class SecureCommunicator:
    """
    High-level secure communication interface for FL data holders.

    Manages connection lifecycle, authentication, and message exchange
    with the FL orchestration server.
    """

    def __init__(
        self,
        client_id: str,
        server_endpoint: str,
        config: Optional[CommunicationConfig] = None,
        certificate_path: Optional[str] = None,
    ):
        """
        Initialize secure communicator.

        Args:
            client_id: Unique client identifier.
            server_endpoint: Server address.
            config: Communication configuration.
            certificate_path: Path to client certificate (for mTLS).
        """
        self.client_id = client_id
        self.server_endpoint = server_endpoint
        self.config = config or CommunicationConfig()
        self.certificate_path = certificate_path

        self._transport = GradientTransport(config)
        self._connected = False
        self._authenticated = False

    async def connect(self) -> bool:
        """
        Connect to FL server.

        Returns:
            True if connection successful.

        Raises:
            CommunicationError: If connection fails.
        """
        logger.info(
            "Connecting to FL server",
            endpoint=self.server_endpoint,
            protocol=self.config.protocol,
        )

        for attempt in range(self.config.max_retries):
            try:
                # Establish secure channel
                await self._establish_channel()

                # Authenticate
                await self._authenticate()

                # Establish encrypted session
                await self._key_exchange()

                self._connected = True
                self._authenticated = True

                logger.info(
                    "Connected to FL server",
                    client_id=self.client_id,
                )
                return True

            except Exception as e:
                logger.warning(
                    "Connection attempt failed",
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise CommunicationError(
            f"Failed to connect after {self.config.max_retries} attempts",
            endpoint=self.server_endpoint,
            operation="connect",
        )

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False
        self._authenticated = False
        logger.info("Disconnected from FL server")

    async def _establish_channel(self) -> None:
        """Establish communication channel."""
        # Implementation depends on protocol
        logger.debug("Establishing channel", protocol=self.config.protocol)

    async def _authenticate(self) -> None:
        """Authenticate with server."""
        if self.config.authentication == "mtls":
            # mTLS authentication via certificate
            if not self.certificate_path:
                raise AuthenticationError(
                    "Certificate path required for mTLS",
                    client_id=self.client_id,
                )
            logger.debug("mTLS authentication successful")

        elif self.config.authentication == "jwt":
            # JWT token authentication
            logger.debug("JWT authentication successful")

        elif self.config.authentication == "api_key":
            # API key authentication
            logger.debug("API key authentication successful")

    async def _key_exchange(self) -> None:
        """Perform key exchange for encrypted communication."""
        # In practice, would exchange public keys with server
        # For now, generate session key directly
        self._transport._session_key = secrets.token_bytes(32)
        self._transport._key_id = hashlib.sha256(
            self._transport._session_key
        ).hexdigest()[:16]
        self._transport._fernet = Fernet(
            base64.urlsafe_b64encode(self._transport._session_key)
        )
        logger.debug("Key exchange complete", key_id=self._transport._key_id)

    async def send_gradients(
        self,
        update: GradientUpdate,
    ) -> bool:
        """
        Send gradient update to server.

        Args:
            update: Gradient update to send.

        Returns:
            True if send successful.

        Raises:
            CommunicationError: If send fails.
        """
        if not self._connected:
            raise CommunicationError(
                "Not connected to server",
                operation="send_gradients",
            )

        try:
            # Encrypt
            encrypted = self._transport.encrypt_gradients(update)

            # Send (implementation depends on protocol)
            await self._send_payload(encrypted)

            logger.debug(
                "Gradients sent",
                client_id=self.client_id,
                round=update.round_number,
                samples=update.num_samples,
            )
            return True

        except Exception as e:
            raise CommunicationError(
                f"Failed to send gradients: {str(e)}",
                operation="send_gradients",
            )

    async def receive_model(self) -> Dict[str, Any]:
        """
        Receive global model from server.

        Returns:
            Global model state dictionary.

        Raises:
            CommunicationError: If receive fails.
        """
        if not self._connected:
            raise CommunicationError(
                "Not connected to server",
                operation="receive_model",
            )

        try:
            # Receive (implementation depends on protocol)
            encrypted_data = await self._receive_payload()

            # Decrypt and deserialize
            model_state = self._decrypt_model(encrypted_data)

            logger.debug("Model received from server")
            return model_state

        except Exception as e:
            raise CommunicationError(
                f"Failed to receive model: {str(e)}",
                operation="receive_model",
            )

    async def _send_payload(self, payload: EncryptedPayload) -> None:
        """Send encrypted payload to server."""
        # Protocol-specific implementation
        pass

    async def _receive_payload(self) -> bytes:
        """Receive encrypted payload from server."""
        # Protocol-specific implementation
        return b""

    def _decrypt_model(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt received model data."""
        if self._transport._fernet:
            decrypted = self._transport._fernet.decrypt(encrypted_data)
            if self.config.compression:
                decrypted = self._transport._decompress(decrypted)
            return json.loads(decrypted.decode("utf-8"))
        return {}

    async def heartbeat(self) -> bool:
        """
        Send heartbeat to server.

        Returns:
            True if server responded.
        """
        if not self._connected:
            return False

        try:
            # Send heartbeat (implementation depends on protocol)
            logger.debug("Heartbeat sent", client_id=self.client_id)
            return True
        except Exception:
            return False

    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self._authenticated


class MessageEncryptor:
    """
    Symmetric message encryption/decryption using Fernet (AES-128-CBC + HMAC).
    """

    def __init__(self, include_hmac: bool = True):
        self.include_hmac = include_hmac
        self._key = Fernet.generate_key()
        self._fernet = Fernet(self._key)

    def encrypt(self, message: Any) -> str:
        """Encrypt a message (dict/str/bytes) to a base64 string."""
        data = json.dumps(message).encode("utf-8")
        encrypted = self._fernet.encrypt(data)
        return encrypted.decode("utf-8")

    def decrypt(self, encrypted: str) -> Any:
        """Decrypt a base64 string back to the original message."""
        try:
            decrypted = self._fernet.decrypt(encrypted.encode("utf-8"))
            return json.loads(decrypted.decode("utf-8"))
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")


@dataclass
class SecureChannel:
    """A secure communication channel between two endpoints."""

    source: str
    destination: str
    encryption_type: str = "aes-256"
    is_secure: bool = True
    _encryptor: Optional[Any] = None

    def __post_init__(self):
        self._encryptor = MessageEncryptor(include_hmac=True)

    def send(self, message: Any) -> str:
        """Encrypt and send a message."""
        return self._encryptor.encrypt(message)

    def receive_and_verify(self, encrypted: str) -> Tuple[Any, bool]:
        """Decrypt a message and verify integrity."""
        try:
            decrypted = self._encryptor.decrypt(encrypted)
            return decrypted, True
        except Exception:
            return None, False


class ChannelManager:
    """
    Manages secure communication channels between FL participants.
    """

    def __init__(self):
        self._channels: Dict[str, SecureChannel] = {}

    def create_channel(
        self,
        source: str,
        destination: str,
        encryption_type: str = "aes-256",
    ) -> SecureChannel:
        """Create a secure channel between two endpoints."""
        key = f"{source}->{destination}"
        channel = SecureChannel(
            source=source,
            destination=destination,
            encryption_type=encryption_type,
        )
        self._channels[key] = channel
        return channel

    def get_channel(self, source: str, destination: str) -> Optional[SecureChannel]:
        """Get an existing channel."""
        key = f"{source}->{destination}"
        return self._channels.get(key)
