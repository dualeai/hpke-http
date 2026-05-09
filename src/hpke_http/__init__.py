"""
RFC 9180 HPKE encryption library for HTTP transport.

This library provides transparent end-to-end encryption for SDK ↔ Platform communication
using RFC 9180 HPKE (Hybrid Public Key Encryption) with PSK mode. Supports DHKEM(X25519,
HKDF-SHA256) and the post-quantum hybrid X-Wing (X25519 + ML-KEM-768); clients pick the
best advertised suite per ``DEFAULT_KEM_PRIORITY``.

Usage (Server - FastAPI):
    from hpke_http.constants import KemId
    from hpke_http.middleware.fastapi import HPKEMiddleware

    app = FastAPI()
    app.add_middleware(
        HPKEMiddleware,
        private_keys={KemId.DHKEM_X25519_HKDF_SHA256: settings.hpke_private_key},
        psk_resolver=resolve_psk,
    )

Usage (Client - aiohttp):
    from hpke_http.middleware.aiohttp import HPKEClientSession

    async with HPKEClientSession(base_url="https://api.example.com", psk=api_key) as session:
        async with session.post("/tasks", json=data) as response:
            async for event in session.iter_sse(response):
                print(event)
"""

from importlib.metadata import version

from hpke_http.constants import AEAD_ID, KDF_ID, KEM_ID, MODE_PSK, VERSION, KemId
from hpke_http.core import (
    RequestDecryptor,
    RequestEncryptor,
    ResponseDecryptor,
    ResponseEncryptor,
    SSEDecryptor,
    SSEEncryptor,
    SSEEventParser,
    SSELineParser,
    is_sse_response,
)
from hpke_http.exceptions import (
    CryptoError,
    DecryptionError,
    EncryptionRequiredError,
    InvalidPSKError,
    UnsupportedKEMError,
)
from hpke_http.primitives import KEM, X25519KEM, XWingKEM, get_kem, registered_kem_ids

__all__ = [
    "AEAD_ID",
    "KDF_ID",
    "KEM",
    "KEM_ID",
    "MODE_PSK",
    "VERSION",
    "X25519KEM",
    "CryptoError",
    "DecryptionError",
    "EncryptionRequiredError",
    "InvalidPSKError",
    "KemId",
    "RequestDecryptor",
    "RequestEncryptor",
    "ResponseDecryptor",
    "ResponseEncryptor",
    "SSEDecryptor",
    "SSEEncryptor",
    "SSEEventParser",
    "SSELineParser",
    "UnsupportedKEMError",
    "XWingKEM",
    "__version__",
    "__version_full__",
    "get_kem",
    "is_sse_response",
    "registered_kem_ids",
]

__version__ = version("hpke_http")
__version_full__ = "dev"
