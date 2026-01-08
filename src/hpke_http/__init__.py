"""
RFC 9180 HPKE encryption library for HTTP transport.

This library provides transparent end-to-end encryption for SDK ↔ Platform communication
using RFC 9180 HPKE (Hybrid Public Key Encryption) with PSK mode.

Usage (Server - FastAPI):
    from hpke_http.middleware.fastapi import HPKEMiddleware

    app = FastAPI()
    app.add_middleware(HPKEMiddleware, private_key=settings.hpke_private_key)

Usage (Client - aiohttp):
    from hpke_http.middleware.aiohttp import HPKEClientSession

    async with HPKEClientSession(base_url="https://api.example.com", psk=api_key) as session:
        async with session.post("/tasks", json=data) as response:
            async for event in session.iter_sse(response):
                print(event)
"""

from importlib.metadata import version

from hpke_http.constants import AEAD_ID, KDF_ID, KEM_ID, MODE_PSK, VERSION
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
)

__all__ = [
    "AEAD_ID",
    "KDF_ID",
    "KEM_ID",
    "MODE_PSK",
    "VERSION",
    "CryptoError",
    "DecryptionError",
    "EncryptionRequiredError",
    "InvalidPSKError",
    "RequestDecryptor",
    "RequestEncryptor",
    "ResponseDecryptor",
    "ResponseEncryptor",
    "SSEDecryptor",
    "SSEEncryptor",
    "SSEEventParser",
    "SSELineParser",
    "__version__",
    "__version_full__",
    "is_sse_response",
]

__version__ = version("hpke_http")
__version_full__ = "dev"
