"""Optional finite and live SSE HTTP adapters.

Import adapters from their dependency-specific modules:

- ``hpke_http.middleware.httpx.HPKEAsyncClient``
- ``hpke_http.middleware.aiohttp.HPKEClientSession``
- ``hpke_http.middleware.fastapi.HPKEMiddleware`` for FastAPI and Starlette

The adapters use ``hpke-http/3``. Share a discovered endpoint across clients,
or use ``PinnedKey`` when the public key is known.
"""

from hpke_http.middleware._discovery import PinnedKey
from hpke_http.middleware._shared_key import KeyLease

__all__ = ["KeyLease", "PinnedKey"]
