"""Optional finite and live SSE HTTP adapters.

Import adapters from their dependency-specific modules:

- ``hpke_http.middleware.httpx.HPKEAsyncClient``
- ``hpke_http.middleware.aiohttp.HPKEClientSession``
- ``hpke_http.middleware.fastapi.HPKEMiddleware`` for FastAPI and Starlette

The adapters use ``hpke-http/2``. Choose ``Discover()`` for one key GET per
call or ``PinnedKey`` to use a known key without a GET.
"""

from hpke_http.middleware._discovery import Discover, PinnedKey

__all__ = ["Discover", "PinnedKey"]
