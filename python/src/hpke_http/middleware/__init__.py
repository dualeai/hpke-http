"""Optional finite and live SSE HTTP adapters.

Import adapters from their dependency-specific modules:

- ``hpke_http.middleware.httpx.HPKEAsyncClient``
- ``hpke_http.middleware.aiohttp.HPKEClientSession``
- ``hpke_http.middleware.fastapi.HPKEMiddleware`` for FastAPI and Starlette

The adapters use ``hpke-http/2``. Recipient keys and PSK identities are
explicit; the adapters do not discover them.
"""
