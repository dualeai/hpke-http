"""Optional buffered HTTP adapters.

Import adapters from their dependency-specific modules:

- ``hpke_http.middleware.httpx.HPKEAsyncClient``
- ``hpke_http.middleware.aiohttp.HPKEClientSession``
- ``hpke_http.middleware.fastapi.HPKEMiddleware`` for FastAPI and Starlette

No adapter performs key discovery or incremental streaming. Recipient keys and
PSK identities are explicit, and protocol version 1 authenticates complete
bounded messages.
"""
