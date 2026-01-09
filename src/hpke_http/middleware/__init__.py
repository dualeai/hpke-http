"""
Plug-and-play middleware for transparent HPKE encryption.

Server (FastAPI):
    from hpke_http.middleware.fastapi import HPKEMiddleware

Client (aiohttp):
    from hpke_http.middleware.aiohttp import HPKEClientSession

Client (httpx):
    from hpke_http.middleware.httpx import HPKEAsyncClient
"""
