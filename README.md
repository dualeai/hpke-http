# hpke-http

End-to-end encryption for HTTP APIs.

```bash
uv add git+https://github.com/duale-ai/hpke-http
```

## Highlights

- **Transparent** - Drop-in middleware, no application code changes
- **E2E encryption** - Protects data even with TLS termination at CDN/LB
- **PSK binding** - Each request cryptographically bound to API key
- **Replay protection** - SSE counter prevents replay attacks
- **RFC 9180 compliant** - Auditable, interoperable standard

## Quick Start

### Server (FastAPI)

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.constants import KemId

app = FastAPI()

async def resolve_psk(scope: dict) -> tuple[bytes, bytes]:
    api_key = dict(scope["headers"]).get(b"authorization", b"").decode()
    return (api_key.encode(), (await lookup_tenant(api_key)).encode())

app.add_middleware(
    HPKEMiddleware,
    private_keys={KemId.DHKEM_X25519_HKDF_SHA256: private_key},
    psk_resolver=resolve_psk,
    # max_sse_event_size=128 * 1024 * 1024,  # Optional: 128MB for large payloads
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()  # Decrypted by middleware

    async def generate():
        yield b"event: progress\ndata: {\"step\": 1}\n\n"
        yield b"event: complete\ndata: {\"result\": \"done\"}\n\n"

    # Just use StreamingResponse - encryption is automatic!
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Client (aiohttp)

```python
from hpke_http.middleware.aiohttp import HPKEClientSession

async with HPKEClientSession(
    base_url="https://api.example.com",
    psk=api_key,        # >= 32 bytes
    psk_id=tenant_id,
) as session:
    resp = await session.post("/chat", json={"prompt": "Hello"})
    async for chunk in session.iter_sse(resp):
        # bytes - matches native aiohttp response.content iteration
        print(chunk)  # b"event: progress\ndata: {...}\n\n"
```

## Documentation

- [RFC 9180 - HPKE](https://datatracker.ietf.org/doc/rfc9180/)
- [RFC 7748 - X25519](https://datatracker.ietf.org/doc/rfc7748/)
- [RFC 5869 - HKDF](https://datatracker.ietf.org/doc/rfc5869/)
- [RFC 8439 - ChaCha20-Poly1305](https://datatracker.ietf.org/doc/rfc8439/)

## Cipher Suite

| Component | Algorithm | ID |
|-----------|-----------|------|
| KEM | DHKEM(X25519, HKDF-SHA256) | 0x0020 |
| KDF | HKDF-SHA256 | 0x0001 |
| AEAD | ChaCha20-Poly1305 | 0x0003 |
| Mode | PSK | 0x01 |

## Wire Format

### Request

```text
┌─────────┬─────────┬─────────┬─────────┬──────┬────────────┐
│ Ver(1B) │ KEM(2B) │ KDF(2B) │AEAD(2B) │Mode  │ Ciphertext │
│  0x01   │ 0x0020  │ 0x0001  │ 0x0003  │(1B)  │  + 16B tag │
└─────────┴─────────┴─────────┴─────────┴──────┴────────────┘
Header: X-HPKE-Enc: <base64url(32B ephemeral key)>
Overhead: 24 bytes (8B header + 16B tag)
```

### SSE Event

```text
event: enc
data: <base64url(counter_be32 || ciphertext || tag)>
Decrypted: raw SSE chunk (e.g., "event: progress\ndata: {...}\n\n")
```

## How SSE Auto-Encryption Works

The middleware automatically encrypts SSE responses when **both** conditions are met:

1. **Request was encrypted** - `SCOPE_HPKE_CONTEXT` exists in scope (from decrypted request)
2. **Response is SSE** - `Content-Type: text/event-stream` header detected

```python
# Middleware detection logic (simplified)
from hpke_http.constants import SCOPE_HPKE_CONTEXT

if scope.get(SCOPE_HPKE_CONTEXT) and b"text/event-stream" in content_type:
    # Auto-encrypt this streaming response
```

This is why `media_type="text/event-stream"` is required - it's the WHATWG-standard MIME type that signals "this is an SSE stream" to both browsers and the middleware.

## Pitfalls

```python
# PSK too short
HPKEClientSession(psk=b"short")                 # ❌ InvalidPSKError
HPKEClientSession(psk=secrets.token_bytes(32))  # ✅ >= 32 bytes

# SSE without proper content-type (won't auto-encrypt)
return StreamingResponse(gen())                                    # ❌ No encryption
return StreamingResponse(gen(), media_type="text/event-stream")    # ✅ Auto-encrypted

# Out-of-order decryption (multi-message context)
recipient.open(aad, ct2)  # ❌ Expects seq=0
recipient.open(aad, ct1)  # ✅ Decrypt in order
```

## Limits

| Resource | Limit |
|----------|-------|
| HPKE messages/context | 2^96-1 |
| SSE events/session | 2^32-1 |
| SSE event buffer | 64MB (configurable) |
| PSK minimum | 32 bytes |
| Overhead | 24 bytes |

> **Note:** SSE is text-only (UTF-8). Binary data must be base64-encoded (+33% overhead).

## Security

Uses OpenSSL constant-time implementations via `cryptography` library.

## Development

```bash
# Install with extras
uv add "hpke-http[fastapi] @ git+https://github.com/duale-ai/hpke-http"  # Server
uv add "hpke-http[aiohttp] @ git+https://github.com/duale-ai/hpke-http"  # Client

# Local development
make install      # Setup venv
make test         # Run tests (1215 tests, 94% coverage)
make test-fuzz    # Property-based fuzz tests
make lint         # Format and lint
```

### Low-Level API

```python
from hpke_http.hpke import seal_psk, open_psk

enc, ct = seal_psk(pk_r, b"info", psk, psk_id, b"aad", b"plaintext")
pt = open_psk(enc, sk_r, b"info", psk, psk_id, b"aad", ct)
```
