# hpke-http

End-to-end encryption for HTTP APIs using RFC 9180 HPKE.

[![CI](https://github.com/dualeai/hpke-http/actions/workflows/test.yml/badge.svg)](https://github.com/dualeai/hpke-http/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/hpke-http)](https://pypi.org/project/hpke-http/)
[![Downloads](https://img.shields.io/pypi/dm/hpke-http)](https://pypi.org/project/hpke-http/)
[![Python](https://img.shields.io/pypi/pyversions/hpke-http)](https://pypi.org/project/hpke-http/)
[![License](https://img.shields.io/pypi/l/hpke-http)](https://opensource.org/licenses/Apache-2.0)

## Highlights

- **Transparent** - Drop-in middleware, no application code changes
- **E2E encryption** - Protects data even with TLS termination at CDN/LB
- **PSK binding** - Each request cryptographically bound to API key
- **Replay protection** - Counter-based nonces prevent replay attacks
- **RFC 9180 compliant** - Auditable, interoperable standard

## Installation

```bash
uv add "hpke-http[fastapi]"       # Server
uv add "hpke-http[aiohttp]"       # Client
uv add "hpke-http[fastapi,zstd]"  # + compression
```

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
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()  # Decrypted automatically

    async def generate():
        yield b"event: progress\ndata: {\"step\": 1}\n\n"
        yield b"event: complete\ndata: {\"result\": \"done\"}\n\n"

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
        print(chunk)  # b"event: progress\ndata: {...}\n\n"
```

## Documentation

- [RFC 9180 - HPKE](https://datatracker.ietf.org/doc/rfc9180/)
- [RFC 7748 - X25519](https://datatracker.ietf.org/doc/rfc7748/)
- [RFC 5869 - HKDF](https://datatracker.ietf.org/doc/rfc5869/)
- [RFC 8439 - ChaCha20-Poly1305](https://datatracker.ietf.org/doc/rfc8439/)
- [RFC 8878 - Zstandard](https://datatracker.ietf.org/doc/rfc8878/) (optional compression)

## Security

Uses OpenSSL constant-time implementations via `cryptography` library.

- [Security Policy](./SECURITY.md) - Vulnerability reporting
- [SBOM](https://github.com/dualeai/hpke-http/releases) - CycloneDX attached to releases

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

```bash
make install      # Setup venv
make test         # Run tests
make lint         # Format and lint
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

---

<details>
<summary>Technical Details</summary>

## Cipher Suite

| Component | Algorithm | ID |
| --------- | --------- | ------ |
| KEM | DHKEM(X25519, HKDF-SHA256) | 0x0020 |
| KDF | HKDF-SHA256 | 0x0001 |
| AEAD | ChaCha20-Poly1305 | 0x0003 |
| Mode | PSK | 0x01 |

## Wire Format

### Request/Response (Chunked Binary)

```text
Headers:
  X-HPKE-Enc: <base64url(32B ephemeral key)>
  X-HPKE-Stream: <base64url(4B session salt)>

Body (repeating chunks):
┌───────────┬────────────┬─────────────────────────────────┐
│ Length(4B)│ Counter(4B)│ Ciphertext (N + 16B tag)        │
│ big-end   │ big-end    │ encrypted: encoding_id || data  │
└───────────┴────────────┴─────────────────────────────────┘
Overhead: 24B/chunk (4B length + 4B counter + 16B tag)
```

### SSE Event

```text
event: enc
data: <base64(counter_be32 || ciphertext)>
Decrypted: raw SSE chunk (e.g., "event: progress\ndata: {...}\n\n")
```

Uses standard base64 (not base64url) - SSE data fields allow +/= characters.

## SSE Auto-Encryption

The middleware automatically encrypts SSE responses when **both** conditions are met:

1. **Request was encrypted** - `SCOPE_HPKE_CONTEXT` exists in scope
2. **Response is SSE** - `Content-Type: text/event-stream` detected

This is why `media_type="text/event-stream"` is required.

## Compression (Optional)

Zstd compression reduces bandwidth by **40-95%** for JSON/text.

```python
HPKEMiddleware(..., compress=True)      # Server
HPKEClientSession(..., compress=True)   # Client
```

| Choice | Rationale |
| ------ | --------- |
| Compress-then-encrypt | Encrypted data is incompressible |
| Zstd (RFC 8878) | Best ratio/speed, Python 3.14 native |
| 64B threshold | Smaller payloads skip compression |

## Pitfalls

```python
# PSK too short
HPKEClientSession(psk=b"short")                 # InvalidPSKError
HPKEClientSession(psk=secrets.token_bytes(32))  # >= 32 bytes

# Missing content-type (won't auto-encrypt)
return StreamingResponse(gen())                                  # No encryption
return StreamingResponse(gen(), media_type="text/event-stream")  # Auto-encrypted
```

## Limits

| Resource | Limit |
| -------- | ----- |
| HPKE messages/context | 2^96-1 |
| SSE events/session | 2^32-1 |
| SSE event buffer | 64MB (configurable) |
| PSK minimum | 32 bytes |
| Chunk overhead | 24B (length + counter + tag) |
| Chunk size | 64KB |

> **Note:** SSE is text-only (UTF-8). Binary data must be base64-encoded (+33% overhead).

## Low-Level API

```python
from hpke_http.hpke import seal_psk, open_psk

enc, ct = seal_psk(pk_r, b"info", psk, psk_id, b"aad", b"plaintext")
pt = open_psk(enc, sk_r, b"info", psk, psk_id, b"aad", ct)
```

</details>
