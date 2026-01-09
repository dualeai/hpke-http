# hpke-http

End-to-end encryption for HTTP APIs using RFC 9180 HPKE (Hybrid Public Key Encryption). Drop-in middleware for FastAPI, aiohttp, and httpx.

[![CI](https://github.com/dualeai/hpke-http/actions/workflows/test.yml/badge.svg)](https://github.com/dualeai/hpke-http/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/hpke-http)](https://pypi.org/project/hpke-http/)
[![Downloads](https://img.shields.io/pypi/dm/hpke-http)](https://pypi.org/project/hpke-http/)
[![Python](https://img.shields.io/pypi/pyversions/hpke-http)](https://pypi.org/project/hpke-http/)
[![License](https://img.shields.io/pypi/l/hpke-http)](https://opensource.org/licenses/Apache-2.0)

## Highlights

- **Transparent** - Drop-in middleware, no application code changes
- **End-to-end encryption** - Protects data even when TLS terminates at CDN or load balancer
- **PSK binding** - Each request cryptographically bound to pre-shared key (API key)
- **Replay protection** - Counter-based nonces (numbers used once) prevent replay attacks
- **RFC 9180 compliant** - Auditable, interoperable standard
- **Memory-efficient** - Streams large file uploads with O(chunk_size) memory

## Installation

```bash
uv add "hpke-http[fastapi]"       # Server
uv add "hpke-http[aiohttp]"       # Client (aiohttp)
uv add "hpke-http[httpx]"         # Client (httpx)
uv add "hpke-http[fastapi,zstd]"  # + zstd compression (gzip fallback included)
```

## Quick Start

Standard JSON requests, SSE (Server-Sent Events) streaming, and file uploads are transparently encrypted.

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

# Standard JSON endpoint - encryption is automatic
@app.post("/users")
async def create_user(request: Request):
    data = await request.json()  # Decrypted automatically
    return {"id": 123, "name": data["name"]}  # Encrypted automatically

# SSE streaming endpoint - encryption is automatic
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
import aiohttp
from hpke_http.middleware.aiohttp import HPKEClientSession

async with HPKEClientSession(
    base_url="https://api.example.com",
    psk=api_key,        # >= 32 bytes
    psk_id=tenant_id,
    # compress=True,           # Compression (zstd preferred, gzip fallback)
    # require_encryption=True, # Raise if server responds unencrypted
    # release_encrypted=True,  # Free encrypted bytes after decryption (saves memory)
) as session:
    # Standard JSON request - encryption is automatic
    async with session.post("/users", json={"name": "Alice"}) as resp:
        user = await resp.json()  # Decrypted automatically
        print(user)  # {"id": 123, "name": "Alice"}

    # SSE streaming request - encryption is automatic
    async with session.post("/chat", json={"prompt": "Hello"}) as resp:
        async for chunk in session.iter_sse(resp):
            print(chunk)  # b"event: progress\ndata: {...}\n\n"

    # File upload - encryption is automatic, streams with O(chunk_size) memory
    form = aiohttp.FormData()
    form.add_field("file", open("large.pdf", "rb"), filename="large.pdf")
    async with session.post("/upload", data=form) as resp:
        result = await resp.json()
```

### Client (httpx)

```python
from hpke_http.middleware.httpx import HPKEAsyncClient

async with HPKEAsyncClient(
    base_url="https://api.example.com",
    psk=api_key,        # >= 32 bytes
    psk_id=tenant_id,
    # compress=True,           # Compression (zstd preferred, gzip fallback)
    # require_encryption=True, # Raise if server responds unencrypted
    # release_encrypted=True,  # Free encrypted bytes after decryption (saves memory)
) as client:
    # Standard JSON request - encryption is automatic
    resp = await client.post("/users", json={"name": "Alice"})
    user = resp.json()  # Decrypted automatically
    print(user)  # {"id": 123, "name": "Alice"}

    # SSE streaming request - encryption is automatic
    resp = await client.post("/chat", json={"prompt": "Hello"})
    async for chunk in client.iter_sse(resp):
        print(chunk)  # b"event: progress\ndata: {...}\n\n"

    # File upload - encryption is automatic, streams with O(chunk_size) memory
    resp = await client.post("/upload", files={"file": open("large.pdf", "rb")})
    result = resp.json()
```

## Documentation

- [RFC 9180 - HPKE](https://datatracker.ietf.org/doc/rfc9180/)
- [RFC 7748 - X25519](https://datatracker.ietf.org/doc/rfc7748/)
- [RFC 5869 - HKDF](https://datatracker.ietf.org/doc/rfc5869/)
- [RFC 8439 - ChaCha20-Poly1305](https://datatracker.ietf.org/doc/rfc8439/)
- [RFC 8878 - Zstandard](https://datatracker.ietf.org/doc/rfc8878/) (preferred compression)
- [RFC 1952 - Gzip](https://datatracker.ietf.org/doc/rfc1952/) (fallback compression, always available)
- [RFC 9110 - HTTP Semantics](https://datatracker.ietf.org/doc/rfc9110/) (Accept-Encoding negotiation)

## Cipher Suite

| Component | Algorithm | ID |
| --------- | --------- | ------ |
| KEM (Key Encapsulation) | DHKEM(X25519, HKDF-SHA256) | 0x0020 |
| KDF (Key Derivation) | HKDF-SHA256 | 0x0001 |
| AEAD (Authenticated Encryption) | ChaCha20-Poly1305 | 0x0003 |
| Mode | PSK (Pre-Shared Key) | 0x01 |

## Wire Format

### Request/Response (Chunked Binary)

See [Header Modifications](#header-modifications) for when headers are added.

```text
Headers:
  X-HPKE-Enc: <base64url(32B ephemeral key)>
  X-HPKE-Stream: <base64url(4B session salt)>

Body (repeating chunks):
┌───────────┬────────────┬─────────────────────────────────┐
│ Length(4B)│ Counter(4B)│ Ciphertext (N + 16B tag)        │
│ big-endian│ big-endian │ encrypted: encoding_id || data  │
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

## Auto-Encryption

The middleware automatically encrypts **all responses** when the request was encrypted. See [Response Types](#response-types) for format selection and [HTTP Methods](#http-methods) for when encryption activates.

## Compression (Optional)

Zstd reduces bandwidth by **40-95%** for JSON/text. Enable with `compress=True` on both client and server. Payloads < 64 bytes skip compression. See [Compression table](#compression) for algorithm priority.

## Pitfalls

```python
# PSK too short (applies to both aiohttp and httpx clients)
HPKEClientSession(psk=b"short")                 # InvalidPSKError
HPKEClientSession(psk=secrets.token_bytes(32))  # >= 32 bytes

# SSE missing content-type (won't use SSE format)
return StreamingResponse(gen())                                  # Binary format (wrong for SSE)
return StreamingResponse(gen(), media_type="text/event-stream")  # SSE format (correct)

# Standard responses work automatically - no special handling needed
return {"data": "value"}  # Auto-encrypted as binary chunks
```

## Limits

| Resource | Limit | Applies to |
| -------- | ----- | ---------- |
| HPKE messages/context | 2^96-1 | All |
| Chunks/session | 2^32-1 | All |
| PSK minimum | 32 bytes | All |
| Chunk size | 64KB | All |
| Binary chunk overhead | 24B (length + counter + tag) | Requests & standard responses |
| SSE event buffer | 64MB (configurable) | SSE only |

> **Note:** SSE is text-only (UTF-8). Binary data must be base64-encoded (+33% overhead).

## HTTP Compatibility

### Protocol Support

| Feature | Supported | Notes |
| ------- | --------- | ----- |
| HTTP/1.1 | Yes | Chunked transfer encoding for streaming |
| HTTP/2 | Yes | Native framing (chunked encoding forbidden by spec) |
| HTTP/3 | Yes | QUIC streams, same semantics as HTTP/2 |
| WebSockets | No | Different protocol, not applicable |

### HTTP Methods

All methods supported. **Encryption requires a request body** to establish the HPKE context.

| Method | Typical Use | With Body | Without Body |
| ------ | ----------- | --------- | ------------ |
| POST | Create | Encrypted (both directions) | Plaintext* |
| PUT | Replace | Encrypted (both directions) | Plaintext* |
| PATCH | Update | Encrypted (both directions) | Plaintext* |
| DELETE | Remove | Encrypted (both directions) | Plaintext |
| GET | Read | Encrypted (both directions) | Plaintext |
| HEAD | Metadata | N/A | Plaintext (no response body) |
| OPTIONS | Preflight | Encrypted (both directions) | Plaintext |

*Unusual - these methods typically have a body.

> **Design note:** The HPKE encryption context is established during request encryption (key exchange + PSK). Without a request body, no encryption context exists for the response.
>
> For read-only endpoints requiring E2E encryption, use POST with a body (common pattern for sensitive queries).

### Response Encryption (Server)

| Content-Type | Wire Format | Memory |
| ------------ | ----------- | ------ |
| Any non-SSE | Length-prefixed 64KB chunks | O(64KB) buffer |
| `text/event-stream` | Base64 SSE events | O(event size) |

### Response Decryption (Client)

| Content-Type | API | Memory | Delivery |
| ------------ | --- | ------ | -------- |
| Any non-SSE | `resp.json()`, `resp.content` | O(response size) | After full download |
| `text/event-stream` | `async for chunk in iter_sse(resp)` | O(event size) | As events arrive |

> **Tip:** For large non-SSE responses, use `release_encrypted=True` to free the encrypted buffer after decryption, reducing peak memory from 2× to 1× response size.

### Compression

| Algorithm | Request | Response | Priority |
| --------- | ------- | -------- | -------- |
| Zstd (RFC 8878) | Yes | Yes | 1 (preferred) |
| Gzip (RFC 1952) | Yes | Yes | 2 (fallback) |
| Identity | Yes | Yes | 3 (no compression) |

Auto-negotiated via `Accept-Encoding` header on discovery endpoint (`/.well-known/hpke-keys`).

#### Why HTTP-Level Compression Doesn't Help

**Disable gzip/brotli on your load balancer or CDN for HPKE endpoints.** The library compresses plaintext *before* encryption (the only effective order). Ciphertext has ~8 bits/byte entropy and is mathematically incompressible—HTTP compression only adds framing overhead and wastes CPU. Use `compress=True` on the client instead.

> **Security note:** Unlike TLS compression (disabled since CRIME/BREACH), hpke-http's compression is safe because it uses per-request ephemeral keys and doesn't mix attacker-controlled input with secrets in the same compressed payload.

## Encryption Scope

> Applies when request has a body (see [HTTP Methods](#http-methods) above).

### What IS Encrypted

| Component | Encrypted | Format |
| --------- | --------- | ------ |
| Request body | Yes | Binary chunks |
| Response body | Yes | Binary chunks or SSE events |
| JSON payloads | Yes | Inside encrypted body |
| Binary data | Yes | Inside encrypted body |
| SSE event content | Yes | Base64 in `data:` field |

### What is NOT Encrypted

| Component | Visible to | Reason |
| --------- | ---------- | ------ |
| URL path | Network | Routing requires plaintext |
| Query parameters | Network | Part of URL |
| HTTP method | Network | Protocol requirement |
| HTTP headers | Network | Routing, caching, auth |
| Status code | Network | Protocol requirement |
| TLS metadata | Network | Transport layer |

### Header Modifications

| Header | Request | Response | Reason |
| ------ | ------- | -------- | ------ |
| `Content-Type` | Set to `application/octet-stream` | Preserved | Encrypted body is binary |
| `Content-Length` | Auto (chunked) | Removed | Size changes after encryption |
| `X-HPKE-Enc` | Added | - | Ephemeral public key |
| `X-HPKE-Stream` | Added | Added | Session salt for nonces |
| `X-HPKE-Encoding` | Added (if compressed) | - | Compression algorithm |
| `X-HPKE-Content-Type` | Added (if body) | - | Original Content-Type for server parsing |

### Security Boundary

```
┌─────────────────────────────────────────────────────────────┐
│ TLS Encrypted (transport)                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ HTTP Layer (visible to CDN/LB/proxies)                │  │
│  │  • Method: POST                                       │  │
│  │  • URL: /api/chat                                     │  │
│  │  • Headers: Authorization, X-HPKE-*, Content-Type     │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │ HPKE Encrypted (end-to-end)                     │  │  │
│  │  │  • Request body: {"prompt": "Hello"}            │  │  │
│  │  │  • Response body: {"response": "Hi!"}           │  │  │
│  │  │  • SSE events: event: done\ndata: {...}\n\n     │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Low-Level API

Direct access to HPKE seal/open operations:

```python
from hpke_http.hpke import seal_psk, open_psk

# pk_r: recipient public key, sk_r: recipient secret key
# psk/psk_id: pre-shared key and identifier, aad: additional authenticated data
enc, ct = seal_psk(pk_r, b"info", psk, psk_id, b"aad", b"plaintext")
pt = open_psk(enc, sk_r, b"info", psk, psk_id, b"aad", ct)
```

## Security

Uses OpenSSL constant-time implementations via `cryptography` library.

- [Security Policy](./SECURITY.md) - Vulnerability reporting
- [SBOM](https://github.com/dualeai/hpke-http/releases) - Software Bill of Materials (CycloneDX format) attached to releases

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

```bash
make install      # Setup venv
make test         # Run tests
make lint         # Format and lint
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)
