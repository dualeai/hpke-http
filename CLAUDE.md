# Welcome to hpke_http

See @README for project overview and @Makefile for available commands for this project.

## Python patterns

See @docs/python-patterns.md for shared Python best practices.

## RFC 9180 Implementation

This library implements RFC 9180 HPKE (Hybrid Public Key Encryption) with PSK mode.

**Cipher Suite:**
- KEM: DHKEM(X25519, HKDF-SHA256) - 0x0020
- KDF: HKDF-SHA256 - 0x0001
- AEAD: ChaCha20-Poly1305 - 0x0003
- Mode: PSK - 0x01

**Key files:**
- `primitives/kdf.py` - LabeledExtract, LabeledExpand (RFC 9180 §4)
- `primitives/kem.py` - X25519 Encap/Decap
- `primitives/aead.py` - ChaCha20-Poly1305 Seal/Open
- `hpke.py` - KeySchedule PSK, seal_psk, open_psk
- `envelope.py` - Wire format encode/decode
- `streaming.py` - SSE session key + counter nonce
- `middleware/fastapi.py` - Server middleware with discovery endpoint
- `middleware/aiohttp.py` - Client session with transparent encryption

## Security Notes

- NEVER truncate cryptographic hashes
- All operations use cryptography library's constant-time implementations
- Test vectors from CFRG must pass before any release
