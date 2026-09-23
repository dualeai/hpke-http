"""CodSpeed benchmarks of the stable Python API, excluding HTTP and fixture setup."""

from __future__ import annotations

from typing import Any

import pytest

from hpke_http import Client, Method, Request, Response, Server, generate_key_pair

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("size", [0, 1024, 1024 * 1024, 8 * 1024 * 1024], ids=["empty", "1KiB", "1MiB", "8MiB"])
def test_roundtrip(benchmark: Any, size: int) -> None:
    keys = generate_key_pair()
    key_id = b"benchmark-key"
    psk = b"a 32-byte minimum benchmark credential"
    client = Client(keys.public_key, key_id, psk, b"benchmark-tenant")
    server = Server(keys.private_key, key_id)
    request = Request(method=Method.POST, authority="api.example.test", path="/benchmark", body=b"B" * size)
    response = Response(status=200, body=b"C" * size)

    def roundtrip() -> bytes:
        protected = client.protect(request)
        authenticated = server.preparse(protected.envelope).authenticate(psk)
        opened = authenticated.admit(accepted=True)
        return protected.open_response(opened.protect_response(response)).body

    try:
        result = benchmark(roundtrip)
        if len(result) != size:
            raise AssertionError("benchmark roundtrip returned the wrong body length")
    finally:
        client.close()
        server.close()
