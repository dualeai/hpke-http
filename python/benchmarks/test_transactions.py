"""CodSpeed benchmarks of the stable Python API, excluding HTTP and fixture setup."""

from __future__ import annotations

from typing import Any

import pytest

from hpke_http import Client, Header, Method, Request, Response, Server, SseSplitter, generate_key_pair

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


def test_sse_joined_split(benchmark: Any) -> None:
    block_count = 8192
    joined = b"data:x\n\n" * block_count

    def split() -> int:
        splitter = SseSplitter(8 * 1024 * 1024)
        offset = 0
        count = 0
        while offset < len(joined):
            used, block = splitter.feed(joined, offset, 64 * 1024)
            offset += used
            count += block is not None
        splitter.finish()
        return count

    if benchmark(split) != block_count:
        raise AssertionError("benchmark splitter lost a block")


def test_sse_100_small_blocks(benchmark: Any) -> None:
    keys = generate_key_pair()
    key_id = b"benchmark-key"
    psk = b"a 32-byte minimum benchmark credential"
    client = Client(keys.public_key, key_id, psk, b"benchmark-tenant")
    server = Server(keys.private_key, key_id)
    request = Request(method=Method.GET, authority="api.example.test", path="/events")
    headers = (Header("content-type", "text/event-stream"),)
    block = b"data:x\n\n"

    def roundtrip() -> int:
        protected = client.protect(request)
        opened = server.preparse(protected.envelope).authenticate(psk).admit(accepted=True)
        writer, start = opened.into_sealer(200, headers)
        reader = protected.into_opener()
        try:
            first = reader.feed(start)[1]
            if first is None or first.kind != "start":
                raise AssertionError("benchmark response START is missing")
            total = 0
            for _ in range(100):
                record = reader.feed(writer.seal_sse_block(block))[1]
                if record is None or record.kind != "data":
                    raise AssertionError("benchmark SSE DATA is missing")
                total += len(record.block)
            end = reader.feed(writer.finish())[1]
            if end is None or end.kind != "end":
                raise AssertionError("benchmark response END is missing")
            reader.finish_eof()
            return total
        finally:
            reader.close()
            writer.close()

    try:
        if benchmark(roundtrip) != len(block) * 100:
            raise AssertionError("benchmark SSE roundtrip lost a block")
    finally:
        client.close()
        server.close()
