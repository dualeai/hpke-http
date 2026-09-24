"""Complete upload checks before ASGI app dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from starlette.datastructures import UploadFile
from starlette.requests import Request as StarletteRequest
from starlette.types import Message, Receive, Scope, Send

from hpke_http import Client, Header, Limits, Method, RequestHead, Response, generate_key_pair
from hpke_http.middleware import PinnedKey
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.middleware.httpx import HPKEAsyncClient
from hpke_http.transport import REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE
from tests.stream_request import open_stream_request
from tests.test_live_sse import live_host

KEY_ID = b"upload-key"
PSK_ID = b"upload-tenant"
PSK = b"a 32-byte minimum upload credential"


def _scope() -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "method": "POST",
            "path": "/protected",
            "raw_path": b"/protected",
            "query_string": b"",
            "scheme": "https",
            "http_version": "1.1",
            "headers": [(b"host", b"api.example.test"), (b"content-type", REQUEST_MEDIA_TYPE.encode())],
        },
    )


async def _resolve(psk_id: bytes, _scope: Scope) -> bytes:
    assert psk_id == PSK_ID
    return PSK


async def _admit(_replay_id: bytes, _deadline: int, _scope: Scope) -> bool:
    return True


async def _receive_from(queue: asyncio.Queue[Message]) -> Message:
    return await queue.get()


def _push_all(writer: Any, clear: bytes) -> bytes:
    frames: list[bytes] = []
    offset = 0
    while offset < len(clear):
        used, frame = writer.push(clear[offset : offset + 64 * 1024])
        assert used > 0 or frame is not None
        offset += used
        if frame is not None:
            frames.append(frame)
    return b"".join(frames)


@pytest.mark.asyncio
async def test_app_starts_only_after_end_and_outer_eof() -> None:
    keys = generate_key_pair()
    app_started = asyncio.Event()
    seen = asyncio.Event()
    waiting_after_data = asyncio.Event()
    waiting_after_end = asyncio.Event()
    events: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        app_started.set()
        assert scope["path"] == "/upload"
        first = await receive()
        assert first == {"type": "http.request", "body": b"onetwo", "more_body": True}
        assert await receive() == {"type": "http.request", "body": b"", "more_body": False}
        seen.set()
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"saved", "more_body": False})

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
    )
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    writer, first = client.begin_stream(RequestHead(Method.POST, "api.example.test", "/upload"))
    part_one = _push_all(writer, b"one")
    part_two = _push_all(writer, b"two")
    end, right = writer.finish()
    queue: asyncio.Queue[Message] = asyncio.Queue()
    reads = 0

    async def receive() -> Message:
        nonlocal reads
        reads += 1
        if reads == 3:
            waiting_after_data.set()
        elif reads == 4:
            waiting_after_end.set()
        return await _receive_from(queue)

    async def send(message: Message) -> None:
        events.append(message)

    task = asyncio.create_task(middleware(_scope(), receive, send))
    await queue.put({"type": "http.request", "body": first[:7], "more_body": True})
    await queue.put({"type": "http.request", "body": first[7:] + part_one, "more_body": True})
    await asyncio.wait_for(waiting_after_data.wait(), 2)
    assert not app_started.is_set()
    assert not seen.is_set()
    assert events == []
    await queue.put({"type": "http.request", "body": part_two + end, "more_body": True})
    await asyncio.wait_for(waiting_after_end.wait(), 2)
    assert not app_started.is_set()
    assert events == []
    await queue.put({"type": "http.request", "body": b"", "more_body": False})
    await asyncio.wait_for(task, 2)
    assert app_started.is_set()
    assert seen.is_set()
    assert events[0]["status"] == 200
    assert right.open_response(cast(bytes, events[1]["body"])) == Response(201, (), b"saved")
    middleware.close()
    client.close()


@pytest.mark.asyncio
async def test_one_asgi_event_carries_many_upload_records() -> None:
    keys = generate_key_pair()
    clear = os.urandom(3 * 64 * 1024 + 17)
    events: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert await StarletteRequest(scope, receive).body() == clear
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"saved", "more_body": False})

    middleware = HPKEMiddleware(app, keys.private_key, KEY_ID, _resolve, _admit, transport_path="/protected")
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    writer, first = client.begin_stream(RequestHead(Method.POST, "api.example.test", "/upload"))
    data = _push_all(writer, clear)
    end, right = writer.finish()
    queue: asyncio.Queue[Message] = asyncio.Queue()
    await queue.put({"type": "http.request", "body": first + data + end, "more_body": False})

    async def send(message: Message) -> None:
        events.append(message)

    try:
        await asyncio.wait_for(middleware(_scope(), queue.get, send), 5)
        assert events[0]["status"] == 200
        assert right.open_response(cast(bytes, events[1]["body"])) == Response(201, (), b"saved")
    finally:
        right.close()
        client.close()
        middleware.close()


@pytest.mark.asyncio
async def test_late_bad_record_never_calls_app() -> None:
    keys = generate_key_pair()
    app_calls = 0
    events: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal app_calls
        app_calls += 1

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
    )
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    writer, first = client.begin_stream(RequestHead(Method.POST, "api.example.test", "/upload"))
    data = _push_all(writer, b"one")
    end, right = writer.finish()
    bad_end = bytearray(end)
    bad_end[-1] ^= 1
    queue: asyncio.Queue[Message] = asyncio.Queue()

    async def receive() -> Message:
        return await queue.get()

    async def send(message: Message) -> None:
        events.append(message)

    task = asyncio.create_task(middleware(_scope(), receive, send))
    await queue.put({"type": "http.request", "body": first + data, "more_body": True})
    await queue.put({"type": "http.request", "body": bytes(bad_end), "more_body": False})
    await asyncio.wait_for(task, 2)
    assert app_calls == 0
    assert events[0]["status"] == 400
    assert all(message.get("headers") != [(b"content-type", b"message/hpke-http-response")] for message in events)
    right.close()
    middleware.close()
    client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_end", "extra_bytes", "disconnect"])
async def test_incomplete_upload_never_calls_app(failure: str) -> None:
    keys = generate_key_pair()
    events: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("app must not run before the full upload passes")

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
    )
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    writer, first = client.begin_stream(RequestHead(Method.POST, "api.example.test", "/upload"))
    data = _push_all(writer, b"x" * (64 * 1024))
    end, right = writer.finish()
    assert data and end
    queue: asyncio.Queue[Message] = asyncio.Queue()
    if failure == "missing_end":
        await queue.put({"type": "http.request", "body": first + data, "more_body": False})
    elif failure == "extra_bytes":
        await queue.put({"type": "http.request", "body": first + data + end + b"x", "more_body": False})
    else:
        await queue.put({"type": "http.request", "body": first + data, "more_body": True})
        await queue.put({"type": "http.disconnect"})

    async def receive() -> Message:
        return await queue.get()

    async def send(message: Message) -> None:
        events.append(message)

    await asyncio.wait_for(middleware(_scope(), receive, send), 2)
    assert events[0]["status"] == 400
    right.close()
    middleware.close()
    client.close()


@pytest.mark.asyncio
async def test_spool_write_error_never_calls_app(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = generate_key_pair()
    opened_files: list[Any] = []
    original = tempfile.SpooledTemporaryFile

    def failing_file(*args: Any, **kwargs: Any) -> Any:
        file = original(*args, **kwargs)
        opened_files.append(file)

        def fail_write(_part: bytes) -> int:
            raise OSError("storage failed")

        monkeypatch.setattr(file, "write", fail_write)
        return file

    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", failing_file)

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("app must not run when upload storage fails")

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
    )
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    writer, first = client.begin_stream(RequestHead(Method.POST, "api.example.test", "/upload"))
    data = _push_all(writer, b"x" * (64 * 1024))
    end, right = writer.finish()
    assert data
    queue: asyncio.Queue[Message] = asyncio.Queue()
    await queue.put({"type": "http.request", "body": first + data + end, "more_body": False})
    events: list[Message] = []

    async def receive() -> Message:
        return await queue.get()

    async def send(message: Message) -> None:
        events.append(message)

    await asyncio.wait_for(middleware(_scope(), receive, send), 2)
    assert events[0]["status"] == 503
    assert opened_files and all(file.closed for file in opened_files)
    right.close()
    middleware.close()
    client.close()


@pytest.mark.asyncio
async def test_bad_part_closes_middleware_spool(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = generate_key_pair()
    opened_files: list[Any] = []
    original = tempfile.SpooledTemporaryFile

    def tracked_file(*args: Any, **kwargs: Any) -> Any:
        file = original(*args, **kwargs)
        opened_files.append(file)
        return file

    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", tracked_file)

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("app must not run for a bad END")

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
    )
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    boundary = "test-boundary"
    writer, first = client.begin_stream(
        RequestHead(
            Method.POST,
            "api.example.test",
            "/upload",
            (Header("content-type", f"multipart/form-data; boundary={boundary}"),),
        )
    )
    clear = (
        b"--test-boundary\r\n"
        b'Content-Disposition: form-data; name="upload"; filename="one.bin"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + b"x" * (128 * 1024)
    )
    frames = _push_all(writer, clear)
    end, right = writer.finish()
    damaged = bytearray(end)
    damaged[-1] ^= 1
    messages = iter([{"type": "http.request", "body": first + frames + damaged, "more_body": False}])
    sent: list[Message] = []

    async def receive() -> Message:
        return cast(Message, next(messages))

    async def send(message: Message) -> None:
        sent.append(message)

    await middleware(_scope(), receive, send)
    assert sent[0]["status"] == 400
    assert opened_files and all(file.closed for file in opened_files)
    right.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_key_get_returns_one_key_record() -> None:
    keys = generate_key_pair()

    async def unexpected_app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("app must not run")

    middleware = HPKEMiddleware(
        unexpected_app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
    )
    sent: list[Message] = []

    async def receive() -> Message:
        pytest.fail("key GET must not read the body")

    async def send(message: Message) -> None:
        sent.append(message)

    scope = _scope()
    scope["method"] = "GET"
    scope["headers"] = [(b"host", b"api.example.test")]
    await middleware(scope, receive, send)
    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b"HHKD\x01" + bytes((len(KEY_ID),)) + KEY_ID + keys.public_key
    middleware.close()


@pytest.mark.asyncio
async def test_stream_host_rejects_upload_limit_before_app() -> None:
    keys = generate_key_pair()

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("app must not run for an over-limit upload")

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
        limits=Limits(max_request_bytes=3),
    )
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    writer, first = client.begin_stream(RequestHead(Method.POST, "api.example.test", "/upload"))
    data = _push_all(writer, b"four")
    end, right = writer.finish()
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": first + data + end, "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await middleware(_scope(), receive, send)
    assert sent[0]["status"] == 400
    right.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_cancelled_httpx_async_source_closes_without_end() -> None:
    keys = generate_key_pair()
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def source() -> AsyncIterator[bytes]:
        try:
            yield b"one"
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def transport(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        await request.aread()
        pytest.fail("source must be cancelled before END")

    async with HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(keys.public_key, KEY_ID),
        PSK,
        PSK_ID,
        transport=httpx.MockTransport(transport),
    ) as client:
        task = asyncio.create_task(client.post("https://api.example.test/upload", content=source()))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closed.wait(), 2)


@pytest.mark.asyncio
async def test_httpx_files_and_async_content_keep_their_normal_body_shape() -> None:
    keys = generate_key_pair()
    from hpke_http import Server

    server = Server(keys.private_key, KEY_ID)
    seen: list[tuple[str, bytes]] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        raw = await request.aread()
        media = request.headers["content-type"]
        assert media == REQUEST_MEDIA_TYPE
        opened = open_stream_request(server, raw, PSK)
        seen.append((media, opened.request.body))
        reply = opened.protect_response(Response(200, (), b"ok"))
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, content=reply)

    adapter = HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(keys.public_key, KEY_ID),
        PSK,
        PSK_ID,
        transport=httpx.MockTransport(transport),
    )
    caller_file = io.BytesIO(b"file-content")

    async def content() -> AsyncIterator[bytes]:
        yield b"raw-"
        yield b"content"

    async with adapter:
        first = await adapter.post("https://api.example.test/upload", files={"file": ("one.bin", caller_file)})
        second = await adapter.post("https://api.example.test/upload", content=content())
        third = await adapter.post("https://api.example.test/upload", content=b"bytes")
    assert first.content == second.content == third.content == b"ok"
    assert caller_file.closed is False
    assert seen[0][0] == REQUEST_MEDIA_TYPE
    assert b'name="file"' in seen[0][1]
    assert b"file-content" in seen[0][1]
    assert seen[1] == (REQUEST_MEDIA_TYPE, b"raw-content")
    assert seen[2] == (REQUEST_MEDIA_TYPE, b"bytes")
    server.close()


@pytest.mark.asyncio
async def test_file_backed_httpx_upload_over_64_mib_over_tls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_size = 64 * 1024 * 1024 + 1
    path = tmp_path / "large-upload.bin"
    expected_hash = hashlib.sha256()
    with path.open("wb") as output:
        for _ in range(64):
            chunk = os.urandom(1024 * 1024)
            output.write(chunk)
            expected_hash.update(chunk)
        last = os.urandom(1)
        output.write(last)
        expected_hash.update(last)
    keys = generate_key_pair()
    seen: list[tuple[int, bytes]] = []
    checked_bytes = 0
    largest_write = 0
    rolled_to_disk = False
    original_spooled_file = tempfile.SpooledTemporaryFile

    def tracked_file(*args: Any, **kwargs: Any) -> Any:
        spool: Any = original_spooled_file(*args, **kwargs)
        if kwargs.get("max_size") != 256 * 1024:
            return spool
        original_write = spool.write

        def checked_write(part: Any) -> int:
            nonlocal checked_bytes, largest_write, rolled_to_disk
            largest_write = max(largest_write, len(part))
            written = original_write(part)
            checked_bytes += written
            rolled_to_disk |= bool(spool._rolled)  # noqa: SLF001
            return written

        spool.write = checked_write
        return spool

    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", tracked_file)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        request = StarletteRequest(scope, receive)
        async with request.form(max_files=1, max_fields=0) as form:
            upload = form["upload"]
            assert isinstance(upload, UploadFile)
            assert upload.filename == path.name
            total = 0
            digest = hashlib.sha256()
            while part := await upload.read(64 * 1024):
                total += len(part)
                digest.update(part)
            seen.append((total, digest.digest()))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, tls):
        async with HPKEAsyncClient(
            endpoint,
            PinnedKey(keys.public_key, KEY_ID),
            PSK,
            PSK_ID,
            target_origin="https://api.example.test",
            verify=tls,
        ) as client:
            with path.open("rb") as upload:
                response = await asyncio.wait_for(
                    client.post("https://api.example.test/upload", files={"upload": upload}), 90
                )
            assert response.content == b"ok"
            assert seen == [(file_size, expected_hash.digest())]
            assert checked_bytes > file_size
            assert 0 < largest_write <= 64 * 1024
            assert rolled_to_disk


@pytest.mark.asyncio
async def test_live_form_parser_gets_fields_and_file_over_tls(tmp_path: Path) -> None:
    keys = generate_key_pair()
    seen: list[tuple[str, str, bytes]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        request = StarletteRequest(scope, receive)
        async with request.form(max_files=1, max_fields=1) as form:
            upload = form["upload"]
            assert isinstance(upload, UploadFile)
            assert upload.filename is not None
            content = await upload.read()
            seen.append((cast(str, form["note"]), upload.filename, content))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        _resolve,
        _admit,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, tls):
        async with HPKEAsyncClient(
            endpoint,
            PinnedKey(keys.public_key, KEY_ID),
            PSK,
            PSK_ID,
            target_origin="https://api.example.test",
            verify=tls,
        ) as client:
            response = await client.post(
                "https://api.example.test/upload",
                data={"note": "one"},
                files={"upload": ("one.bin", b"file-content")},
            )
            assert response.content == b"ok"
    assert seen == [("one", "one.bin", b"file-content")]
