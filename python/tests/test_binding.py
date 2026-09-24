"""The public Python API must traverse the private PyO3 extension."""

from __future__ import annotations

import importlib
import importlib.machinery
import time
from typing import cast

import pytest

from hpke_http import (
    BINDING_ABI_VERSION,
    PACKAGE_VERSION,
    PROTOCOL_ID,
    Client,
    Header,
    Limits,
    Method,
    ProtocolError,
    Request,
    RequestHead,
    Response,
    Server,
    StateError,
    StreamResponseRight,
    _native,
    generate_key_pair,
)
from hpke_http.protocol import validate_build_info

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"


def _engines() -> tuple[Client, Server]:
    key_pair = generate_key_pair()
    return (
        Client(key_pair.public_key, KEY_ID, PSK, PSK_ID),
        Server(key_pair.private_key, KEY_ID),
    )


def _request(body: bytes = b'{"name":"Ada"}') -> Request:
    return Request(
        method=Method.POST,
        authority="api.example.test",
        path="/items?limit=2",
        headers=(Header("content-type", "application/json"),),
        body=body,
    )


def test_native_bootstrap_identity_and_extension_origin() -> None:
    version, protocol, abi = _native.native_build_info()
    assert version == PACKAGE_VERSION
    assert protocol == PROTOCOL_ID
    assert abi == BINDING_ABI_VERSION
    assert _native.__file__ is not None
    assert any(_native.__file__.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def test_server_public_key_matches_current_private_key_and_close_blocks_access() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    assert server.public_key == keys.public_key
    server.close()
    with pytest.raises(StateError):
        _ = server.public_key


def test_removed_python_protocol_modules_are_not_importable() -> None:
    for module in (
        "hpke_http.constants",
        "hpke_http.core",
        "hpke_http.exceptions",
        "hpke_http.headers",
        "hpke_http.hpke",
        "hpke_http.primitives.aead",
        "hpke_http.primitives.kem",
        "hpke_http.streaming",
    ):
        with pytest.raises(ModuleNotFoundError) as error:
            importlib.import_module(module)
        missing_module = error.value.name
        assert missing_module is not None
        assert module == missing_module or module.startswith(f"{missing_module}.")


@pytest.mark.parametrize(
    ("version", "protocol", "abi"),
    [
        ("different", PROTOCOL_ID, BINDING_ABI_VERSION),
        (PACKAGE_VERSION, "different", BINDING_ABI_VERSION),
        (PACKAGE_VERSION, PROTOCOL_ID, BINDING_ABI_VERSION + 1),
    ],
)
def test_build_info_skew_is_rejected(version: str, protocol: str, abi: int) -> None:
    with pytest.raises(ImportError, match="native module mismatch"):
        validate_build_info(version, protocol, abi)


def test_complete_python_transaction_and_one_shot_lifecycle() -> None:
    client, server = _engines()
    protected = client.protect(_request())
    preparsed = server.preparse(protected.envelope)
    assert preparsed.psk_id == PSK_ID

    authenticated = preparsed.authenticate(PSK)
    assert len(authenticated.replay_id) == 32
    assert isinstance(authenticated.retain_until_exclusive, int)
    assert authenticated.retain_until_exclusive > int(time.time())
    opened = authenticated.admit(accepted=True)
    assert opened.request == _request()

    expected = Response(
        status=201,
        headers=(Header("content-type", "application/json"),),
        body=b'{"id":"item-1"}',
    )
    encrypted_response = opened.protect_response(expected)
    assert protected.open_response(encrypted_response) == expected

    assert protected.consumed
    assert preparsed.consumed
    assert authenticated.consumed
    assert opened.response_consumed
    with pytest.raises(StateError, match="continuation already consumed"):
        protected.open_response(encrypted_response)


def test_stream_completion_returns_response_right_without_request_body() -> None:
    client, server = _engines()
    writer, first = client.begin_stream(RequestHead(Method.POST, "api.example.test", "/items"))
    end, opener = writer.finish()
    opened = server.preparse_stream(first).authenticate(PSK).admit(accepted=True)
    used, record = opened.feed(end)
    assert used == len(end)
    assert record == ("end", b"")
    response_right = opened.finish_eof()
    assert isinstance(response_right, StreamResponseRight)
    assert not hasattr(response_right, "request")
    expected = Response(status=200, body=b"ok")
    assert opener.open_response(response_right.protect_response(expected)) == expected
    client.close()
    server.close()


@pytest.mark.parametrize("body", [b"", b"finite body"])
def test_low_level_finite_response_sealer(body: bytes) -> None:
    client, server = _engines()
    try:
        protected = client.protect(_request())
        opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
        headers = (Header("content-type", "text/plain"),)
        writer, start = opened.into_sealer(200, headers)
        data = writer.seal_finite_body(body)
        assert (data is None) == (body == b"")
        end = writer.finish()
        assert protected.open_response(start + (data or b"") + end) == Response(200, headers, body)
    finally:
        client.close()
        server.close()


def test_native_body_coding_preserves_logical_http() -> None:
    keys = generate_key_pair()
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    server = Server(keys.private_key, KEY_ID)
    body = b"request-data-" * 1024
    expected_request = Request(
        method=Method.POST,
        authority="api.example.test",
        path="/compressed",
        headers=(Header("content-length", str(len(body))),),
        body=body,
    )
    protected = client.protect(expected_request)
    assert len(protected.envelope) < len(body)
    opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
    assert opened.request == expected_request
    response_body = b"response-data-" * 1024
    expected_response = Response(
        status=200,
        headers=(Header("content-length", str(len(response_body))),),
        body=response_body,
    )
    encrypted = opened.protect_response(expected_response)
    assert len(encrypted) < len(response_body)
    assert protected.open_response(encrypted) == expected_response
    client.close()
    server.close()


def test_binding_size_guards_preserve_one_shot_consumption() -> None:
    keys = generate_key_pair()
    limits = Limits(max_body_len=1, max_request_bytes=1)
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID, limits=limits)
    server = Server(keys.private_key, KEY_ID, limits=limits)
    with pytest.raises(ProtocolError) as oversized_request:
        client.protect(_request(b"ab"))
    assert oversized_request.value.code == "limit_exceeded"
    with pytest.raises(ProtocolError) as oversized_envelope:
        server.preparse(b"x" * 100_000)
    assert oversized_envelope.value.code == "limit_exceeded"

    protected = client.protect(_request(b""))
    opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
    with pytest.raises(ProtocolError) as oversized_response:
        opened.protect_response(Response(status=200, body=b"ab"))
    assert oversized_response.value.code == "limit_exceeded"
    assert opened.response_consumed
    large_server = Server(keys.private_key, KEY_ID)
    large_opened = large_server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
    valid_large_response = large_opened.protect_response(Response(status=200, body=b"ab"))
    with pytest.raises(ProtocolError) as oversized_protected_response:
        protected.open_response(valid_large_response)
    assert oversized_protected_response.value.code == "limit_exceeded"
    assert protected.consumed
    client.close()
    server.close()
    large_server.close()


def test_replay_rejection_never_releases_plaintext() -> None:
    client, server = _engines()
    protected = client.protect(_request(b"secret"))
    authenticated = server.preparse(protected.envelope).authenticate(PSK)
    with pytest.raises(ProtocolError) as captured:
        authenticated.admit(accepted=False)
    assert captured.value.code == "replay_rejected"


def test_wrong_psk_and_invalid_headers_have_stable_errors() -> None:
    client, server = _engines()
    protected = client.protect(_request())
    with pytest.raises(ProtocolError) as wrong_psk:
        server.preparse(protected.envelope).authenticate(b"wrong credential with enough bytes!!")
    assert wrong_psk.value.code == "authentication_failed"

    invalid = Request(
        method=Method.GET,
        authority="api.example.test",
        path="/",
        headers=(Header("Content-Type", "application/json"),),
    )
    with pytest.raises(ProtocolError) as bad_header:
        client.protect(invalid)
    assert bad_header.value.code == "invalid_configuration"


@pytest.mark.parametrize("value", [-1, 1.5, True, 64 * 1024 * 1024 + 1])
def test_invalid_python_limit_values_have_stable_errors(value: object) -> None:
    key_pair = generate_key_pair()
    with pytest.raises(ProtocolError) as captured:
        Client(
            key_pair.public_key,
            KEY_ID,
            PSK,
            PSK_ID,
            limits=Limits(max_body_len=cast(int, value)),
        )
    assert captured.value.code == "invalid_configuration"


@pytest.mark.parametrize("status", [-1, 199, 1.5, True, 600, 70_000])
def test_invalid_python_status_values_have_stable_errors(status: object) -> None:
    client, server = _engines()
    protected = client.protect(_request())
    opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
    with pytest.raises(ProtocolError) as captured:
        opened.protect_response(Response(status=cast(int, status)))
    assert captured.value.code == "invalid_configuration"


def test_explicit_close_discards_each_live_continuation() -> None:
    client, server = _engines()
    protected = client.protect(_request())
    protected.close()
    assert protected.consumed

    protected = client.protect(_request())
    preparsed = server.preparse(protected.envelope)
    preparsed.close()
    assert preparsed.consumed

    protected = client.protect(_request())
    authenticated = server.preparse(protected.envelope).authenticate(PSK)
    authenticated.close()
    assert authenticated.consumed

    protected = client.protect(_request())
    opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
    opened.close()
    assert opened.response_consumed


def test_client_and_server_close_release_native_handles() -> None:
    client, server = _engines()
    client.close()
    client.close()
    server.close()
    server.close()
    assert client.closed
    assert server.closed

    with pytest.raises(StateError):
        client.protect(_request())
    with pytest.raises(StateError):
        server.preparse(b"not an envelope")


def test_server_close_revokes_pre_auth_but_not_authenticated_work() -> None:
    client, server = _engines()
    pending = server.preparse(client.protect(_request()).envelope)
    server.close()
    assert pending.consumed
    with pytest.raises(StateError, match="server is closed"):
        pending.authenticate(PSK)

    client, server = _engines()
    protected = client.protect(_request())
    authenticated = server.preparse(protected.envelope).authenticate(PSK)
    server.close()
    opened = authenticated.admit(accepted=True)
    response = Response(status=200, body=b"done")
    assert protected.open_response(opened.protect_response(response)) == response


def test_client_and_server_context_managers_close() -> None:
    client, server = _engines()
    with client as entered_client:
        assert entered_client is client
        assert not client.closed
    with server as entered_server:
        assert entered_server is server
        assert not server.closed
    assert client.closed
    assert server.closed
