"""Unit tests for HPKEMiddleware non-HTTP scope handling and SSE body paths.

NOTE: These tests mock the ASGI interface because HTTP test clients CANNOT
send non-HTTP ASGI scopes (websocket, lifespan). This is a fundamental
protocol limitation, not a testing convenience choice.

SSE body path tests exercise _handle_sse_body via _create_encrypting_send to
cover all three code paths (A/B/C) including the Path C regression where
more_body=False was not forwarded when events were already sent and flush()
returned empty.

All HTTP request scenarios (including malformed requests) are tested via
E2E with real granian server in test_middleware.py and test_malformed_requests.py.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from starlette.types import Send

from hpke_http.constants import (
    SCOPE_HPKE_CONTEXT,
    SSE_MAX_EVENT_SIZE,
    KemId,
)
from hpke_http.exceptions import DecryptionError
from hpke_http.hpke import setup_recipient_psk, setup_sender_psk
from hpke_http.middleware.fastapi import HPKEMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_sse_send(
    *,
    max_sse_event_size: int = SSE_MAX_EVENT_SIZE,
) -> tuple[Send, AsyncMock]:
    """Create an SSE-ready encrypting_send with real crypto and a mock send.

    Returns (encrypting_send, send_mock) after initializing SSE state via
    a http.response.start with content-type: text/event-stream.
    """
    private_key = X25519PrivateKey.generate()
    sk = private_key.private_bytes_raw()
    pk = private_key.public_key().public_bytes_raw()
    psk = b"test-api-key-for-hpke-psk-mode!!"
    psk_id = b"tenant-123"

    sender_ctx = setup_sender_psk(pk, psk_id, psk, psk_id)
    recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk, psk_id, psk, psk_id)

    middleware = HPKEMiddleware(
        app=AsyncMock(),
        private_keys={KemId.DHKEM_X25519_HKDF_SHA256: sk},
        psk_resolver=AsyncMock(),
        max_sse_event_size=max_sse_event_size,
    )
    scope: dict[str, Any] = {"type": "http", SCOPE_HPKE_CONTEXT: recipient_ctx}
    send_mock = AsyncMock()
    encrypting_send = middleware._create_encrypting_send(scope, send_mock)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    # Initialize SSE state by sending response start
    await encrypting_send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    return encrypting_send, send_mock


def _body_calls(send_mock: AsyncMock) -> list[dict[str, Any]]:
    """Extract http.response.body messages from send mock."""
    return [call.args[0] for call in send_mock.call_args_list if call.args[0].get("type") == "http.response.body"]


def _assert_stream_terminated(send_mock: AsyncMock, expected_events: int) -> None:
    """Assert stream has N encrypted events + 1 terminal more_body=False."""
    calls = _body_calls(send_mock)
    assert len(calls) == expected_events + 1
    # All intermediate calls: encrypted data, more_body=True
    for c in calls[:-1]:
        assert c["more_body"] is True
        assert c["body"] != b""  # encrypted, non-empty
    # Final call: empty body, more_body=False
    assert calls[-1]["more_body"] is False
    assert calls[-1]["body"] == b""


# ---------------------------------------------------------------------------
# Non-HTTP scopes (existing)
# ---------------------------------------------------------------------------


class TestNonHTTPScopes:
    """Test middleware behavior with non-HTTP ASGI scopes.

    NOTE: Cannot test via E2E - HTTP test clients only send HTTP requests.
    WebSocket and lifespan are different ASGI scope types that require
    direct ASGI interface testing.
    """

    async def test_websocket_scope_passes_through(self) -> None:
        """WebSocket scope should pass through unchanged.

        HPKE middleware only handles HTTP - WebSocket encryption would need
        a separate implementation.
        """
        app = AsyncMock()
        middleware = HPKEMiddleware(
            app=app,
            private_keys={KemId.DHKEM_X25519_HKDF_SHA256: b"x" * 32},
            psk_resolver=AsyncMock(),
        )

        scope: dict[str, Any] = {"type": "websocket", "path": "/ws"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        # App should be called directly without HPKE processing
        app.assert_called_once_with(scope, receive, send)

    async def test_lifespan_scope_passes_through(self) -> None:
        """Lifespan scope should pass through unchanged.

        Lifespan events (startup/shutdown) don't carry request data.
        """
        app = AsyncMock()
        middleware = HPKEMiddleware(
            app=app,
            private_keys={KemId.DHKEM_X25519_HKDF_SHA256: b"x" * 32},
            psk_resolver=AsyncMock(),
        )

        scope: dict[str, Any] = {"type": "lifespan"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        app.assert_called_once_with(scope, receive, send)


# ---------------------------------------------------------------------------
# SSE stream termination — Path C regression tests
# (All FAIL before fix, PASS after)
# ---------------------------------------------------------------------------


class TestSSEStreamTermination:
    """Path C regression: more_body=False must always be forwarded."""

    async def test_single_event_with_end(self) -> None:
        """Single complete event + more_body=False in one message."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"data: x\n\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=1)

    async def test_multiple_events_single_body_with_end(self) -> None:
        """Two complete events in one message + more_body=False."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send(
            {
                "type": "http.response.body",
                "body": b"data: 1\n\ndata: 2\n\n",
                "more_body": False,
            }
        )
        _assert_stream_terminated(send_mock, expected_events=2)

    async def test_partial_then_complete_with_end(self) -> None:
        """Partial chunk, then completion + more_body=False."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"data: hel", "more_body": True})
        await encrypting_send({"type": "http.response.body", "body": b"lo\n\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=1)

    async def test_streamed_events_final_has_data(self) -> None:
        """Multi-message stream where final message has an event + more_body=False."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"data: 1\n\n", "more_body": True})
        await encrypting_send({"type": "http.response.body", "body": b"data: 2\n\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=2)


# ---------------------------------------------------------------------------
# All three SSE paths — safety nets (PASS before and after fix)
# ---------------------------------------------------------------------------


class TestSSEAllPaths:
    """Exercise paths A, B, and edge cases that pass regardless of fix."""

    async def test_path_a_flush_has_remainder(self) -> None:
        """Path A: flush() returns non-empty — encrypted with more_body=False."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send(
            {
                "type": "http.response.body",
                "body": b"data: x\n\ntrailing",
                "more_body": False,
            }
        )
        calls = _body_calls(send_mock)
        # 1 event (data: x) encrypted + 1 flush remainder encrypted with more_body=False
        assert len(calls) == 2
        assert calls[0]["more_body"] is True
        assert calls[0]["body"] != b""
        assert calls[1]["more_body"] is False
        assert calls[1]["body"] != b""  # encrypted remainder, not empty

    async def test_path_b_empty_stream(self) -> None:
        """Path B: empty body + more_body=False — no events, terminal sent."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"", "more_body": False})
        calls = _body_calls(send_mock)
        assert len(calls) == 1
        assert calls[0]["more_body"] is False
        assert calls[0]["body"] == b""

    async def test_path_b_starlette_pattern(self) -> None:
        """Starlette's actual pattern: data then empty terminator."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"data: x\n\n", "more_body": True})
        await encrypting_send({"type": "http.response.body", "body": b"", "more_body": False})
        calls = _body_calls(send_mock)
        # 1 encrypted event + 1 empty terminal
        assert len(calls) == 2
        assert calls[0]["more_body"] is True
        assert calls[0]["body"] != b""
        assert calls[1]["more_body"] is False
        assert calls[1]["body"] == b""

    async def test_more_body_key_missing_defaults_false(self) -> None:
        """ASGI spec: missing more_body key defaults to False."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"data: x\n\n"})
        _assert_stream_terminated(send_mock, expected_events=1)


# ---------------------------------------------------------------------------
# Weird but valid ASGI/SSE patterns
# ---------------------------------------------------------------------------


class TestSSEWeirdCases:
    """Unusual but valid ASGI message sequences and SSE content."""

    async def test_body_only_newlines(self) -> None:
        """Empty SSE event (just boundary \\n\\n)."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"\n\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=1)

    async def test_crlf_boundaries(self) -> None:
        """WHATWG SSE allows \\r\\n\\r\\n as event boundary."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"data: x\r\n\r\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=1)

    async def test_keepalive_comment(self) -> None:
        """SSE comment (: prefix) is a valid event."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b":ping\n\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=1)

    async def test_split_boundary_across_chunks(self) -> None:
        """\\n\\n split across two ASGI messages."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"data: x\n", "more_body": True})
        await encrypting_send({"type": "http.response.body", "body": b"\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=1)

    async def test_empty_intermediates_then_event(self) -> None:
        """Framework sends empty chunks before actual data."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send({"type": "http.response.body", "body": b"", "more_body": True})
        await encrypting_send({"type": "http.response.body", "body": b"", "more_body": True})
        await encrypting_send({"type": "http.response.body", "body": b"data: x\n\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=1)


# ---------------------------------------------------------------------------
# DoS and stress
# ---------------------------------------------------------------------------


class TestSSEOutOfBounds:
    """DoS protection and high-volume scenarios."""

    async def test_dos_exceeds_max_event_size(self) -> None:
        """Exceeding max_sse_event_size raises DecryptionError."""
        encrypting_send, _ = await _make_sse_send(max_sse_event_size=100)
        with pytest.raises(DecryptionError, match="SSE event too large"):
            await encrypting_send(
                {
                    "type": "http.response.body",
                    "body": b"x" * 200,
                    "more_body": False,
                }
            )

    async def test_dos_buffer_resets_after_event(self) -> None:
        """Buffer size resets between events — no false DoS trigger across messages."""
        encrypting_send, send_mock = await _make_sse_send(max_sse_event_size=100)
        # First message: 52 bytes, under 100. Event parsed → buffer resets to 0.
        await encrypting_send({"type": "http.response.body", "body": b"d" * 50 + b"\n\n", "more_body": True})
        # Second message: 52 bytes, under 100 because buffer was reset.
        await encrypting_send({"type": "http.response.body", "body": b"e" * 50 + b"\n\n", "more_body": False})
        _assert_stream_terminated(send_mock, expected_events=2)

    async def test_hundred_events_single_body(self) -> None:
        """100 events in one message — all encrypted + terminal."""
        encrypting_send, send_mock = await _make_sse_send()
        await encrypting_send(
            {
                "type": "http.response.body",
                "body": b"data: x\n\n" * 100,
                "more_body": False,
            }
        )
        _assert_stream_terminated(send_mock, expected_events=100)
