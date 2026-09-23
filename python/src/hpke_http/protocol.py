"""Stable Python protocol API backed exclusively by the shared Rust engine.

The API accepts complete bounded requests and checked response records.
``ProtocolError.code`` is language-neutral and
safe for control flow; error messages never contain credentials, plaintext,
ciphertext, or parser offsets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import version
from typing import Literal, TypeVar, cast

from typing_extensions import Self

from hpke_http import _native

PROTOCOL_ID = "hpke-http/2"
BINDING_ABI_VERSION = 2
PACKAGE_VERSION = version("hpke_http")

_HARD_LIMITS = (
    ("max_body_len", 64 * 1024 * 1024),
    ("max_header_bytes", 64 * 1024),
    ("max_header_count", 256),
    ("max_target_len", 8 * 1024),
)
_MIN_STATUS = 200
_MAX_STATUS = 599
_DEFAULT_MAX_BODY_LEN = 8 * 1024 * 1024
_DEFAULT_MAX_HEADER_BYTES = 16 * 1024
_ENVELOPE_ALLOWANCE = 64 * 1024


def validate_build_info(engine_version: str, engine_protocol: str, engine_abi: int) -> None:
    """Reject a private extension that does not match this Python package."""
    if engine_version == PACKAGE_VERSION and engine_protocol == PROTOCOL_ID and engine_abi == BINDING_ABI_VERSION:
        return
    msg = (
        "hpke-http native module mismatch: "
        f"expected version={PACKAGE_VERSION!r}, protocol={PROTOCOL_ID!r}, abi={BINDING_ABI_VERSION}; "
        f"got version={engine_version!r}, protocol={engine_protocol!r}, abi={engine_abi}"
    )
    raise ImportError(msg)


validate_build_info(*_native.native_build_info())


class Method(str, Enum):
    """HTTP methods supported by the protocol."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass(frozen=True, slots=True)
class Header:
    """One ordered end-to-end HTTP field.

    ``name`` must be a lower-case HTTP token. ``value`` must be canonical ASCII
    without leading or trailing optional whitespace. Repeated names remain
    separate fields and keep their order.
    """

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class Request:
    """One complete bounded HTTPS request.

    ``authority`` is an RFC 3986 host with an optional decimal port and no user
    information. ``path`` is an absolute path with an optional query, or ``*``
    for ``OPTIONS``. The Rust engine validates all fields before protection.
    """

    method: Method
    authority: str
    path: str
    headers: tuple[Header, ...] = ()
    body: bytes = b""


@dataclass(frozen=True, slots=True)
class Response:
    """One complete bounded HTTP response.

    ``status`` must be from 200 through 599. ``HEAD`` responses and statuses
    204, 205, and 304 cannot contain a body.
    """

    status: int
    headers: tuple[Header, ...] = ()
    body: bytes = b""


@dataclass(frozen=True, slots=True)
class Limits:
    """Optional limits passed to the native engine.

    ``None`` selects the native default. Body bytes default to 8 MiB with a
    64 MiB hard maximum. Header bytes default to 16 KiB with a 64 KiB hard
    maximum. Header count defaults to 64 with a hard maximum of 256. Combined
    authority and path bytes default to, and cannot exceed, 8 KiB.
    """

    max_body_len: int | None = None
    max_header_bytes: int | None = None
    max_header_count: int | None = None
    max_target_len: int | None = None


_DEFAULT_LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class KeyPair:
    """Encoded X25519 recipient key pair.

    Both values are owned Python ``bytes``. Python cannot erase immutable byte
    strings in place, so move ``private_key`` into protected application storage
    and avoid retaining unnecessary copies.
    """

    private_key: bytes
    public_key: bytes


class ProtocolError(Exception):
    """Stable protocol error with a language-neutral ``code``.

    Configuration and parsing codes are ``invalid_configuration``,
    ``limit_exceeded``, ``malformed_envelope``, ``unsupported_version``,
    ``unsupported_suite``, and ``unsupported_method``. Credential and
    authentication codes are ``unknown_recipient_key``, ``invalid_credential``,
    and ``authentication_failed``. Replay and time codes are
    ``replay_rejected``, ``invalid_request_time``,
    ``replay_decision_mismatch``, and ``clock_unavailable``. Platform and local
    failures are ``entropy_unavailable``,
    ``crypto_failure``, and ``compression_failure``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _bounded_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ProtocolError("invalid_configuration", f"{name} is outside the supported range")
    return value


def _native_limits(limits: Limits) -> tuple[int | None, int | None, int | None, int | None]:
    values = (
        limits.max_body_len,
        limits.max_header_bytes,
        limits.max_header_count,
        limits.max_target_len,
    )
    validated = tuple(
        None if value is None else _bounded_integer(value, name, 0, maximum)
        for (name, maximum), value in zip(_HARD_LIMITS, values, strict=True)
    )
    return cast(tuple[int | None, int | None, int | None, int | None], validated)


def _maximum_body_len(limits: Limits) -> int:
    return limits.max_body_len if limits.max_body_len is not None else _DEFAULT_MAX_BODY_LEN


def _maximum_envelope_len(limits: Limits) -> int:
    headers = limits.max_header_bytes if limits.max_header_bytes is not None else _DEFAULT_MAX_HEADER_BYTES
    return _maximum_body_len(limits) + headers + _ENVELOPE_ALLOWANCE


def _check_length(length: int, maximum: int) -> None:
    if length > maximum:
        raise ProtocolError("limit_exceeded", "limit exceeded")


class StateError(ProtocolError):
    """A native handle was closed or a one-shot continuation was consumed."""

    def __init__(self, message: str = "state already consumed or closed") -> None:
        super().__init__("state_consumed", message)


def generate_key_pair() -> KeyPair:
    """Generate a new X25519 recipient key pair with platform secure entropy.

    Raises:
        ProtocolError: If the platform cannot provide secure entropy.
    """
    private_key, public_key = _call(_native.native_generate_key_pair)
    return KeyPair(private_key=private_key, public_key=public_key)


class Client:
    """Reusable client configuration for one recipient and PSK identity.

    The recipient public key is an encoded 32-byte X25519 key. Both public IDs
    are non-empty and at most 255 bytes. The PSK is at least 32 bytes, and
    ``psk_id`` must not equal it. Inputs are copied into native storage; closing
    the client cannot clear byte strings retained by the caller. ``compression``
    opts into Rust-owned gzip/zstd body coding and requires an enabled server.
    Leave it off when a body combines secrets with attacker-controlled data.
    """

    __slots__ = ("_inner", "_maximum_body")

    def __init__(
        self,
        recipient_public_key: bytes,
        recipient_key_id: bytes,
        psk: bytes,
        psk_id: bytes,
        *,
        limits: Limits = _DEFAULT_LIMITS,
        compression: Literal["gzip", "zstd"] | None = None,
    ) -> None:
        native_limits = _native_limits(limits)
        self._inner: _native.Client | None = _call(
            _native.Client,
            bytes(recipient_public_key),
            bytes(recipient_key_id),
            bytes(psk),
            bytes(psk_id),
            native_limits,
            compression,
        )
        self._maximum_body = _maximum_body_len(limits)

    def protect(self, request: Request) -> ProtectedRequest:
        """Protect one request and return its one-shot response transaction.

        Each call creates a fresh envelope. A transport retry must call this
        method again rather than resend an earlier envelope.
        """
        inner = self._inner
        if inner is None:
            raise StateError("client is closed")
        _check_length(len(request.body), self._maximum_body)
        native = _call(
            inner.protect,
            request.method.value,
            _ascii(request.authority, "authority"),
            _ascii(request.path, "path"),
            _export_headers(request.headers),
            bytes(request.body),
        )
        return ProtectedRequest(native)

    @property
    def closed(self) -> bool:
        """Whether the native configuration and its PSK were released."""
        return self._inner is None

    def close(self) -> None:
        """Release the native configuration and its zeroized PSK storage."""
        self._inner = None

    def __enter__(self) -> Self:
        if self.closed:
            raise StateError("client is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class ProtectedRequest:
    """Protected request bytes plus one response reader right.

    Call :meth:`close` in a ``finally`` block when transport can fail before
    :meth:`open_response` or :meth:`into_opener` takes the response right.
    After :meth:`into_opener`, close the reader instead.
    """

    __slots__ = ("_envelope", "_inner")

    def __init__(self, inner: _native.ProtectedRequest) -> None:
        self._inner = inner
        self._envelope = inner.take_envelope()

    @property
    def envelope(self) -> bytes:
        """Return the owned protected request envelope."""
        return self._envelope

    @property
    def consumed(self) -> bool:
        """Whether the response continuation is no longer usable."""
        return self._inner.consumed

    def open_response(self, envelope: bytes) -> Response:
        """Check a complete finite reply with the v2 record reader."""
        status, headers, body = _call(self._inner.open_finite_response, bytes(envelope))
        return Response(status=status, headers=_import_headers(headers), body=body)

    def into_opener(self) -> ResponseOpener:
        """Transfer the one response right to a live record reader."""
        return ResponseOpener(_call(self._inner.take_opener))

    def close(self) -> None:
        """Discard the response continuation now."""
        self._inner.discard()


class _ServerState:
    """Shared close cell used to revoke every pending pre-auth continuation."""

    __slots__ = ("inner", "maximum_body", "maximum_envelope")

    def __init__(self, inner: _native.Server, limits: Limits) -> None:
        self.inner: _native.Server | None = inner
        self.maximum_body = _maximum_body_len(limits)
        self.maximum_envelope = _maximum_envelope_len(limits)

    @property
    def closed(self) -> bool:
        return self.inner is None

    def require(self) -> _native.Server:
        inner = self.inner
        if inner is None:
            raise StateError("server is closed")
        return inner

    def close(self) -> None:
        self.inner = None


class Server:
    """Reusable server configuration for one static recipient key.

    The private key is an encoded 32-byte X25519 key. ``recipient_key_id`` is
    public, non-empty, and at most 255 bytes. Closing the server releases its
    native key copy and revokes pending pre-authentication stages. Set
    ``compression=True`` to accept opt-in gzip/zstd protocol bodies.
    """

    __slots__ = ("_state",)

    def __init__(
        self,
        recipient_private_key: bytes,
        recipient_key_id: bytes,
        *,
        limits: Limits = _DEFAULT_LIMITS,
        compression: bool = False,
    ) -> None:
        native_limits = _native_limits(limits)
        self._state = _ServerState(
            _call(
                _native.Server,
                bytes(recipient_private_key),
                bytes(recipient_key_id),
                native_limits,
                compression,
            ),
            limits,
        )

    def preparse(self, envelope: bytes) -> PreparsedRequest:
        """Read bounded public fields before credential lookup."""
        inner = self._state.require()
        _check_length(len(envelope), self._state.maximum_envelope)
        return PreparsedRequest(_call(inner.preparse, bytes(envelope)), self._state)

    @property
    def closed(self) -> bool:
        """Whether the native recipient-key configuration was released."""
        return self._state.closed

    def close(self) -> None:
        """Release the native recipient-key configuration."""
        self._state.close()

    def __enter__(self) -> Self:
        if self.closed:
            raise StateError("server is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class PreparsedRequest:
    """One-shot continuation waiting for host credential resolution.

    Only the public PSK ID is available before authentication. Close the stage
    if credential lookup cannot complete.
    """

    __slots__ = ("_inner", "_server_state")

    def __init__(self, inner: _native.PreparsedRequest, server_state: _ServerState) -> None:
        self._inner = inner
        self._server_state: _ServerState | None = server_state

    @property
    def psk_id(self) -> bytes:
        """Return the public PSK identity for the host resolver."""
        return self._inner.psk_id

    @property
    def consumed(self) -> bool:
        """Whether this continuation is no longer usable."""
        state = self._server_state
        return self._inner.consumed or state is None or state.closed

    def authenticate(self, psk: bytes) -> AuthenticatedRequest:
        """Consume this stage and authenticate the complete bounded request."""
        state = self._server_state
        if state is None:
            raise StateError("continuation already consumed")
        server = state.require()
        self._server_state = None
        native = _call(self._inner.authenticate, server, bytes(psk))
        return AuthenticatedRequest(native, state.maximum_body)

    def close(self) -> None:
        """Discard this continuation now."""
        self._server_state = None
        self._inner.discard()


class AuthenticatedRequest:
    """Authenticated request paused for atomic replay admission.

    Plaintext remains inside the native engine. The host must atomically reserve
    ``replay_id`` if absent, keep it through ``retain_until_exclusive``, and pass
    an uncertain store result as ``accepted=False``.
    """

    __slots__ = ("_inner", "_maximum_body")

    def __init__(self, inner: _native.AuthenticatedRequest, maximum_body: int) -> None:
        self._inner = inner
        self._maximum_body = maximum_body

    @property
    def replay_id(self) -> bytes:
        """Return the stable 32-byte admission key for this request attempt."""
        return self._inner.replay_id

    @property
    def retain_until_exclusive(self) -> int:
        """Return the Unix second at which the replay reservation may expire."""
        return self._inner.retain_until_exclusive

    @property
    def consumed(self) -> bool:
        """Whether this continuation is no longer usable."""
        return self._inner.consumed

    def admit(self, *, accepted: bool) -> OpenedRequest:
        """Consume the stage, apply one replay decision, and release plaintext."""
        native = _call(self._inner.admit, accepted)
        return OpenedRequest(native, self._maximum_body)

    def close(self) -> None:
        """Discard this continuation now."""
        self._inner.discard()


class OpenedRequest:
    """Verified plaintext request plus its one-shot response protector.

    ``request`` is safe for application dispatch. Protect one response or call
    :meth:`close` to discard the response capability.
    """

    __slots__ = ("_inner", "_maximum_body", "request")

    def __init__(self, inner: _native.OpenedRequest, maximum_body: int) -> None:
        self._inner = inner
        self._maximum_body = maximum_body
        self.request = Request(
            method=Method(inner.method),
            authority=inner.authority.decode("ascii"),
            path=inner.path.decode("ascii"),
            headers=_import_headers(inner.headers),
            body=inner.take_body(),
        )

    @property
    def response_consumed(self) -> bool:
        """Whether the response capability is no longer usable."""
        return self._inner.response_consumed

    def protect_response(self, response: Response) -> bytes:
        """Protect one complete finite reply with the v2 record writer."""
        status = _bounded_integer(response.status, "response status", _MIN_STATUS, _MAX_STATUS)
        if len(response.body) > self._maximum_body:
            self.close()
            raise ProtocolError("limit_exceeded", "limit exceeded")
        return _call(
            self._inner.protect_response,
            status,
            _export_headers(response.headers),
            bytes(response.body),
        )

    def into_sealer(self, status: int, headers: Sequence[Header]) -> tuple[ResponseSealer, bytes]:
        """Start a checked reply and return its prefix and START record."""
        checked_status = _bounded_integer(status, "response status", _MIN_STATUS, _MAX_STATUS)
        inner, first = _call(self._inner.take_sealer, checked_status, _export_headers(headers))
        return ResponseSealer(inner), first

    def close(self) -> None:
        """Discard the response capability now."""
        self._inner.discard_response()


@dataclass(frozen=True, slots=True)
class CheckedRecord:
    """One checked response record. Only SSE DATA has clear block bytes."""

    kind: Literal["start", "data", "end"]
    status: int = 0
    headers: tuple[Header, ...] = ()
    mode: Literal["finite", "sse", ""] = ""
    block: bytes = b""


class ResponseOpener:
    """One live response reader; input can end at any byte."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _native.ResponseOpener) -> None:
        self._inner: _native.ResponseOpener | None = inner

    def feed(self, data: bytes, offset: int = 0) -> tuple[int, CheckedRecord | None]:
        """Read at most one record from ``data[offset:]``.

        The byte count is relative to ``offset``; add it to ``offset`` before
        the next call. Finite DATA stays private.
        """
        inner = self._inner
        if inner is None:
            raise StateError()
        used, record = _call(inner.feed, bytes(data), offset)
        if record is None:
            return used, None
        kind, status, headers, mode, block = record
        return used, CheckedRecord(
            kind=cast("Literal['start', 'data', 'end']", kind),
            status=status,
            headers=_import_headers(headers),
            mode=cast("Literal['finite', 'sse', '']", mode),
            block=block,
        )

    def finish_eof(self) -> Response | None:
        """Check END at true outer body EOF and return a finite response."""
        inner = self._inner
        if inner is None:
            raise StateError()
        result = _call(inner.finish_eof)
        self._inner = None
        if result is None:
            return None
        status, headers, body = result
        return Response(status=status, headers=_import_headers(headers), body=body)

    def close(self) -> None:
        """Stop the response check without claiming a complete reply."""
        if self._inner is not None:
            self._inner.close()
            self._inner = None


class ResponseSealer:
    """One checked response writer for a finite body or clear SSE blocks."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _native.ResponseSealer) -> None:
        self._inner: _native.ResponseSealer | None = inner

    def seal_finite_body(self, body: bytes) -> bytes | None:
        """Protect a finite body once; an empty body emits no DATA record."""
        inner = self._inner
        if inner is None:
            raise StateError()
        return _call(inner.seal_finite_body, bytes(body))

    def seal_sse_block(self, block: bytes) -> bytes:
        """Protect one complete LF-normalized SSE block."""
        inner = self._inner
        if inner is None:
            raise StateError()
        return _call(inner.seal_sse_block, bytes(block))

    def finish(self) -> bytes:
        """Protect END; the outer sender must then end its HTTP body."""
        inner = self._inner
        if inner is None:
            raise StateError()
        result = _call(inner.finish)
        self._inner = None
        return result

    def close(self) -> None:
        """Stop without an END record."""
        if self._inner is not None:
            self._inner.close()
            self._inner = None


class SseSplitter:
    """Split app bytes into complete LF-normalized SSE blocks."""

    __slots__ = ("_inner",)

    def __init__(self, max_block_len: int) -> None:
        self._inner: _native.SseSplitter | None = _native.SseSplitter(max_block_len)

    def feed(self, data: bytes, offset: int = 0, max_bytes: int | None = None) -> tuple[int, bytes | None]:
        """Read at most ``max_bytes`` bytes from ``data[offset:]`` and return at most one block.

        The byte count is relative to ``offset``; add it to ``offset`` before
        the next call. Without ``max_bytes``, the input limit is the rest of
        ``data``.
        """
        inner = self._inner
        if inner is None:
            raise StateError()
        return _call(inner.feed, bytes(data), offset, max_bytes)

    def finish(self) -> None:
        if self._inner is not None:
            self._inner.finish()
            self._inner = None

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None


_T = TypeVar("_T")


def _call(function: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    try:
        return function(*args, **kwargs)
    except _native.NativeError as error:
        values = cast(tuple[object, ...], error.args)
        code = str(values[0]) if values else "native_error"
        message = str(values[1]) if len(values) > 1 else str(error)
        raise ProtocolError(code, message) from None
    except RuntimeError as error:
        if str(error) == "continuation already consumed":
            raise StateError("continuation already consumed") from None
        raise


def _ascii(value: str, field: str) -> bytes:
    try:
        return value.encode("ascii")
    except UnicodeEncodeError as error:
        msg = f"{field} must contain ASCII characters only"
        raise ValueError(msg) from error


def _export_headers(headers: Sequence[Header]) -> list[tuple[str, str]]:
    return [(field.name, field.value) for field in headers]


def _import_headers(headers: Sequence[tuple[str, str]]) -> tuple[Header, ...]:
    return tuple(Header(name=name, value=value) for name, value in headers)


__all__ = [
    "BINDING_ABI_VERSION",
    "PACKAGE_VERSION",
    "PROTOCOL_ID",
    "AuthenticatedRequest",
    "CheckedRecord",
    "Client",
    "Header",
    "KeyPair",
    "Limits",
    "Method",
    "OpenedRequest",
    "PreparsedRequest",
    "ProtectedRequest",
    "ProtocolError",
    "Request",
    "Response",
    "ResponseOpener",
    "ResponseSealer",
    "Server",
    "SseSplitter",
    "StateError",
    "generate_key_pair",
]
