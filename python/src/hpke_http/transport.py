"""Shared HTTP transport rules for native protocol adapters."""

from __future__ import annotations

from collections.abc import Iterable

from hpke_http.protocol import Header, Limits

REQUEST_MEDIA_TYPE = "message/hpke-http-request"
RESPONSE_MEDIA_TYPE = "message/hpke-http-response"

_DEFAULT_MAX_BODY_LEN = 8 * 1024 * 1024
_DEFAULT_MAX_HEADER_BYTES = 16 * 1024
_ENVELOPE_OVERHEAD = 64 * 1024
_NON_FORWARDABLE_FIELDS = frozenset(
    {
        "connection",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authentication-info",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_REQUEST_ONLY_NON_FORWARDABLE_FIELDS = frozenset({"accept-encoding", "content-length", "expect"})


class TransportError(RuntimeError):
    """The HTTP adapter could not produce a supported logical exchange.

    ``code`` is stable for adapter control flow. Current codes cover invalid
    targets, unsupported logical content coding, request body size, network
    failure, and invalid outer status, media type, or content coding. Key GET
    failures use ``discovery_network``, ``discovery_status``, or
    ``discovery_response``. ``status_code`` is set only for an invalid outer
    or discovery HTTP status.
    Protocol failures, including response record limits, use
    :class:`hpke_http.ProtocolError` instead.
    """

    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def filter_request_headers(fields: Iterable[tuple[str, str]]) -> tuple[Header, ...]:
    """Drop request transport fields and preserve ordered end-to-end fields."""
    return _filter_headers(fields, _NON_FORWARDABLE_FIELDS | _REQUEST_ONLY_NON_FORWARDABLE_FIELDS)


def filter_response_headers(fields: Iterable[tuple[str, str]]) -> tuple[Header, ...]:
    """Drop response transport fields while preserving response metadata."""
    return _filter_headers(fields, _NON_FORWARDABLE_FIELDS)


def _filter_headers(fields: Iterable[tuple[str, str]], denied: frozenset[str]) -> tuple[Header, ...]:
    normalized = tuple((name.lower(), value) for name, value in fields)
    for name, value in normalized:
        if name == "content-encoding" and any(coding.strip().lower() != "identity" for coding in value.split(",")):
            raise TransportError(
                "inner_content_encoding",
                "logical HTTP messages support only identity content coding",
            )
    connection_options = {
        option.strip().lower()
        for name, value in normalized
        if name == "connection"
        for option in value.split(",")
        if option.strip()
    }
    excluded = denied | connection_options
    return tuple(Header(name, value) for name, value in normalized if name not in excluded)


def media_type(value: str | None) -> str:
    """Return a normalized media type without optional parameters."""
    if value is None:
        return ""
    return value.partition(";")[0].strip().lower()


def max_envelope_len(limits: Limits) -> int:
    """Bound an outer envelope before passing it to the native parser."""
    body = limits.max_body_len if limits.max_body_len is not None else _DEFAULT_MAX_BODY_LEN
    headers = limits.max_header_bytes if limits.max_header_bytes is not None else _DEFAULT_MAX_HEADER_BYTES
    return body + headers + _ENVELOPE_OVERHEAD


def max_body_len(limits: Limits) -> int:
    """Return the effective plaintext body limit used by adapters."""
    return limits.max_body_len if limits.max_body_len is not None else _DEFAULT_MAX_BODY_LEN


__all__ = [
    "REQUEST_MEDIA_TYPE",
    "RESPONSE_MEDIA_TYPE",
    "TransportError",
]
