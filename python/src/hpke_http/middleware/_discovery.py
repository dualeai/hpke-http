"""Shared key discovery, client, and fixed HTTPS origin rules."""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

import idna

from hpke_http.middleware._native_async import run_native
from hpke_http.protocol import (
    Client,
    Limits,
    ProtectedRequest,
    ProtocolError,
    Request,
    _native_limits,  # pyright: ignore[reportPrivateUsage]
)
from hpke_http.transport import TransportError, media_type

KEY_MEDIA_TYPE = "application/octet-stream"
MAX_KEY_RECORD = 293
_KEY_OK_STATUS = 200
_MAX_ID_LEN = 255
_PUBLIC_KEY_LEN = 32
_MIN_KEY_RECORD = 39


@dataclass(frozen=True, slots=True)
class Discover:
    """Fetch the current public key before each protected call."""


@dataclass(frozen=True, slots=True)
class PinnedKey:
    """Use one public key without a discovery GET."""

    public_key: bytes
    key_id: bytes


def validate_client_configuration(
    psk: bytes, psk_id: bytes, limits: Limits, compression: Literal["gzip", "zstd"] | None
) -> None:
    """Reject local client faults before a discovery GET."""
    _native_limits(limits)
    if len(psk) < _PUBLIC_KEY_LEN or not 1 <= len(psk_id) <= _MAX_ID_LEN or psk == psk_id:
        raise ProtocolError("invalid_configuration", "invalid PSK or PSK identity")
    if compression not in (None, "gzip", "zstd"):
        raise ProtocolError("invalid_configuration", "unsupported protocol body coding")


def encode_key_record(key_id: bytes, public_key: bytes) -> bytes:
    """Encode one fixed X25519 key record."""
    if not 1 <= len(key_id) <= _MAX_ID_LEN or len(public_key) != _PUBLIC_KEY_LEN:
        raise ValueError("invalid discovery key")
    return b"HHKD\x01" + bytes((len(key_id),)) + key_id + public_key


def parse_key_record(record: bytes) -> tuple[bytes, bytes]:
    """Reject all records except the exact one-key format."""
    if (
        not _MIN_KEY_RECORD <= len(record) <= MAX_KEY_RECORD
        or record[:5] != b"HHKD\x01"
        or record[5] == 0
        or len(record) != 38 + record[5]
    ):
        raise TransportError("discovery_response", "key endpoint returned an invalid key record")
    end = 6 + record[5]
    return record[6:end], record[end:]


def validate_key_response(status: int, content_types: Sequence[str], content_encodings: Sequence[str]) -> None:
    """Check one key GET response before reading its raw body."""
    if status != _KEY_OK_STATUS:
        raise TransportError("discovery_status", "key endpoint returned an invalid status", status_code=status)
    if len(content_types) != 1 or media_type(content_types[0]) != KEY_MEDIA_TYPE:
        raise TransportError("discovery_response", "key endpoint returned an invalid content type")
    if len(content_encodings) > 1 or (content_encodings and content_encodings[0].lower() != "identity"):
        raise TransportError("discovery_response", "key endpoint returned encoded key bytes")


async def read_key_record(chunks: AsyncIterable[bytes]) -> tuple[bytes, bytes]:
    """Read at most one bounded key record from raw response chunks."""
    record = bytearray()
    async for chunk in chunks:
        if len(chunk) > MAX_KEY_RECORD - len(record):
            raise TransportError("discovery_response", "key record exceeds 293 bytes")
        record.extend(chunk)
    return parse_key_record(bytes(record))


def make_discovered_client(
    public_key: bytes,
    key_id: bytes,
    psk: bytes,
    psk_id: bytes,
    limits: Limits,
    compression: Literal["gzip", "zstd"] | None,
) -> Client:
    """Map an unusable key from the GET to a discovery response fault."""
    try:
        return Client(public_key, key_id, psk, psk_id, limits=limits, compression=compression)
    except ProtocolError as error:
        raise TransportError("discovery_response", "key endpoint returned an unusable public key") from error


async def protect_discovered_client(client: Client, request: Request) -> ProtectedRequest:
    """Protect one request and close a temporary discovered client."""
    try:
        try:
            return await run_native(client.protect, request)
        except ProtocolError as error:
            if error.code in ("invalid_configuration", "crypto_failure"):
                raise TransportError("discovery_response", "key endpoint returned an unusable public key") from error
            raise
    finally:
        client.close()


def https_origin(value: str, *, endpoint: bool = False) -> tuple[str, int]:
    """Return the canonical HTTPS host and port, using IDNA2008 for DNS names."""
    if type(value) is not str or not value or any(char.isspace() or char == "\\" for char in value):
        raise TransportError("invalid_target", "an absolute HTTPS URL is required")
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() != "https" or not parts.netloc or parts.username is not None:
            raise ValueError
        if endpoint:
            if "?" in value or "#" in value:
                raise ValueError
        elif parts.path not in ("", "/") or "?" in value or "#" in value:
            raise ValueError
        host = parts.hostname
        if host is None or "%" in host:
            raise ValueError
        try:
            normalized_host = ipaddress.ip_address(host).compressed.lower()
        except ValueError:
            normalized_host = idna.encode(host.lower()).decode("ascii")
            if not normalized_host or any(label == "" for label in normalized_host.split(".")[:-1]):
                raise ValueError from None
        port = parts.port
        if port is None:
            port = 443
        if port == 0:
            raise ValueError
    except (UnicodeError, ValueError) as error:
        raise TransportError("invalid_target", "an absolute HTTPS URL without credentials is required") from error
    return normalized_host, port


def validate_endpoint(value: str) -> str:
    """Require one fixed HTTPS URL without credentials, query, or fragment."""
    https_origin(value, endpoint=True)
    return value


def validate_target_origin(value: str | None, endpoint: str) -> tuple[str, int]:
    """Return the one logical origin accepted by an adapter."""
    return https_origin(endpoint, endpoint=True) if value is None else https_origin(value)


def same_origin(url: str, origin: tuple[str, int]) -> bool:
    """Check a logical HTTPS URL before its body is read."""
    try:
        parts = urlsplit(url)
        return https_origin(f"https://{parts.netloc}/") == origin and parts.scheme.lower() == "https"
    except TransportError:
        return False
