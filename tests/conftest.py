"""Shared test fixtures for hpke_http tests."""

import asyncio
import logging
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import aiohttp
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import PSK_MIN_SIZE
from hpke_http.streaming import SSEDecryptor, SSEEncryptor, StreamingSession

# Enable hpke_http debug logging during tests
logging.getLogger("hpke_http").setLevel(logging.DEBUG)
logging.getLogger("hpke_http").addHandler(logging.StreamHandler())

# === Key Fixtures ===


@pytest.fixture(scope="session")
def platform_keypair() -> tuple[bytes, bytes]:
    """Generate a platform X25519 keypair for testing.

    Session-scoped: one keypair per xdist worker, shared across all tests.
    """
    private_key = x25519.X25519PrivateKey.generate()
    return (
        private_key.private_bytes_raw(),
        private_key.public_key().public_bytes_raw(),
    )


@pytest.fixture
def client_keypair() -> tuple[bytes, bytes]:
    """Generate a client X25519 keypair for testing."""
    private_key = x25519.X25519PrivateKey.generate()
    return (
        private_key.private_bytes_raw(),
        private_key.public_key().public_bytes_raw(),
    )


# === PSK Fixtures ===


@pytest.fixture
def psk_factory() -> Callable[[int], bytes]:
    """Factory for generating PSKs of specified length.

    Usage:
        def test_something(psk_factory):
            psk = psk_factory(32)  # Generate 32-byte PSK
            psk_64 = psk_factory(64)  # Generate 64-byte PSK
    """

    def _make_psk(length: int = PSK_MIN_SIZE) -> bytes:
        return secrets.token_bytes(length)

    return _make_psk


@pytest.fixture(scope="session")
def test_psk() -> bytes:
    """Test pre-shared key (API key). Fixed value for deterministic tests.

    Session-scoped: shared with granian_server fixture.
    """
    return b"test-api-key-for-hpke-psk-mode!!"  # 32 bytes


@pytest.fixture(scope="session")
def test_psk_id() -> bytes:
    """Test pre-shared key ID (tenant ID).

    Session-scoped: shared with granian_server fixture.
    """
    return b"tenant-123"


@pytest.fixture
def wrong_psk() -> bytes:
    """A valid-length PSK that differs from test_psk.

    Use for testing decryption failures due to wrong key (not invalid key).
    """
    return b"wrong-key-but-still-32-bytes!!!!"  # 32 bytes, different from test_psk


@pytest.fixture
def wrong_psk_id() -> bytes:
    """A PSK ID that differs from test_psk_id."""
    return b"wrong-tenant"


# === SSE Test Utilities ===


def extract_sse_data_field(sse: bytes) -> str:
    """Extract data field from encrypted SSE output.

    Args:
        sse: Raw SSE event bytes (e.g., b"event: enc\\ndata: <payload>\\n\\n")

    Returns:
        The base64url-encoded payload string

    Raises:
        ValueError: If no data field found
    """
    for line in sse.decode("ascii").split("\n"):
        if line.startswith("data: "):
            return line[6:]
    raise ValueError("No data field found in SSE event")


def make_sse_session(
    session_key: bytes = b"k" * 32,
    session_salt: bytes = b"salt",
) -> StreamingSession:
    """Create a deterministic SSE session for testing.

    Args:
        session_key: 32-byte key (default: b"k" * 32)
        session_salt: 4-byte salt (default: b"salt")

    Returns:
        StreamingSession with specified parameters
    """
    return StreamingSession(session_key=session_key, session_salt=session_salt)


@dataclass
class SSETestPair:
    """Matched SSE encryptor/decryptor pair for testing.

    Provides a convenient way to create encrypt/decrypt pairs with
    optional warmup for memory measurement tests.

    Example:
        pair = SSETestPair.create(warmup_count=1)
        encrypted = pair.encryptor.encrypt(b"test")
        decrypted = pair.decryptor.decrypt(extract_sse_data_field(encrypted))
    """

    encryptor: SSEEncryptor
    decryptor: SSEDecryptor
    session: StreamingSession

    @classmethod
    def create(
        cls,
        *,
        compress: bool = False,
        warmup_count: int = 0,
        session_key: bytes = b"k" * 32,
        session_salt: bytes = b"salt",
    ) -> "SSETestPair":
        """Create a matched encryptor/decryptor pair.

        Args:
            compress: Enable Zstd compression for encryptor
            warmup_count: Number of warmup roundtrips to perform
            session_key: 32-byte session key
            session_salt: 4-byte session salt

        Returns:
            SSETestPair ready for use
        """
        session = StreamingSession(session_key=session_key, session_salt=session_salt)
        pair = cls(
            encryptor=SSEEncryptor(session, compress=compress),
            decryptor=SSEDecryptor(session),
            session=session,
        )
        if warmup_count > 0:
            pair.warmup(warmup_count)
        return pair

    def warmup(self, count: int = 1, chunk: bytes = b"warmup") -> None:
        """Perform warmup roundtrips to initialize cipher internals.

        Args:
            count: Number of roundtrips
            chunk: Data to encrypt/decrypt
        """
        for _ in range(count):
            encrypted = self.encryptor.encrypt(chunk)
            data = extract_sse_data_field(encrypted)
            self.decryptor.decrypt(data)

    def roundtrip(self, chunk: bytes) -> bytes:
        """Encrypt then decrypt a chunk.

        Args:
            chunk: Data to roundtrip

        Returns:
            Decrypted data (should equal input)
        """
        encrypted = self.encryptor.encrypt(chunk)
        data = extract_sse_data_field(encrypted)
        return self.decryptor.decrypt(data)


@pytest.fixture
def sse_session() -> StreamingSession:
    """Fixture providing a deterministic SSE session."""
    return make_sse_session()


@pytest.fixture
def sse_pair() -> SSETestPair:
    """Fixture providing a matched SSE encryptor/decryptor pair."""
    return SSETestPair.create()


@pytest.fixture
def sse_pair_warmed() -> SSETestPair:
    """Fixture providing a warmed-up SSE pair (for memory tests)."""
    return SSETestPair.create(warmup_count=1)


# === E2E Server Fixtures ===


def get_free_port() -> int:
    """Get a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


async def wait_for_server(host: str, port: int, timeout: float = 10.0) -> None:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{host}:{port}/health") as resp:
                    if resp.status == 200:
                        return
        except (aiohttp.ClientError, OSError):
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Server not ready after {timeout}s")


# Server module path for granian
TEST_SERVER_MODULE = "tests.e2e_server:app"


async def _start_granian_server(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
    *,
    compress: bool = False,
) -> AsyncIterator[tuple[str, int, bytes]]:
    """Start granian server with HPKE middleware.

    Args:
        platform_keypair: (private_key, public_key) tuple
        test_psk: Pre-shared key
        test_psk_id: Pre-shared key ID
        compress: Enable Zstd compression for SSE responses

    Yields:
        Tuple of (host, port, public_key)
    """
    sk, pk = platform_keypair
    port = get_free_port()
    host = "127.0.0.1"

    # Start granian as subprocess with env vars for config
    env = {
        **dict(os.environ),
        "TEST_HPKE_PRIVATE_KEY": sk.hex(),
        "TEST_PSK": test_psk.hex(),
        "TEST_PSK_ID": test_psk_id.hex(),
    }
    if compress:
        env["TEST_COMPRESS"] = "true"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "granian",
            TEST_SERVER_MODULE,
            "--interface",
            "asgi",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            "1",
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        await wait_for_server(host, port)
        yield (host, port, pk)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def granian_server(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[tuple[str, int, bytes]]:
    """Start granian server with HPKE middleware.

    Session-scoped: one server per xdist worker, shared across all tests.
    Eliminates port reuse issues and speeds up test execution.
    """
    async for result in _start_granian_server(platform_keypair, test_psk, test_psk_id):
        yield result


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def granian_server_compressed(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[tuple[str, int, bytes]]:
    """Start granian server with HPKE middleware and compression enabled.

    Session-scoped: one server per xdist worker, shared across all tests.
    """
    async for result in _start_granian_server(platform_keypair, test_psk, test_psk_id, compress=True):
        yield result
