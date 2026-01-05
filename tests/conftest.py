"""Shared test fixtures for hpke_http tests."""

import asyncio
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable

import aiohttp
import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import PSK_MIN_SIZE

# === Key Fixtures ===


@pytest.fixture
def platform_keypair() -> tuple[bytes, bytes]:
    """Generate a platform X25519 keypair for testing."""
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


@pytest.fixture
def test_psk() -> bytes:
    """Test pre-shared key (API key). Fixed value for deterministic tests."""
    return b"test-api-key-for-hpke-psk-mode!!"  # 32 bytes


@pytest.fixture
def test_psk_id() -> bytes:
    """Test pre-shared key ID (tenant ID)."""
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


@pytest.fixture
async def granian_server(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[tuple[str, int, bytes]]:
    """Start granian server with HPKE middleware.

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
            # Force kill if terminate doesn't work
            proc.kill()
            proc.wait()  # No timeout after SIGKILL
        proc = None
