"""Shared test fixtures for hpke_http tests."""

import asyncio
import contextlib
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import IO, Any

import aiohttp
import httpx
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import PSK_MIN_SIZE
from hpke_http.streaming import ChunkDecryptor, ChunkEncryptor, StreamingSession

# Enable hpke_http debug logging during tests
logging.getLogger("hpke_http").setLevel(logging.DEBUG)
logging.getLogger("hpke_http").addHandler(logging.StreamHandler())


# === Large Payload Test Constants ===

# Parametrized test sizes for large payload tests (slow tests, require large CI runners)
LARGE_PAYLOAD_SIZES_MB = [10, 50, 100, 250, 500, 1024]
LARGE_PAYLOAD_SIZES_IDS = ["10MB", "50MB", "100MB", "250MB", "500MB", "1GB"]


def _is_python_314t_gc_bug() -> bool:
    """Check if running Python 3.14.0-3.14.2 free-threaded (has GC regression).

    Python 3.14t versions before 3.14.3 have a severe GC performance regression
    causing quadratic behavior with large allocations. The fix (gh-142531) was
    merged Dec 12, 2025, after 3.14.2 was released (Dec 5, 2025).

    See: https://github.com/python/cpython/issues/142531
    """
    if sys.version_info[:2] != (3, 14):
        return False
    if sys.version_info.micro > 2:  # 3.14.3+ has the fix
        return False
    # Check if free-threaded (GIL disabled) - _is_gil_enabled only exists in 3.13t+
    return hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()  # noqa: SLF001  # pyright: ignore


# pytest marker for skipping slow tests on Python 3.14.0-3.14.2t
skip_on_314t_gc_bug = pytest.mark.skipif(
    _is_python_314t_gc_bug(),
    reason="Python 3.14.0-3.14.2t has GC regression (gh-142531), fixed in 3.14.3",
)


# === Cryptographic Analysis Helpers ===


def calculate_shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy in bits per byte (0-8 scale)."""
    import math
    from collections import Counter

    if not data:
        return 0.0
    freq = Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def chi_square_byte_uniformity(data: bytes) -> tuple[float, float]:
    """Test if byte distribution is uniform. Returns (chi2, p_value)."""
    import numpy as np
    from scipy import stats  # type: ignore[import-untyped]

    observed = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    expected = len(data) / 256
    chi2_result = stats.chisquare(observed, f_exp=[expected] * 256)  # type: ignore[reportUnknownMemberType]
    return float(chi2_result.statistic), float(chi2_result.pvalue)  # type: ignore[reportUnknownMemberType]


# Chi-square test parameters for uniformity tests
# Even truly random data fails chi-square at rate = threshold (by definition).
# With p > 0.01, expect ~1% false positives per trial. Running multiple trials
# and allowing few failures reduces flakiness.
# This catches real bugs (which fail most/all trials) while tolerating rare statistical outliers.
CHI_SQUARE_TRIALS: int = 10
CHI_SQUARE_MIN_PASS: int = 8
CHI_SQUARE_P_THRESHOLD: float = 0.01


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

    encryptor: ChunkEncryptor
    decryptor: ChunkDecryptor
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
            encryptor=ChunkEncryptor(session, compress=compress),
            decryptor=ChunkDecryptor(session),
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


@dataclass
class E2EServer:
    """E2E test server info with log capture."""

    host: str
    port: int
    public_key: bytes
    _log_file: IO[bytes]

    def get_logs(self) -> str:
        """Read captured server logs."""
        self._log_file.seek(0)
        return self._log_file.read().decode("utf-8", errors="replace")


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
    disable_zstd: bool = False,
) -> AsyncIterator[E2EServer]:
    """Start granian server with HPKE middleware.

    Args:
        platform_keypair: (private_key, public_key) tuple
        test_psk: Pre-shared key
        test_psk_id: Pre-shared key ID
        compress: Enable Zstd compression for SSE responses
        disable_zstd: Simulate zstd being unavailable (for 415 tests)

    Yields:
        E2EServer with host, port, public_key, and log access
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
    if disable_zstd:
        env["HPKE_DISABLE_ZSTD"] = "true"

    # Capture server logs to temp file for per-test debugging
    # Note: intentionally not using context manager - file must stay open across yield
    log_file = tempfile.TemporaryFile(mode="w+b")

    # Use start_new_session=True to create a new process group.
    # This allows us to kill granian and all its child workers together.
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
            "info",
        ],
        env=env,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    def _kill_process_group(sig: int) -> None:
        """Kill the entire process group (granian + workers)."""
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(proc.pid), sig)

    try:
        await wait_for_server(host, port)
        yield E2EServer(host=host, port=port, public_key=pk, _log_file=log_file)
    finally:
        # Kill entire process group to avoid orphaned workers
        _kill_process_group(signal.SIGTERM)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _kill_process_group(signal.SIGKILL)
            proc.wait()
        log_file.close()


@pytest_asyncio.fixture
async def granian_server(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
    request: pytest.FixtureRequest,
) -> AsyncIterator[E2EServer]:
    """Start granian server with HPKE middleware.

    Function-scoped: each test gets its own server with isolated logs.
    Server logs are printed to console after each test.
    """
    async for server in _start_granian_server(platform_keypair, test_psk, test_psk_id):
        yield server
        # Print server logs after test completes
        logs = server.get_logs()
        if logs.strip():
            test_name: str = request.node.name  # type: ignore[attr-defined]
            sys.stdout.write(f"\n{'=' * 60}\n")
            sys.stdout.write(f"Server logs for: {test_name}\n")
            sys.stdout.write(f"{'=' * 60}\n")
            sys.stdout.write(logs)
            sys.stdout.write(f"\n{'=' * 60}\n\n")
            sys.stdout.flush()


@pytest_asyncio.fixture
async def granian_server_compressed(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
    request: pytest.FixtureRequest,
) -> AsyncIterator[E2EServer]:
    """Start granian server with HPKE middleware and compression enabled.

    Function-scoped: each test gets its own server with isolated logs.
    Server logs are printed to console after each test.
    """
    async for server in _start_granian_server(platform_keypair, test_psk, test_psk_id, compress=True):
        yield server
        # Print server logs after test completes
        logs = server.get_logs()
        if logs.strip():
            test_name: str = request.node.name  # type: ignore[attr-defined]
            sys.stdout.write(f"\n{'=' * 60}\n")
            sys.stdout.write(f"Server logs for: {test_name} (compressed)\n")
            sys.stdout.write(f"{'=' * 60}\n")
            sys.stdout.write(logs)
            sys.stdout.write(f"\n{'=' * 60}\n\n")
            sys.stdout.flush()


@pytest_asyncio.fixture
async def granian_server_no_zstd(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
    request: pytest.FixtureRequest,
) -> AsyncIterator[E2EServer]:
    """Start granian server with HPKE middleware and zstd disabled.

    Function-scoped: each test gets its own server with isolated logs.
    Used for testing 415 rejection when client sends zstd-compressed requests.
    """
    async for server in _start_granian_server(platform_keypair, test_psk, test_psk_id, disable_zstd=True):
        yield server
        # Print server logs after test completes
        logs = server.get_logs()
        if logs.strip():
            test_name: str = request.node.name  # type: ignore[attr-defined]
            sys.stdout.write(f"\n{'=' * 60}\n")
            sys.stdout.write(f"Server logs for: {test_name} (no-zstd)\n")
            sys.stdout.write(f"{'=' * 60}\n")
            sys.stdout.write(logs)
            sys.stdout.write(f"\n{'=' * 60}\n\n")
            sys.stdout.flush()


@pytest_asyncio.fixture
async def granian_server_gzip_only(
    platform_keypair: tuple[bytes, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
    request: pytest.FixtureRequest,
) -> AsyncIterator[E2EServer]:
    """Start granian server with compression enabled but zstd unavailable (gzip fallback).

    Function-scoped: each test gets its own server with isolated logs.
    Used for testing compression negotiation with gzip-only server.
    """
    async for server in _start_granian_server(
        platform_keypair, test_psk, test_psk_id, compress=True, disable_zstd=True
    ):
        yield server
        # Print server logs after test completes
        logs = server.get_logs()
        if logs.strip():
            test_name: str = request.node.name  # type: ignore[attr-defined]
            sys.stdout.write(f"\n{'=' * 60}\n")
            sys.stdout.write(f"Server logs for: {test_name} (gzip-only)\n")
            sys.stdout.write(f"{'=' * 60}\n")
            sys.stdout.write(logs)
            sys.stdout.write(f"\n{'=' * 60}\n\n")
            sys.stdout.flush()


# === HPKE Client Fixtures (Separate) ===

# Default timeouts (httpx 5s, aiohttp 300s) are misaligned and can cause flaky
# failures on CI runners during large encrypted uploads. Use 180s for both
# clients to handle 1GB+ payloads with virtualized I/O variance.
_TEST_TIMEOUT_SECS = 180.0

# --- aiohttp fixtures ---


@pytest_asyncio.fixture
async def aiohttp_client(
    granian_server: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """aiohttp HPKEClientSession connected to test server."""
    from hpke_http.middleware.aiohttp import HPKEClientSession

    base_url = f"http://{granian_server.host}:{granian_server.port}"
    async with HPKEClientSession(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        timeout=aiohttp.ClientTimeout(total=_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def aiohttp_client_compressed(
    granian_server_compressed: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """aiohttp HPKEClientSession with compression enabled."""
    from hpke_http.middleware.aiohttp import HPKEClientSession

    base_url = f"http://{granian_server_compressed.host}:{granian_server_compressed.port}"
    async with HPKEClientSession(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        compress=True,
        timeout=aiohttp.ClientTimeout(total=_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def aiohttp_client_small_pool(
    granian_server: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """aiohttp HPKEClientSession with small pool to detect leaks."""
    from hpke_http.middleware.aiohttp import HPKEClientSession

    base_url = f"http://{granian_server.host}:{granian_server.port}"
    connector = aiohttp.TCPConnector(limit=2)
    async with HPKEClientSession(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=5.0),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def aiohttp_client_no_compress_server_compress(
    granian_server_compressed: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """aiohttp client without compression, server with compression."""
    from hpke_http.middleware.aiohttp import HPKEClientSession

    base_url = f"http://{granian_server_compressed.host}:{granian_server_compressed.port}"
    async with HPKEClientSession(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        compress=False,
        timeout=aiohttp.ClientTimeout(total=_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def aiohttp_client_gzip_only(
    granian_server_gzip_only: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """aiohttp client with compression to gzip-only server (no zstd)."""
    from hpke_http.middleware.aiohttp import HPKEClientSession

    base_url = f"http://{granian_server_gzip_only.host}:{granian_server_gzip_only.port}"
    async with HPKEClientSession(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        compress=True,
        timeout=aiohttp.ClientTimeout(total=_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


# --- httpx fixtures ---


@pytest_asyncio.fixture
async def httpx_client(
    granian_server: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """httpx HPKEAsyncClient connected to test server."""
    from hpke_http.middleware.httpx import HPKEAsyncClient

    base_url = f"http://{granian_server.host}:{granian_server.port}"
    async with HPKEAsyncClient(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        timeout=httpx.Timeout(_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def httpx_client_compressed(
    granian_server_compressed: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """httpx HPKEAsyncClient with compression enabled."""
    from hpke_http.middleware.httpx import HPKEAsyncClient

    base_url = f"http://{granian_server_compressed.host}:{granian_server_compressed.port}"
    async with HPKEAsyncClient(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        compress=True,
        timeout=httpx.Timeout(_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def httpx_client_no_compress_server_compress(
    granian_server_compressed: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """httpx client without compression, server with compression."""
    from hpke_http.middleware.httpx import HPKEAsyncClient

    base_url = f"http://{granian_server_compressed.host}:{granian_server_compressed.port}"
    async with HPKEAsyncClient(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        compress=False,
        timeout=httpx.Timeout(_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def httpx_client_gzip_only(
    granian_server_gzip_only: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """httpx client with compression to gzip-only server (no zstd)."""
    from hpke_http.middleware.httpx import HPKEAsyncClient

    base_url = f"http://{granian_server_gzip_only.host}:{granian_server_gzip_only.port}"
    async with HPKEAsyncClient(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        compress=True,
        timeout=httpx.Timeout(_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


# --- release_encrypted fixtures ---


@pytest_asyncio.fixture
async def aiohttp_client_release_encrypted(
    granian_server: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """aiohttp HPKEClientSession with release_encrypted=True."""
    from hpke_http.middleware.aiohttp import HPKEClientSession

    base_url = f"http://{granian_server.host}:{granian_server.port}"
    async with HPKEClientSession(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        release_encrypted=True,
        timeout=aiohttp.ClientTimeout(total=_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def httpx_client_release_encrypted(
    granian_server: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """httpx HPKEAsyncClient with release_encrypted=True."""
    from hpke_http.middleware.httpx import HPKEAsyncClient

    base_url = f"http://{granian_server.host}:{granian_server.port}"
    async with HPKEAsyncClient(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        release_encrypted=True,
        timeout=httpx.Timeout(_TEST_TIMEOUT_SECS),
    ) as client:
        yield client


# --- connection leak testing fixtures ---


@pytest_asyncio.fixture
async def httpx_client_small_pool(
    granian_server: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[Any]:
    """httpx HPKEAsyncClient with small pool to detect leaks."""
    from hpke_http.middleware.httpx import HPKEAsyncClient

    base_url = f"http://{granian_server.host}:{granian_server.port}"
    async with HPKEAsyncClient(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
        limits=httpx.Limits(max_connections=2),
        timeout=httpx.Timeout(5.0),
    ) as client:
        yield client


# === Network Traffic Capture ===


@pytest_asyncio.fixture
async def tcpdump_capture(granian_server: E2EServer) -> AsyncIterator[str]:
    """Capture network traffic during test using tcpdump.

    Auto-skips if tcpdump is not available or lacks permissions.
    """
    pcap_file = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
    pcap_path = pcap_file.name
    pcap_file.close()

    try:
        # Use "tcp port X" to match both IPv4 and IPv6 TCP traffic
        proc = subprocess.Popen(
            [
                "tcpdump",
                "-i",
                "lo0" if sys.platform == "darwin" else "lo",
                "-U",  # Packet-buffered output (flush after each packet)
                "-l",  # Line-buffered output to stderr
                "--immediate-mode",  # Deliver packets immediately, don't buffer
                "-w",
                pcap_path,
                "tcp",
                "port",
                str(granian_server.port),
                "-c",
                "1000",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        os.unlink(pcap_path)
        pytest.skip("tcpdump not installed")
    except PermissionError:
        os.unlink(pcap_path)
        pytest.skip("tcpdump requires root or CAP_NET_RAW capability")

    # Wait for tcpdump to output "listening on" which means BPF filter is ready
    # Set stderr to non-blocking mode
    import fcntl

    stderr_fd = proc.stderr.fileno()  # type: ignore[union-attr]
    flags = fcntl.fcntl(stderr_fd, fcntl.F_GETFL)
    fcntl.fcntl(stderr_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    stderr_output = b""
    for _ in range(30):  # Up to 3 seconds
        await asyncio.sleep(0.1)
        try:
            chunk = os.read(stderr_fd, 4096)
            if chunk:
                stderr_output += chunk
                if b"listening on" in stderr_output:
                    # BPF filter is being set up, wait a bit for it to be fully active
                    # This is a known timing issue with tcpdump startup
                    await asyncio.sleep(0.3)
                    break
        except BlockingIOError:
            pass  # No data available yet

        if proc.poll() is not None:
            # Process exited - check for error
            with contextlib.suppress(BlockingIOError):
                stderr_output += os.read(stderr_fd, 4096)
            with contextlib.suppress(OSError):
                os.unlink(pcap_path)
            pytest.skip(f"tcpdump failed: {stderr_output.decode(errors='replace').strip()}")
    else:
        # Timeout
        proc.terminate()
        proc.wait(timeout=1)
        with contextlib.suppress(OSError):
            os.unlink(pcap_path)
        pytest.skip(f"tcpdump did not start within timeout. Output: {stderr_output.decode(errors='replace')}")

    try:
        yield pcap_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        with contextlib.suppress(OSError):
            os.unlink(pcap_path)
