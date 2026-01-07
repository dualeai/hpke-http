"""Property-based fuzz tests for HPKE encryption.

Uses Hypothesis to find edge cases in:
- Encrypt/decrypt roundtrip for arbitrary payloads
- AAD binding (wrong AAD must fail)
- Ciphertext integrity (bit flips must fail)
- Key isolation (different keys produce different ciphertexts)
- Sequence counter monotonicity
"""

import secrets

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hpke_http.constants import PSK_MIN_SIZE
from hpke_http.exceptions import DecryptionError, InvalidPSKError, SequenceOverflowError
from hpke_http.hpke import (
    setup_recipient_psk,
    setup_sender_psk,
)
from tests.conftest import extract_sse_data_field


def make_psk(length: int = PSK_MIN_SIZE) -> bytes:
    """Generate a random PSK of specified length."""
    return secrets.token_bytes(length)


# Test constants (avoid hardcoding in every test)
TEST_PSK_ID = b"test-tenant"
TEST_INFO = b"test-info"
TEST_AAD = b"test-aad"

# Strategies for generating test data
binary_payload = st.binary(min_size=0, max_size=1024)
small_binary = st.binary(min_size=1, max_size=64)
aad_data = st.binary(min_size=0, max_size=256)
psk_data = st.binary(min_size=PSK_MIN_SIZE, max_size=64)  # PSK must be >= PSK_MIN_SIZE
psk_id_data = st.binary(min_size=1, max_size=64)
info_data = st.binary(min_size=0, max_size=128)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate X25519 keypair."""
    sk = x25519.X25519PrivateKey.generate()
    return sk.private_bytes_raw(), sk.public_key().public_bytes_raw()


@pytest.mark.fuzz
class TestFuzzRoundtrip:
    """Property: decrypt(encrypt(plaintext)) == plaintext for any payload."""

    @given(plaintext=binary_payload, aad=aad_data)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_roundtrip_any_payload(self, plaintext: bytes, aad: bytes) -> None:
        """Encrypt then decrypt recovers original plaintext."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        # Encrypt
        sender_ctx = setup_sender_psk(pk_r, TEST_INFO, psk, TEST_PSK_ID)
        ciphertext = sender_ctx.seal(aad, plaintext)

        # Decrypt
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, TEST_INFO, psk, TEST_PSK_ID)
        decrypted = recipient_ctx.open(aad, ciphertext)

        assert decrypted == plaintext

    @given(plaintext=binary_payload)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_roundtrip_empty_aad(self, plaintext: bytes) -> None:
        """Roundtrip works with empty AAD."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        sender_ctx = setup_sender_psk(pk_r, b"", psk, TEST_PSK_ID)
        ciphertext = sender_ctx.seal(b"", plaintext)

        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", psk, TEST_PSK_ID)
        decrypted = recipient_ctx.open(b"", ciphertext)

        assert decrypted == plaintext


@pytest.mark.fuzz
class TestFuzzAADBinding:
    """Property: decryption with wrong AAD must fail."""

    @given(plaintext=small_binary, aad1=aad_data, aad2=aad_data)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_wrong_aad_fails(self, plaintext: bytes, aad1: bytes, aad2: bytes) -> None:
        """Decryption with different AAD must fail."""
        if aad1 == aad2:
            return  # Skip when AADs are equal

        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        # Encrypt with aad1
        sender_ctx = setup_sender_psk(pk_r, TEST_INFO, psk, TEST_PSK_ID)
        ciphertext = sender_ctx.seal(aad1, plaintext)

        # Decrypt with aad2 should fail
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, TEST_INFO, psk, TEST_PSK_ID)
        with pytest.raises(DecryptionError):
            recipient_ctx.open(aad2, ciphertext)


@pytest.mark.fuzz
class TestFuzzCiphertextIntegrity:
    """Property: any bit flip in ciphertext must fail authentication."""

    @given(plaintext=small_binary, flip_position=st.integers(min_value=0, max_value=1000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_bit_flip_fails(self, plaintext: bytes, flip_position: int) -> None:
        """Flipping any bit in ciphertext causes decryption failure."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        # Encrypt
        sender_ctx = setup_sender_psk(pk_r, TEST_INFO, psk, TEST_PSK_ID)
        ciphertext = sender_ctx.seal(TEST_AAD, plaintext)

        # Flip a bit
        if len(ciphertext) == 0:
            return  # Can't flip bits in empty ciphertext

        byte_pos = flip_position % len(ciphertext)
        bit_pos = flip_position % 8
        modified = bytearray(ciphertext)
        modified[byte_pos] ^= 1 << bit_pos
        corrupted = bytes(modified)

        # Decryption must fail
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, TEST_INFO, psk, TEST_PSK_ID)
        with pytest.raises(DecryptionError):
            recipient_ctx.open(TEST_AAD, corrupted)


@pytest.mark.fuzz
class TestFuzzKeyIsolation:
    """Property: different keys produce different ciphertexts."""

    @given(plaintext=small_binary)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_different_psk_different_ciphertext(self, plaintext: bytes) -> None:
        """Different PSKs produce different ciphertexts."""
        _sk_r, pk_r = generate_keypair()
        psk1 = make_psk()
        psk2 = make_psk()

        # Same plaintext, different PSKs
        ctx1 = setup_sender_psk(pk_r, TEST_INFO, psk1, TEST_PSK_ID)
        ct1 = ctx1.seal(TEST_AAD, plaintext)

        ctx2 = setup_sender_psk(pk_r, TEST_INFO, psk2, TEST_PSK_ID)
        ct2 = ctx2.seal(TEST_AAD, plaintext)

        # Ciphertexts should differ (different keys)
        assert ct1 != ct2

    @given(plaintext=small_binary)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_different_recipient_different_ciphertext(self, plaintext: bytes) -> None:
        """Different recipient keys produce different ciphertexts."""
        _, pk_r1 = generate_keypair()
        _, pk_r2 = generate_keypair()
        psk = make_psk()

        ctx1 = setup_sender_psk(pk_r1, TEST_INFO, psk, TEST_PSK_ID)
        ctx1.seal(TEST_AAD, plaintext)  # Seal to generate enc

        ctx2 = setup_sender_psk(pk_r2, TEST_INFO, psk, TEST_PSK_ID)
        ctx2.seal(TEST_AAD, plaintext)  # Seal to generate enc

        # enc values differ (different recipient keys)
        assert ctx1.enc != ctx2.enc


@pytest.mark.fuzz
class TestFuzzSequenceMonotonicity:
    """Property: sequence counter always increases, never repeats."""

    @given(message_count=st.integers(min_value=2, max_value=20))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_sequence_monotonic(self, message_count: int) -> None:
        """Sequence counter increases with each seal operation."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        sender_ctx = setup_sender_psk(pk_r, TEST_INFO, psk, TEST_PSK_ID)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, TEST_INFO, psk, TEST_PSK_ID)

        seen_seqs: set[int] = set()
        for i in range(message_count):
            # Record sequence before seal
            seq = sender_ctx.seq
            assert seq == i, f"Expected seq={i}, got {seq}"
            assert seq not in seen_seqs, f"Sequence {seq} repeated!"
            seen_seqs.add(seq)

            # Seal increments sequence
            ct = sender_ctx.seal(b"", b"msg")
            assert sender_ctx.seq == i + 1

            # Recipient sequence also increases
            assert recipient_ctx.seq == i
            recipient_ctx.open(b"", ct)
            assert recipient_ctx.seq == i + 1


@pytest.mark.fuzz
class TestFuzzContextReuse:
    """Property: same context encrypts multiple messages correctly."""

    @given(messages=st.lists(binary_payload, min_size=1, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_multiple_messages_roundtrip(self, messages: list[bytes]) -> None:
        """Multiple messages through same context all decrypt correctly."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        sender_ctx = setup_sender_psk(pk_r, TEST_INFO, psk, TEST_PSK_ID)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, TEST_INFO, psk, TEST_PSK_ID)

        for i, msg in enumerate(messages):
            aad = f"message-{i}".encode()
            ct = sender_ctx.seal(aad, msg)
            decrypted = recipient_ctx.open(aad, ct)
            assert decrypted == msg, f"Message {i} failed roundtrip"


@pytest.mark.fuzz
class TestFuzzSecurityBoundaries:
    """Security boundary tests: overflow, large payloads, edge cases."""

    def test_sequence_overflow_raises_error(self) -> None:
        """Sequence counter overflow must raise SequenceOverflowError.

        Nonce reuse is catastrophic for ChaCha20-Poly1305 - attacker can
        recover plaintext XOR and forge messages.
        """
        _sk_r, pk_r = generate_keypair()
        psk = make_psk()

        sender_ctx = setup_sender_psk(pk_r, TEST_INFO, psk, TEST_PSK_ID)

        # Simulate reaching max sequence (can't actually do 2^96 iterations)
        # Directly set seq to max - 1, then verify overflow on next seal
        # Max is 2^96 - 1 for 12-byte nonce
        max_seq = (1 << 96) - 1
        sender_ctx.seq = max_seq - 1

        # This should work (last valid sequence)
        sender_ctx.seal(b"", b"msg")

        # This should raise SequenceOverflowError
        with pytest.raises(SequenceOverflowError):
            sender_ctx.seal(b"", b"msg")

    @given(payload_size=st.integers(min_value=0, max_value=10 * 1024 * 1024))  # Up to 10MB
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_large_payload_size_range(self, payload_size: int) -> None:
        """Large payloads don't cause issues (memory permitting)."""
        # Only actually allocate for smaller sizes to avoid OOM in CI
        if payload_size > 100 * 1024:  # Skip actual allocation above 100KB
            return

        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        plaintext = b"\x00" * payload_size
        sender_ctx = setup_sender_psk(pk_r, b"", psk, TEST_PSK_ID)
        ciphertext = sender_ctx.seal(b"", plaintext)

        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", psk, TEST_PSK_ID)
        decrypted = recipient_ctx.open(b"", ciphertext)

        assert decrypted == plaintext

    def test_empty_plaintext_roundtrip(self) -> None:
        """Empty plaintext encrypts and decrypts correctly."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        sender_ctx = setup_sender_psk(pk_r, b"", psk, TEST_PSK_ID)
        ciphertext = sender_ctx.seal(b"", b"")

        # Ciphertext should be non-empty (has auth tag)
        assert len(ciphertext) == 16  # Poly1305 tag size

        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", psk, TEST_PSK_ID)
        decrypted = recipient_ctx.open(b"", ciphertext)

        assert decrypted == b""

    def test_max_aad_size(self) -> None:
        """Large AAD (64KB) works correctly."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()

        large_aad = b"X" * (64 * 1024)  # 64KB AAD
        plaintext = b"secret"

        sender_ctx = setup_sender_psk(pk_r, b"", psk, TEST_PSK_ID)
        ciphertext = sender_ctx.seal(large_aad, plaintext)

        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", psk, TEST_PSK_ID)
        decrypted = recipient_ctx.open(large_aad, ciphertext)

        assert decrypted == plaintext

    @given(key_size=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_psk_size_validation(self, key_size: int) -> None:
        """PSK must be at least PSK_MIN_SIZE bytes for security."""
        _sk_r, pk_r = generate_keypair()
        psk = make_psk(key_size) if key_size > 0 else b""

        if key_size < PSK_MIN_SIZE:
            # Should reject weak PSKs
            with pytest.raises(InvalidPSKError):
                setup_sender_psk(pk_r, b"", psk, TEST_PSK_ID)
        else:
            # Should accept strong PSKs
            ctx = setup_sender_psk(pk_r, b"", psk, TEST_PSK_ID)
            assert ctx is not None


@pytest.mark.fuzz
class TestFuzzSSE:
    """Property-based tests for SSE streaming encryption."""

    @given(chunk=st.binary(min_size=0, max_size=1000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_roundtrip_any_chunk(self, chunk: bytes) -> None:
        """Any raw chunk roundtrips correctly."""
        from hpke_http.streaming import ChunkDecryptor, ChunkEncryptor, StreamingSession

        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        encrypted = encryptor.encrypt(chunk)
        data = extract_sse_data_field(encrypted)
        decrypted = decryptor.decrypt(data)

        assert decrypted == chunk

    @given(event_count=st.integers(min_value=1, max_value=50))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_counter_always_increments(self, event_count: int) -> None:
        """Property: counter monotonically increases."""
        from hpke_http.streaming import ChunkDecryptor, ChunkEncryptor, StreamingSession

        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        for i in range(event_count):
            assert encryptor.counter == i + 1
            assert decryptor.expected_counter == i + 1

            encrypted = encryptor.encrypt(f"event: test\ndata: {i}\n\n".encode())
            data = extract_sse_data_field(encrypted)
            decryptor.decrypt(data)

        assert encryptor.counter == event_count + 1
        assert decryptor.expected_counter == event_count + 1

    @given(salt1=st.binary(min_size=4, max_size=4), salt2=st.binary(min_size=4, max_size=4))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_different_salt_different_ciphertext(self, salt1: bytes, salt2: bytes) -> None:
        """Property: different salts produce different ciphertexts."""
        from hpke_http.streaming import ChunkEncryptor, StreamingSession

        # Skip if salts are identical
        if salt1 == salt2:
            return

        key = b"k" * 32
        session1 = StreamingSession(session_key=key, session_salt=salt1)
        session2 = StreamingSession(session_key=key, session_salt=salt2)

        encryptor1 = ChunkEncryptor(session1)
        encryptor2 = ChunkEncryptor(session2)

        sse1 = encryptor1.encrypt(b"event: test\ndata: same\n\n")
        sse2 = encryptor2.encrypt(b"event: test\ndata: same\n\n")

        data1 = extract_sse_data_field(sse1)
        data2 = extract_sse_data_field(sse2)

        # Ciphertexts should differ due to different salts
        assert data1 != data2
