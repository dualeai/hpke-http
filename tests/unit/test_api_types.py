"""API type contract tests.

These tests verify that public APIs maintain their type signatures.
Uses typing_extensions.assert_type for STATIC type checking by pyright.

If someone changes a return type, pyright will fail BEFORE tests run.
This prevents the anti-pattern of changing tests to match broken code.
"""

from typing_extensions import assert_type

from hpke_http.streaming import SSEDecryptor, SSEEncryptor, StreamingSession


class TestSSEEncryptorTypes:
    """Verify SSEEncryptor type contracts."""

    def test_encrypt_returns_bytes(self) -> None:
        """SSEEncryptor.encrypt must return bytes (matches ASGI wire format)."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = SSEEncryptor(session)

        result = encryptor.encrypt(b"event: test\n\n")

        # Static assertion - pyright validates at type-check time
        assert_type(result, bytes)
        # Runtime assertion - pytest validates at test time
        assert isinstance(result, bytes)

    def test_encrypt_accepts_bytes(self) -> None:
        """SSEEncryptor.encrypt must accept bytes input."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = SSEEncryptor(session)

        # This should type-check without errors
        chunk: bytes = b"event: test\n\n"
        result = encryptor.encrypt(chunk)
        assert_type(result, bytes)


class TestSSEDecryptorTypes:
    """Verify SSEDecryptor type contracts."""

    def test_decrypt_returns_bytes(self) -> None:
        """SSEDecryptor.decrypt must return bytes (matches native aiohttp)."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Create valid encrypted data
        encrypted = encryptor.encrypt(b"event: test\n\n")
        data_field = encrypted.decode("ascii").split("\n")[1][6:]  # Extract data: field

        result = decryptor.decrypt(data_field)

        # Static assertion - pyright validates at type-check time
        assert_type(result, bytes)
        # Runtime assertion - pytest validates at test time
        assert isinstance(result, bytes)

    def test_decrypt_accepts_str(self) -> None:
        """SSEDecryptor.decrypt must accept str input (base64url encoded)."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        encrypted = encryptor.encrypt(b"test")
        data_field: str = encrypted.decode("ascii").split("\n")[1][6:]

        result = decryptor.decrypt(data_field)
        assert_type(result, bytes)
