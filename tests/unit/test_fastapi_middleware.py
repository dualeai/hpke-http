"""Unit tests for HPKEMiddleware non-HTTP scope handling.

NOTE: These tests mock the ASGI interface because HTTP test clients CANNOT
send non-HTTP ASGI scopes (websocket, lifespan). This is a fundamental
protocol limitation, not a testing convenience choice.

All HTTP request scenarios (including malformed requests) are tested via
E2E with real granian server in test_middleware.py and test_malformed_requests.py.
"""

from typing import Any
from unittest.mock import AsyncMock

from hpke_http.constants import KemId
from hpke_http.middleware.fastapi import HPKEMiddleware


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
