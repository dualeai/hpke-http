"""
Opt-in logging for HPKE middleware debugging.

Logging is DISABLED by default (NullHandler) - users must explicitly enable it.

Security: Logging is restricted to middleware only. Cryptographic primitives
(primitives/, hpke.py, envelope.py) NEVER log to prevent information leakage.

Usage:
    import logging

    # Enable debug logging during development
    logging.getLogger("hpke_http").setLevel(logging.DEBUG)
    logging.getLogger("hpke_http").addHandler(logging.StreamHandler())
"""

from __future__ import annotations

import logging

__all__ = [
    "get_logger",
]

# Package-level logger (silent by default)
_root_logger = logging.getLogger("hpke_http")
_root_logger.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for an HPKE module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance under the hpke_http namespace
    """
    return logging.getLogger(name)
