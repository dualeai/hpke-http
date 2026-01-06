"""
RFC 9180 §4 Key Derivation Functions.

Implements LabeledExtract and LabeledExpand for HKDF-SHA256.

Reference: https://datatracker.ietf.org/doc/rfc9180/ §4
"""

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

from hpke_http.constants import HKDF_SHA256_HASH_SIZE, SUITE_ID, VERSION

__all__ = [
    "extract_and_expand",
    "labeled_expand",
    "labeled_extract",
]


def labeled_extract(
    salt: bytes,
    label: bytes,
    ikm: bytes,
    suite_id: bytes = SUITE_ID,
) -> bytes:
    """
    RFC 9180 §4: LabeledExtract(salt, label, ikm).

    Produces a pseudorandom key from input keying material using HKDF-Extract
    with a labeled input.

    labeled_ikm = "HPKE-v1" || suite_id || label || ikm
    return Extract(salt, labeled_ikm)

    Args:
        salt: Optional salt value (can be empty bytes)
        label: ASCII label for domain separation
        ikm: Input keying material
        suite_id: HPKE suite identifier (default: our cipher suite)

    Returns:
        Pseudorandom key (32 bytes for HKDF-SHA256)
    """
    labeled_ikm = VERSION + suite_id + label + ikm

    # HKDF-Extract is defined as: PRK = HMAC-Hash(salt, IKM)
    # RFC 5869 §2.2: if salt is empty, use a string of HashLen zeros
    actual_salt = salt if salt else b"\x00" * HKDF_SHA256_HASH_SIZE
    h = hmac.HMAC(actual_salt, hashes.SHA256())
    h.update(labeled_ikm)
    return h.finalize()


def labeled_expand(
    prk: bytes,
    label: bytes,
    info: bytes,
    length: int,
    suite_id: bytes = SUITE_ID,
) -> bytes:
    """
    RFC 9180 §4: LabeledExpand(prk, label, info, L).

    Expands a pseudorandom key to the desired length using HKDF-Expand
    with a labeled info string.

    labeled_info = I2OSP(L, 2) || "HPKE-v1" || suite_id || label || info
    return Expand(prk, labeled_info, L)

    Args:
        prk: Pseudorandom key from LabeledExtract
        label: ASCII label for domain separation
        info: Context information
        length: Desired output length in bytes
        suite_id: HPKE suite identifier (default: our cipher suite)

    Returns:
        Expanded key material of specified length
    """
    labeled_info = length.to_bytes(2, "big") + VERSION + suite_id + label + info

    hkdf = HKDFExpand(
        algorithm=hashes.SHA256(),
        length=length,
        info=labeled_info,
    )
    return hkdf.derive(prk)


def extract_and_expand(
    salt: bytes,
    label_extract: bytes,
    ikm: bytes,
    label_expand: bytes,
    info: bytes,
    length: int,
    suite_id: bytes = SUITE_ID,
) -> bytes:
    """
    Convenience function: LabeledExtract followed by LabeledExpand.

    Args:
        salt: Salt for Extract
        label_extract: Label for Extract step
        ikm: Input keying material
        label_expand: Label for Expand step
        info: Context info for Expand
        length: Output length
        suite_id: HPKE suite identifier

    Returns:
        Expanded key material
    """
    prk = labeled_extract(salt, label_extract, ikm, suite_id)
    return labeled_expand(prk, label_expand, info, length, suite_id)
