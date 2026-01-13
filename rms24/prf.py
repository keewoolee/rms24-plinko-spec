"""
Pseudorandom Function (PRF) implementation for RMS24.

Uses AES-ECB as a PRF to generate deterministic pseudorandom values.
Two types of PRF calls are used:
- PRF("select" || j || k): Determines if block k is selected by hint j
- PRF("offset" || j || k): Determines which index is picked from block k for hint j
"""

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import struct


class PRF:
    """
    AES-based Pseudorandom Function.

    Generates deterministic pseudorandom values from (prefix, hint_id, block_id).
    """

    # Prefixes for different PRF uses
    PREFIX_SELECT = b"select"
    PREFIX_OFFSET = b"offset"

    def __init__(self, key: bytes = None):
        """
        Initialize PRF with a secret key.

        Args:
            key: 16-byte AES key. If None, generates a random key.
        """
        if key is None:
            key = get_random_bytes(16)
        if len(key) != 16:
            raise ValueError("Key must be 16 bytes")
        self.key = key

    def _evaluate(self, prefix: bytes, hint_id: int, block_id: int) -> int:
        """
        Core PRF evaluation.

        Args:
            prefix: Domain separation prefix ("select" or "offset")
            hint_id: The hint ID (j)
            block_id: The block ID (k)

        Returns:
            32-bit pseudorandom value
        """
        # Build input: prefix || hint_id (4 bytes) || block_id (4 bytes)
        # Pad to 16 bytes for AES block
        data = prefix + struct.pack("<II", hint_id, block_id)
        data = data.ljust(16, b"\x00")

        # Use AES-ECB for PRF (deterministic)
        cipher = AES.new(self.key, AES.MODE_ECB)
        output = cipher.encrypt(data)

        # Extract 32-bit value from output
        return struct.unpack("<I", output[:4])[0]

    def select(self, hint_id: int, block_id: int) -> int:
        """
        PRF("select" || j || k) - value for block selection.

        Used to determine if block k is in the "selected" half for hint j.
        Compare with median cutoff to decide selection.

        Args:
            hint_id: Hint ID (j)
            block_id: Block ID (k)

        Returns:
            32-bit pseudorandom value v_{j,k}
        """
        return self._evaluate(self.PREFIX_SELECT, hint_id, block_id)

    def offset(self, hint_id: int, block_id: int) -> int:
        """
        PRF("offset" || j || k) - offset within block.

        Used to determine which index is picked from block k for hint j.
        The actual index is: k * w + (r_{j,k} mod w).

        Args:
            hint_id: Hint ID (j)
            block_id: Block ID (k)

        Returns:
            32-bit pseudorandom value r_{j,k}
        """
        return self._evaluate(self.PREFIX_OFFSET, hint_id, block_id)

    def select_vector(self, hint_id: int, num_blocks: int) -> list[int]:
        """
        Compute the full selection vector V_j for a hint.

        Args:
            hint_id: Hint ID (j)
            num_blocks: c (number of blocks)

        Returns:
            List of v_{j,k} for k in [0, num_blocks)
        """
        return [self.select(hint_id, k) for k in range(num_blocks)]


def find_median_cutoff(values: list[int]) -> int:
    """
    Find the median cutoff value that divides values into two equal halves.

    Since c is even, the median is the average of the two middle elements.

    Args:
        values: List of PRF values (length must be even)

    Returns:
        Median cutoff value v_hat such that exactly len(values)/2 elements
        are smaller than v_hat.
    """
    if len(values) % 2 != 0:
        raise ValueError("Length must be even")

    sorted_values = sorted(values)
    mid = len(values) // 2

    # Cutoff is average of the two middle elements
    # This ensures exactly mid elements are < cutoff
    return (sorted_values[mid - 1] + sorted_values[mid]) // 2


def block_selected(v_jk: int, cutoff: int, flip: bool = False) -> bool:
    """
    Check if a block is selected by a hint.

    Args:
        v_jk: PRF value v_{j,k}
        cutoff: Median cutoff v_hat_j
        flip: If True, use >= instead of <

    Returns:
        True if the block is in the "selected" half
    """
    if flip:
        return v_jk > cutoff
    return v_jk < cutoff
