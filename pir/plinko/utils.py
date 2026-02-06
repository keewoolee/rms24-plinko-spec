"""
Utility functions for Plinko PIR scheme.

Includes:
- PRF for block selection and offset computation
- PRG (Pseudorandom Generator) for deterministic random sampling
- derive_seed for key derivation
- find_cutoff for hint generation
- xor_bytes and zero_entry utilities
"""

import hashlib
import secrets
import struct


class PRF:
    """
    SHAKE-256 based PRF for Plinko PIR.

    Used for block selection (determining which blocks are in a hint's subset)
    and offset computation within blocks.

    OPT: Could use a faster PRF (e.g., AES-based) and batch evaluate.

    SEC: The PRF key is client-secret and must never be shared with
    the server. The server learns nothing about which entries the client
    accesses as long as the key remains secret.
    """

    def __init__(self):
        """Initialize PRF with a randomly generated secret key."""
        self._key = secrets.token_bytes(32)

    def select(self, hint_id: int, block: int, output_bits: int) -> int:
        """
        Return pseudorandom value for block selection.

        Args:
            hint_id: Hint identifier
            block: Block index
            output_bits: Number of bits in output (1-256)

        Returns:
            Value in [0, 2^output_bits)
        """
        output_bytes = (output_bits + 7) // 8
        # SEC: Domain separation between select and offset
        data = b"select" + struct.pack("<II", hint_id, block)
        output = hashlib.shake_256(self._key + data).digest(output_bytes)
        value = int.from_bytes(output, "little")
        return value & ((1 << output_bits) - 1)

    def offset(self, hint_id: int, block: int, block_size: int) -> int:
        """
        Return pseudorandom offset within a block.

        Args:
            hint_id: Hint identifier
            block: Block index
            block_size: Size of block (must be power of 2)

        Returns:
            Value in [0, block_size)
        """
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError("block_size must be a power of 2")
        output_bits = block_size.bit_length() - 1
        output_bytes = (output_bits + 7) // 8
        data = b"offset" + struct.pack("<II", hint_id, block)
        output = hashlib.shake_256(self._key + data).digest(output_bytes)
        value = int.from_bytes(output, "little")
        return value & (block_size - 1)

    def select_vector(self, hint_id: int, num_blocks: int, output_bits: int) -> list[int]:
        """Compute select values for all blocks."""
        return [self.select(hint_id, k, output_bits) for k in range(num_blocks)]


def find_cutoff(values: list[int], select_count: int) -> int:
    """
    Find cutoff that selects exactly select_count values.

    Values with PRF < cutoff are selected.

    Args:
        values: List of PRF select values
        select_count: Number of values to select

    Returns:
        Cutoff such that exactly select_count elements are smaller.

    Note:
        Returns 0 if there's a collision at the selection boundary, meaning
        the values at positions (select_count-1) and (select_count) are equal.
        This makes it impossible to select exactly select_count values.
        Collision probability is approximately len(values) / 2^output_bits.
        Callers should handle the 0 return value appropriately.
    """
    if select_count <= 0:
        return 0

    # OPT: Full sort is O(n log n). A selection algorithm like quickselect
    # or numpy.partition would be O(n) average.
    sorted_values = sorted(values)

    # Edge case: select all values
    if select_count >= len(values):
        return sorted_values[-1] + 1

    # Check for collision at the boundary
    if sorted_values[select_count - 1] == sorted_values[select_count]:
        return 0

    return sorted_values[select_count]


def derive_seed(key: bytes, label: bytes) -> bytes:
    """
    Derive a deterministic seed from key and label using SHAKE-256.

    Returns 32 bytes suitable for seeding random sampling.
    """
    return hashlib.shake_256(key + label).digest(32)


class PRG:
    """
    Pseudorandom Number Generator using SHAKE-256 in counter mode.

    OPT: Could use a faster PRG (e.g., AES-CTR).

    Deterministic: same seed produces same sequence.
    Each call to random_bytes() produces output based on the seed
    and an internal counter.
    """

    def __init__(self, seed: bytes):
        """
        Initialize with a seed.

        Args:
            seed: Seed bytes (typically 32 bytes from derive_seed)
        """
        self._seed = seed
        self._counter = 0

    def random_bytes(self, n: int) -> bytes:
        """Get n random bytes."""
        data = hashlib.shake_256(
            self._seed + self._counter.to_bytes(8, "little")
        ).digest(n)
        self._counter += 1
        return data

    def random_float(self) -> float:
        """Get a random float in [0, 1)."""
        return int.from_bytes(self.random_bytes(8), "little") / (2**64)

    def binomial(self, n: int, p: float) -> int:
        """Sample from Binomial(n, p).

        OPT: O(n) implementation. For large n, could use normal approximation
        or BTPE algorithm for O(1) expected time.
        """
        if n == 0 or p == 0.0:
            return 0
        if p == 1.0:
            return n

        count = 0
        for _ in range(n):
            if self.random_float() < p:
                count += 1
        return count


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings of equal length.

    OPT: Could use SIMD (e.g., AVX2/AVX-512) for bulk XOR.
    """
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    return bytes(x ^ y for x, y in zip(a, b))


def zero_entry(entry_size: int) -> bytes:
    """Create a zero-filled entry."""
    return bytes(entry_size)
