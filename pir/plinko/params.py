"""
Parameters for the Plinko PIR scheme.

Key parameters:
- num_entries: Total number of database entries
- block_size: Entries per block, defaults to ⌈√num_entries⌉
- num_blocks: Number of blocks, must be even
- security_param: Security parameter controlling failure probability
- entry_size: Size of each database entry in bytes
- num_reg_hints: Number of regular hints
- num_backup_hints: Number of backup hints (default num_reg_hints)

OPT Tradeoffs:
- Query size grows with num_blocks
- Hint storage grows with num_reg_hints (= security_param * block_size)
- num_backup_hints determines queries per offline phase
- Default block_size = √num_entries balances query size and hint storage
"""

from dataclasses import dataclass
import math


@dataclass
class Params:
    """Parameters for Plinko PIR scheme."""

    num_entries: int  # Number of database entries
    entry_size: int  # Size of each entry in bytes
    # OPT: 128 is conservative; could decouple computational and statistical security parameters
    security_param: int = 128  # Query failure probability ≈ 2^{-security_param}
    block_size: int | None = None  # Entries per block
    num_backup_hints: int | None = None  # Configurable backup hints

    def __post_init__(self):
        # Validate parameters
        if self.num_entries < 1:
            raise ValueError("num_entries must be at least 1")
        if self.entry_size < 1:
            raise ValueError("entry_size must be at least 1")
        if self.security_param < 1:
            raise ValueError("security_param must be at least 1")

        # SEC: block_size must be a power of 2 to avoid modular bias in offset_in_block.
        # Default: ceil(sqrt(num_entries)), rounded up to power of 2.
        if self.block_size is None:
            raw_block_size = math.ceil(math.sqrt(self.num_entries))
            self.block_size = 1 << (raw_block_size - 1).bit_length()
        else:
            # Validate user-provided block_size is power of 2
            if self.block_size < 1:
                raise ValueError("block_size must be at least 1")
            if (self.block_size & (self.block_size - 1)) != 0:
                raise ValueError("block_size must be a power of 2")

        # Number of blocks = ceil(num_entries / block_size)
        self._num_blocks = math.ceil(self.num_entries / self.block_size)

        # SEC: num_blocks must be even: each hint splits blocks into two equal halves.
        # Query sends offsets that could belong to either half, hiding the real query.
        if self._num_blocks % 2 == 1:
            self._num_blocks += 1

        # Number of regular hints
        # This ensures high probability of finding a hint for any target
        self._num_reg_hints = self.security_param * self.block_size

        # Default num_backup_hints: num_reg_hints
        if self.num_backup_hints is None:
            self.num_backup_hints = self._num_reg_hints

        if self.num_backup_hints < 0:
            raise ValueError("num_backup_hints must be non-negative")

    @property
    def num_blocks(self) -> int:
        """Number of blocks. Always even."""
        return self._num_blocks

    @property
    def num_reg_hints(self) -> int:
        """Number of regular hints."""
        return self._num_reg_hints

    @property
    def num_total_hints(self) -> int:
        """Total number of hints (regular + backup)."""
        return self._num_reg_hints + self.num_backup_hints

    def block_of(self, index: int) -> int:
        """Return the block index for a given database index."""
        return index // self.block_size

    def offset_in_block(self, index: int) -> int:
        """Return the offset within the block for a given index."""
        return index % self.block_size

    def index_from_block_offset(self, block: int, offset: int) -> int:
        """Compute database index from block and offset."""
        return block * self.block_size + offset

    @property
    def select_output_bits(self) -> int:
        """Number of bits for PRF select output.

        Computed as ceil(log2(num_blocks)) + 8. Collision probability at
        cutoff boundary is ~num_blocks / 2^output_bits ≈ 2^-8 per hint.
        """
        return self.num_blocks.bit_length() + 8

    def __repr__(self) -> str:
        return (
            f"Params(num_entries={self.num_entries}, "
            f"block_size={self.block_size}, num_blocks={self.num_blocks}, "
            f"num_reg_hints={self.num_reg_hints}, "
            f"num_backup_hints={self.num_backup_hints}, "
            f"entry_size={self.entry_size}, security_param={self.security_param})"
        )
