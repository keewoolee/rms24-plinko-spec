"""
Hint data structures for RMS24.

Regular hints: (j, v_hat_j, e_j, P_j, flip) where
- j: unique hint ID
- v_hat_j: median cutoff for block selection
- e_j: extra (block, offset) pair from an unselected block
- P_j: parity (XOR of all entries in the hint's subset)
- flip: whether < is redefined as > for this hint

Backup hints: (j, v_hat_j, P_j, P'_j) where
- P_j: parity of the "selected" half (c/2 blocks)
- P'_j: parity of the "unselected" half (c/2 blocks)
"""

from dataclasses import dataclass, field
from typing import Optional

from .prf import block_selected


@dataclass
class RegHint:
    """
    A regular hint used for online queries.

    The hint's subset contains c/2 + 1 entries:
    - c/2 entries from the "selected" blocks (one per block)
    - 1 extra entry from an "unselected" block
    """

    hint_id: int                    # j - unique identifier
    cutoff: int                     # v_hat_j - median cutoff value
    extra: tuple[int, int]          # e_j - (block, offset) from unselected block
    parity: bytes                   # P_j - XOR of entries in subset
    flip: bool = False              # whether < means > for this hint

    def contains(
        self,
        block: int,
        offset: int,
        prf,  # PRF instance
        w: int,
    ) -> bool:
        """
        Check if this hint's subset contains the given (block, offset).

        Args:
            block: Block index
            offset: Offset within block
            prf: PRF instance for computing selection/offset values
            w: Block size (entries per block)

        Returns:
            True if (block, offset) is in this hint's subset
        """
        # Case 1: entry is the extra
        if (block, offset) == self.extra:
            return True

        # Case 2: block is selected AND offset is the picked entry
        v_jk = prf.select(self.hint_id, block)
        if not block_selected(v_jk, self.cutoff, self.flip):
            return False

        return prf.offset(self.hint_id, block) % w == offset


@dataclass
class BackupHint:
    """
    A backup hint for hint replenishment.

    Stores parities for both halves of the blocks, allowing
    replenishment regardless of which block the queried index
    belongs to.
    """

    hint_id: int                # j - unique identifier
    cutoff: int                 # v_hat_j - median cutoff value
    parity_low: bytes           # P_j - parity of blocks where v < cutoff
    parity_high: bytes          # P'_j - parity of blocks where v >= cutoff


@dataclass
class HintStorage:
    """
    Storage for all hints (regular + backup).
    """

    reg_hints: list[RegHint] = field(default_factory=list)
    backup_hints: list[BackupHint] = field(default_factory=list)

    def find_hint(
        self,
        block: int,
        offset: int,
        prf,  # PRF instance
        w: int,
    ) -> Optional[tuple[int, RegHint]]:
        """
        Find a regular hint whose subset contains the given (block, offset).

        Args:
            block: Block index
            offset: Offset within block
            prf: PRF instance
            w: Block size

        Returns:
            Tuple of (hint_position, hint) if found, None otherwise
        """
        for pos, hint in enumerate(self.reg_hints):
            if hint.contains(block, offset, prf, w):
                return (pos, hint)
        return None

