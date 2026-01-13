"""
Hint data structures for RMS24.

Regular hints: (j, v_hat_j, e_j, P_j, flip) where
- j: unique hint ID
- v_hat_j: median cutoff for block selection
- e_j: extra index (one index from an unselected block)
- P_j: parity (XOR of all entries in the hint's subset)
- flip: whether < is redefined as > for this hint

Backup hints: (j, v_hat_j, P_j, P'_j) where
- P_j: parity of the "selected" half (c/2 blocks)
- P'_j: parity of the "unselected" half (c/2 blocks)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegHint:
    """
    A regular hint used for online queries.

    The hint's subset contains c/2 + 1 indices:
    - c/2 indices from the "selected" blocks (one per block)
    - 1 extra index from an "unselected" block
    """

    hint_id: int                # j - unique identifier
    cutoff: int                 # v_hat_j - median cutoff value
    extra_index: int            # e_j - extra index from unselected block
    parity: bytes               # P_j - XOR of entries in subset
    flip: bool = False          # whether < means > for this hint

    def contains_index(
        self,
        index: int,
        prf,  # PRF instance
        params,  # Params instance
    ) -> bool:
        """
        Check if this hint's subset contains the given index.

        Args:
            index: Database index to check
            prf: PRF instance for computing selection/offset values
            params: Parameters

        Returns:
            True if index is in this hint's subset
        """
        # Case 1: index is the extra index
        if index == self.extra_index:
            return True

        # Case 2: index's block is selected AND index is the picked index
        block = params.block_of(index)
        offset = params.offset_in_block(index)

        # Check if block is selected
        v_jk = prf.select(self.hint_id, block)
        if self.flip:
            is_selected = v_jk > self.cutoff
        else:
            is_selected = v_jk < self.cutoff

        if not is_selected:
            return False

        # Check if this index is the one picked from the block
        r_jk = prf.offset(self.hint_id, block) % params.w
        return r_jk == offset


@dataclass
class BackupHint:
    """
    A backup hint for hint replenishment.

    Stores parities for both halves of the blocks, allowing
    replenishment regardless of which block the queried index
    belongs to.
    """

    hint_id: int                # J - unique identifier
    cutoff: int                 # v_hat_J - median cutoff value
    parity_low: bytes           # P_J - parity of blocks where v < cutoff
    parity_high: bytes          # P'_J - parity of blocks where v >= cutoff


@dataclass
class HintStorage:
    """
    Storage for all hints (regular + backup).
    """

    reg_hints: list[RegHint] = field(default_factory=list)
    backup_hints: list[BackupHint] = field(default_factory=list)
    next_backup_idx: int = 0    # index of next unused backup hint

    def add_reg_hint(self, hint: RegHint) -> None:
        """Add a regular hint."""
        self.reg_hints.append(hint)

    def add_backup_hint(self, hint: BackupHint) -> None:
        """Add a backup hint."""
        self.backup_hints.append(hint)

    def find_hint_for_index(
        self,
        index: int,
        prf,  # PRF instance
        params,  # Params instance
    ) -> Optional[tuple[int, RegHint]]:
        """
        Find a regular hint whose subset contains the given index.

        Args:
            index: Database index to find
            prf: PRF instance
            params: Parameters

        Returns:
            Tuple of (hint_position, hint) if found, None otherwise
        """
        for pos, hint in enumerate(self.reg_hints):
            if hint.contains_index(index, prf, params):
                return (pos, hint)
        return None

    def get_next_backup(self) -> Optional[BackupHint]:
        """Get the next unused backup hint."""
        if self.next_backup_idx >= len(self.backup_hints):
            return None
        hint = self.backup_hints[self.next_backup_idx]
        self.next_backup_idx += 1
        return hint

    def replace_hint(self, position: int, new_hint: RegHint) -> None:
        """Replace a regular hint at the given position."""
        self.reg_hints[position] = new_hint

    def remaining_queries(self) -> int:
        """Number of queries remaining before offline phase needed."""
        return len(self.backup_hints) - self.next_backup_idx
