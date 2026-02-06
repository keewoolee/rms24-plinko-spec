"""
Client implementation for RMS24 single-server PIR.

The client's role:
1. Generate hints during offline phase (streaming the database)
2. Prepare queries that hide which entry is being accessed
3. Extract results from server responses
4. Replenish consumed hints using backup hints
5. Update hints when database entries change
"""

import secrets
from collections.abc import Iterator
from dataclasses import dataclass, field

from .params import Params
from .messages import Query, Response, EntryUpdate
from .utils import PRF, entries_to_blocks, find_cutoff, xor_bytes, zero_entry


@dataclass
class _QueryState:
    """Internal state from query() needed for extract() and replenish()."""
    queried_index: int
    hint_idx: int
    real_is_first: bool
    queried_entry: bytes | None = None


@dataclass
class HintState:
    """
    Hint storage using parallel arrays.

    hint_id == index (no separate hint_ids needed).
    Indices 0..num_reg_hints-1 are regular hints.
    Indices num_reg_hints..num_total_hints-1 are backup hints.
    """
    cutoffs: list[int] = field(default_factory=list)  # 0 = invalid/consumed
    extras: list[int] = field(default_factory=list)  # extra entry index for each hint
    parities: list[bytes] = field(default_factory=list)
    flips: list[bool] = field(default_factory=list)
    backup_parities_high: list[bytes] = field(default_factory=list)
    next_backup_idx: int = 0


class Client:
    """
    PIR Client for the single-server RMS24 scheme.

    The client maintains hints that allow sublinear online queries.
    After num_backup_hints queries, the offline phase must be re-run.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, params: Params):
        self.params = params
        self.prf = PRF()
        self.hints = HintState()
        self._query_states: list[_QueryState] = []

    def _init_hint_arrays(self) -> None:
        """Initialize/reset hint arrays to proper sizes."""
        p = self.params
        h = self.hints

        h.cutoffs = [0] * p.num_total_hints
        h.extras = [-1] * p.num_total_hints  # -1 = unset (0 is a valid index)
        h.flips = [False] * p.num_total_hints
        h.parities = [zero_entry(p.entry_size) for _ in range(p.num_total_hints)]
        h.backup_parities_high = [zero_entry(p.entry_size) for _ in range(p.num_backup_hints)]
        h.next_backup_idx = p.num_reg_hints

    # -------------------------------------------------------------------------
    # Online phase
    # -------------------------------------------------------------------------

    def _hint_contains(self, hint_idx: int, index: int) -> bool:
        """Check if hint's subset contains index."""
        h = self.hints
        p = self.params
        block = p.block_of(index)
        offset = p.offset_in_block(index)

        # Check extra first (O(1))
        if h.extras[hint_idx] == index:
            return True

        if self.prf.offset(hint_idx, block, p.block_size) != offset:
            return False

        # Check if block is selected
        select_value = self.prf.select(hint_idx, block, p.select_output_bits)
        cutoff = h.cutoffs[hint_idx]
        flip = h.flips[hint_idx]
        return (select_value >= cutoff) if flip else (select_value < cutoff)

    def _block_selected(self, hint_idx: int, block: int) -> bool:
        """Check if block is selected by hint (PRF < cutoff or PRF >= cutoff if flipped)."""
        h = self.hints
        p = self.params
        select_value = self.prf.select(hint_idx, block, p.select_output_bits)
        cutoff = h.cutoffs[hint_idx]
        flip = h.flips[hint_idx]
        return (select_value >= cutoff) if flip else (select_value < cutoff)

    # -------------------------------------------------------------------------
    # Offline phase
    # -------------------------------------------------------------------------

    def generate_hints(self, db_stream: Iterator[bytes]) -> None:
        """
        Generate hints by streaming the database.

        Can be called after extract() even if replenish_hints() hasn't been called.

        Args:
            db_stream: Iterator yielding each entry in order
        """
        # Allow regeneration after extract() but before replenish_hints()
        if self._query_states and self._query_states[0].queried_entry is None:
            raise RuntimeError("Must call extract() before regenerating hints")
        self._query_states = []

        # SEC: Refresh PRF key for independent hints on repeated calls
        self.prf = PRF()

        self._init_hint_arrays()

        p = self.params
        h = self.hints

        # Phase 1: Build skeleton (cutoffs and extras)
        half = p.num_blocks // 2
        for hint_idx in range(p.num_total_hints):
            select_values = self.prf.select_vector(hint_idx, p.num_blocks, p.select_output_bits)
            h.cutoffs[hint_idx] = find_cutoff(select_values, half)

            if hint_idx < p.num_reg_hints and h.cutoffs[hint_idx] != 0:
                while True:
                    block = secrets.randbelow(p.num_blocks)
                    if self.prf.select(hint_idx, block, p.select_output_bits) >= h.cutoffs[hint_idx]:
                        break
                offset = secrets.randbelow(p.block_size)
                h.extras[hint_idx] = p.index_from_block_offset(block, offset)

        # Phase 2: Stream database and accumulate parities
        # OPT: Hints are independent; could process concurrently.
        block_stream = entries_to_blocks(db_stream, p.block_size)
        for block, block_entries in enumerate(block_stream):
            for hint_idx in range(p.num_total_hints):
                cutoff = h.cutoffs[hint_idx]
                if cutoff == 0:
                    continue

                select_value = self.prf.select(hint_idx, block, p.select_output_bits)
                picked_offset = self.prf.offset(hint_idx, block, p.block_size)
                entry = block_entries[picked_offset]
                is_selected = select_value < cutoff

                if hint_idx < p.num_reg_hints:
                    if is_selected:
                        h.parities[hint_idx] = xor_bytes(h.parities[hint_idx], entry)
                    elif block == p.block_of(h.extras[hint_idx]):
                        extra_offset = p.offset_in_block(h.extras[hint_idx])
                        h.parities[hint_idx] = xor_bytes(
                            h.parities[hint_idx], block_entries[extra_offset]
                        )
                else:
                    backup_idx = hint_idx - p.num_reg_hints
                    if is_selected:
                        h.parities[hint_idx] = xor_bytes(h.parities[hint_idx], entry)
                    else:
                        h.backup_parities_high[backup_idx] = xor_bytes(
                            h.backup_parities_high[backup_idx], entry
                        )

    def _build_query(self, hint_idx: int, queried_index: int) -> tuple[Query, bool]:
        """Build a query for the given hint and queried index."""
        h = self.hints
        p = self.params
        num_blocks = p.num_blocks
        queried_block = p.block_of(queried_index)
        extra_block = p.block_of(h.extras[hint_idx])
        extra_offset = p.offset_in_block(h.extras[hint_idx])

        real_is_first = secrets.randbelow(2) == 0  # SEC: random subset assignment
        mask_int = 0
        offsets = []

        for block in range(num_blocks):
            if block == queried_block:
                is_real = False
            elif self._block_selected(hint_idx, block):
                is_real = True
            elif block == extra_block:
                is_real = True
            else:
                is_real = False

            if is_real:
                if block == extra_block:
                    offsets.append(extra_offset)
                else:
                    offsets.append(self.prf.offset(hint_idx, block, p.block_size))

            if is_real == real_is_first:
                mask_int |= 1 << block

        mask = mask_int.to_bytes((num_blocks + 7) // 8, "little")
        return Query(mask=mask, offsets=offsets), real_is_first

    def query(self, indices: list[int]) -> list[Query]:
        """Prepare queries for multiple indices."""
        h = self.hints
        if not h.cutoffs:
            raise RuntimeError("Must call generate_hints() before querying")

        if self._query_states:
            raise RuntimeError("Previous query batch not yet completed")

        num_queries = len(indices)
        queries: list[Query | None] = [None] * num_queries
        states: list[_QueryState | None] = [None] * num_queries
        remaining: set[int] = set(range(num_queries))

        for hint_idx in range(h.next_backup_idx):
            if not remaining:
                break
            if h.cutoffs[hint_idx] == 0:
                continue

            matched_pos = None
            for i in remaining:
                if self._hint_contains(hint_idx, indices[i]):
                    matched_pos = i
                    break

            if matched_pos is None:
                continue

            remaining.remove(matched_pos)
            queried_index = indices[matched_pos]

            query, real_is_first = self._build_query(hint_idx, queried_index)
            h.cutoffs[hint_idx] = 0  # Mark consumed after building query
            queries[matched_pos] = query
            states[matched_pos] = _QueryState(
                queried_index=queried_index,
                hint_idx=hint_idx,
                real_is_first=real_is_first,
            )

        if remaining:
            unmatched = [indices[i] for i in remaining]
            raise RuntimeError(f"No hints found for indices: {unmatched}")

        self._query_states = states
        return queries

    def extract(self, responses: list[Response]) -> list[bytes]:
        """Extract results from server responses."""
        if not self._query_states:
            raise RuntimeError("Must call query() before extract()")

        if len(responses) != len(self._query_states):
            raise RuntimeError(
                f"Response count ({len(responses)}) doesn't match query count ({len(self._query_states)})"
            )

        h = self.hints
        results = []
        for state, response in zip(self._query_states, responses):
            real_parity = response.parity_0 if state.real_is_first else response.parity_1
            queried_entry = xor_bytes(real_parity, h.parities[state.hint_idx])
            state.queried_entry = queried_entry
            results.append(queried_entry)

        return results

    def replenish_hints(self) -> None:
        """Replenish all consumed hints using backup hints."""
        if not self._query_states:
            raise RuntimeError("Must call query() before replenish_hints()")

        for state in self._query_states:
            if state.queried_entry is None:
                raise RuntimeError("Must call extract() before replenish_hints()")

        p = self.params
        h = self.hints

        for state in self._query_states:
            # Find next valid backup (used hint already marked consumed in query())
            while h.next_backup_idx < p.num_total_hints and h.cutoffs[h.next_backup_idx] == 0:
                h.next_backup_idx += 1

            if h.next_backup_idx >= p.num_total_hints:
                raise RuntimeError("Not enough backup hints")

            backup_idx = h.next_backup_idx
            h.next_backup_idx += 1

            # Check which half of backup contains the queried entry
            queried_block = p.block_of(state.queried_index)
            select_value = self.prf.select(backup_idx, queried_block, p.select_output_bits)
            cutoff = h.cutoffs[backup_idx]

            if select_value >= cutoff:
                parity = h.parities[backup_idx]
                flip = False
            else:
                parity = h.backup_parities_high[backup_idx - p.num_reg_hints]
                flip = True

            h.extras[backup_idx] = state.queried_index
            h.parities[backup_idx] = xor_bytes(parity, state.queried_entry)
            h.flips[backup_idx] = flip

        self._query_states = []

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def update_hints(self, updates: list[EntryUpdate]) -> None:
        """Update hints affected by database entry changes."""
        h = self.hints
        if not h.cutoffs:
            raise RuntimeError("Must call generate_hints() before update_hints()")

        if not updates:
            return

        p = self.params

        for u in updates:
            # Update regular hints (including promoted backups)
            for hint_idx in range(h.next_backup_idx):
                if h.cutoffs[hint_idx] == 0:
                    continue
                if self._hint_contains(hint_idx, u.index):
                    h.parities[hint_idx] = xor_bytes(h.parities[hint_idx], u.delta)

            # Update un-promoted backup hints
            block = p.block_of(u.index)
            offset = p.offset_in_block(u.index)
            for hint_idx in range(h.next_backup_idx, p.num_total_hints):
                if h.cutoffs[hint_idx] == 0:
                    continue
                picked_offset = self.prf.offset(hint_idx, block, p.block_size)
                if picked_offset != offset:
                    continue
                backup_idx = hint_idx - p.num_reg_hints
                select_value = self.prf.select(hint_idx, block, p.select_output_bits)
                if select_value < h.cutoffs[hint_idx]:
                    h.parities[hint_idx] = xor_bytes(h.parities[hint_idx], u.delta)
                else:
                    h.backup_parities_high[backup_idx] = xor_bytes(
                        h.backup_parities_high[backup_idx], u.delta
                    )

    def remaining_queries(self) -> int:
        """Return number of queries remaining before offline phase needed."""
        p = self.params
        h = self.hints
        count = 0
        for hint_idx in range(h.next_backup_idx, p.num_total_hints):
            if h.cutoffs[hint_idx] != 0:
                count += 1
        return count
