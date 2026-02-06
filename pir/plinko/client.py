"""
Client implementation for Plinko PIR.

The key innovation in Plinko is using Invertible PRFs (iPRFs) for:
- Õ(1) hint search: find hints containing (block, offset)
- Õ(1) hint update: update affected hints

Architecture:
- For each block α, we have an iPRF that maps hint indices to offsets:
  F_α: [num_total_hints] -> [block_size]
- Hint j covers offset F_α(j) in block α
- To find hints covering (α, β): compute F_α^{-1}(β)

Block selection (which blocks contribute to hint parity):
- PRF-based cutoff splits blocks into two subsets
- Regular hints select (num_blocks/2 + 1) blocks, backup hints select num_blocks/2
"""

import secrets
from dataclasses import dataclass, field
from collections.abc import Iterator

from .params import Params
from .iprf import InvertiblePRF
from .messages import Query, Response, EntryUpdate
from .utils import PRF, find_cutoff, xor_bytes, zero_entry


@dataclass
class _QueryState:
    """Internal state from query() needed for extract() and replenish().

    - requested_index: the index user asked for (always set)
    - queried_index: the index actually sent to server
      - Same as requested_index for normal queries
      - Deterministic different index for decoy queries (when requested is cached)
    - queried_entry: the entry for queried_index (set by extract)
    """
    requested_index: int
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

    Query cache (Q from paper):
    - Maps index -> (answer, promoted_hint_idx)
    - Enables Õ(1) repeated queries and Õ(1) update of promoted hints by extra
    """
    cutoffs: list[int] = field(default_factory=list)  # 0 = invalid/consumed
    extras: list[int] = field(default_factory=list)  # extra entry index for backup hints
    parities: list[bytes] = field(default_factory=list)
    flips: list[bool] = field(default_factory=list)
    backup_parities_high: list[bytes] = field(default_factory=list)
    next_backup_idx: int = 0
    query_cache: dict[int, tuple[bytes, int]] = field(default_factory=dict)


class Client:
    """
    PIR Client for the Plinko scheme.

    The client maintains hints that allow sublinear online queries.
    Key feature: Õ(1) hint search and update using iPRF inverse.
    After num_backup_hints queries, the offline phase must be re-run.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, params: Params):
        self.params = params
        self.hints = HintState()
        self._query_states: list[_QueryState] = []

        # Initialize PRF and iPRFs
        self._refresh_keys()

    def _init_hint_arrays(self) -> None:
        """Initialize/reset hint arrays to proper sizes."""
        p = self.params
        h = self.hints

        h.cutoffs = [0] * p.num_total_hints
        h.flips = [False] * p.num_total_hints
        h.parities = [zero_entry(p.entry_size) for _ in range(p.num_total_hints)]
        # Only backup hints use extras (set during replenish, -1 = unset)
        h.extras = [-1] * p.num_backup_hints
        h.backup_parities_high = [zero_entry(p.entry_size) for _ in range(p.num_backup_hints)]
        h.next_backup_idx = p.num_reg_hints
        h.query_cache = {}
        self._next_decoy_idx = 0

    def _refresh_keys(self) -> None:
        """Refresh PRF keys for independent hints."""
        self.prf = PRF()

        # Recreate iPRFs
        self._iprfs = []
        for block in range(self.params.num_blocks):
            iprf = InvertiblePRF(
                domain_size=self.params.num_total_hints,
                range_size=self.params.block_size,
                security_param=self.params.security_param,
            )
            self._iprfs.append(iprf)

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

        # SEC: Refresh keys for independent hints on repeated calls
        self._refresh_keys()

        self._init_hint_arrays()

        p = self.params
        h = self.hints

        # Phase 1: Compute cutoffs
        # Regular hints select (n/2 + 1) blocks, backup hints select n/2 blocks
        half = p.num_blocks // 2

        for hint_idx in range(p.num_total_hints):
            select_values = self.prf.select_vector(hint_idx, p.num_blocks, p.select_output_bits)
            if hint_idx < p.num_reg_hints:
                h.cutoffs[hint_idx] = find_cutoff(select_values, half + 1)
            else:
                h.cutoffs[hint_idx] = find_cutoff(select_values, half)

        # Phase 2: Stream database and accumulate parities
        # OPT: Hints are independent; could process concurrently.
        for idx, entry in enumerate(db_stream):
            block = p.block_of(idx)
            offset = p.offset_in_block(idx)

            for hint_idx in self._iprfs[block].inverse(offset):
                cutoff = h.cutoffs[hint_idx]
                if cutoff == 0:
                    continue

                select_value = self.prf.select(hint_idx, block, p.select_output_bits)
                is_selected = select_value < cutoff

                if is_selected:
                    h.parities[hint_idx] = xor_bytes(h.parities[hint_idx], entry)
                elif hint_idx >= p.num_reg_hints:
                    # Backup hints track both halves for later promotion
                    backup_idx = hint_idx - p.num_reg_hints
                    h.backup_parities_high[backup_idx] = xor_bytes(
                        h.backup_parities_high[backup_idx], entry
                    )

    # -------------------------------------------------------------------------
    # Online phase
    # -------------------------------------------------------------------------

    def _block_selected(self, hint_idx: int, block: int) -> bool:
        """Check if block is selected by hint (PRF < cutoff or PRF >= cutoff if flipped)."""
        h = self.hints
        p = self.params
        select_value = self.prf.select(hint_idx, block, p.select_output_bits)
        cutoff = h.cutoffs[hint_idx]
        flip = h.flips[hint_idx]
        return (select_value >= cutoff) if flip else (select_value < cutoff)

    def _get_hint(self, index: int) -> int | None:
        """Find a valid hint covering index. Returns hint_idx or None."""
        h, p = self.hints, self.params
        block = p.block_of(index)
        offset = p.offset_in_block(index)

        # Õ(1) lookup via iPRF inverse
        for hint_idx in self._iprfs[block].inverse(offset):
            if hint_idx >= h.next_backup_idx:
                continue
            if h.cutoffs[hint_idx] == 0:
                continue
            if self._block_selected(hint_idx, block):
                return hint_idx
            # For promoted backups, also check extra
            if hint_idx >= p.num_reg_hints:
                backup_idx = hint_idx - p.num_reg_hints
                if h.extras[backup_idx] == index:
                    return hint_idx

        return None

    def _build_query(self, hint_idx: int, queried_index: int) -> tuple[Query, bool]:
        """Build a query for the given hint and queried index."""
        h = self.hints
        p = self.params
        num_blocks = p.num_blocks
        queried_block = p.block_of(queried_index)

        # Get extra index (-1 if not a promoted backup)
        extra_index = -1
        if hint_idx >= p.num_reg_hints:
            extra_index = h.extras[hint_idx - p.num_reg_hints]

        if extra_index != -1:
            extra_block = p.block_of(extra_index)
            extra_offset = p.offset_in_block(extra_index)

        real_is_first = secrets.randbelow(2) == 0  # SEC: random subset assignment
        mask_int = 0
        offsets = []

        for block in range(num_blocks):
            if block == queried_block:
                # Queried block is never real (client knows the answer)
                is_real = False
            elif self._block_selected(hint_idx, block):
                # Selected blocks are real
                is_real = True
            elif extra_index != -1 and block == extra_block:
                # Extra block is also real (only for promoted backups)
                is_real = True
            else:
                is_real = False

            if is_real:
                # Add offset for this real block
                if extra_index != -1 and block == extra_block:
                    offsets.append(extra_offset)
                else:
                    offsets.append(self._iprfs[block].forward(hint_idx))

            if is_real == real_is_first:
                mask_int |= 1 << block

        mask = mask_int.to_bytes((num_blocks + 7) // 8, "little")
        return Query(mask=mask, offsets=offsets), real_is_first

    def query(self, indices: list[int]) -> list[Query]:
        """
        Prepare queries for multiple indices.

        Uses iPRF inverse for Õ(1) hint search per target.
        For cached indices, queries a deterministic decoy index to maintain privacy.
        """
        if not self.hints.cutoffs:
            raise RuntimeError("Must call generate_hints() before querying")

        if self._query_states:
            raise RuntimeError("Previous query batch not yet completed")

        h = self.hints
        p = self.params
        queries: list[Query] = []
        states: list[_QueryState] = []
        batch_queried: set[int] = set()

        for requested_idx in indices:
            # SEC: If requested index is cached, query a decoy to hide repeated access
            if requested_idx in h.query_cache:
                # Find smallest decoy not in cache and not queried in this batch
                while (self._next_decoy_idx < p.num_entries and
                       (self._next_decoy_idx in h.query_cache or
                        self._next_decoy_idx in batch_queried)):
                    self._next_decoy_idx += 1
                if self._next_decoy_idx >= p.num_entries:
                    raise RuntimeError("No decoy indices available - all entries cached")
                query_idx = self._next_decoy_idx
                self._next_decoy_idx += 1
            else:
                query_idx = requested_idx

            batch_queried.add(query_idx)

            hint_idx = self._get_hint(query_idx)
            if hint_idx is None:
                raise RuntimeError(f"No hint found for index: {query_idx}")

            query, real_is_first = self._build_query(hint_idx, query_idx)
            h.cutoffs[hint_idx] = 0  # Mark consumed after building query
            queries.append(query)
            states.append(_QueryState(
                requested_index=requested_idx,
                queried_index=query_idx,
                hint_idx=hint_idx,
                real_is_first=real_is_first,
            ))

        self._query_states = states
        return queries

    def extract(self, responses: list[Response]) -> list[bytes]:
        """Extract results from server responses.

        For decoy queries (when requested_index is cached), returns the cached answer.
        """
        if not self._query_states:
            raise RuntimeError("Must call query() before extract()")

        if len(responses) != len(self._query_states):
            raise RuntimeError(
                f"Response count ({len(responses)}) doesn't match query count ({len(self._query_states)})"
            )

        h = self.hints
        results = []
        for state, response in zip(self._query_states, responses):
            # Extract queried entry from server response
            real_parity = response.parity_0 if state.real_is_first else response.parity_1
            queried_entry = xor_bytes(real_parity, h.parities[state.hint_idx])
            state.queried_entry = queried_entry

            # For decoy queries, return cached answer; otherwise return queried entry
            if state.requested_index != state.queried_index:
                # Decoy query - return cached answer for requested index
                cached_answer, _ = h.query_cache[state.requested_index]
                results.append(cached_answer)
            else:
                results.append(queried_entry)

        return results

    def replenish_hints(self) -> None:
        """Replenish all consumed hints using backup hints.

        Also populates query_cache for Õ(1) repeated queries and Õ(1) update by extra.
        """
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

            hint_idx = h.next_backup_idx
            h.next_backup_idx += 1
            backup_idx = hint_idx - p.num_reg_hints

            # Check which half of backup contains the queried entry
            queried_block = p.block_of(state.queried_index)
            select_value = self.prf.select(hint_idx, queried_block, p.select_output_bits)
            cutoff = h.cutoffs[hint_idx]

            if select_value >= cutoff:
                # Entry is in high half, use low parity
                parity = h.parities[hint_idx]
                flip = False
            else:
                # Entry is in low half, use high parity and flip
                parity = h.backup_parities_high[backup_idx]
                flip = True

            # Set extra to the queried index
            h.extras[backup_idx] = state.queried_index

            # Update parity to include the queried entry
            h.parities[hint_idx] = xor_bytes(parity, state.queried_entry)
            h.flips[hint_idx] = flip

            # Populate query cache: maps queried_index -> (answer, promoted_hint_idx)
            h.query_cache[state.queried_index] = (state.queried_entry, hint_idx)

        self._query_states = []

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def update_hints(self, updates: list[EntryUpdate]) -> None:
        """
        Update hints affected by database entry changes.

        Uses iPRF inverse for Õ(1) hint update (for most hints).
        Uses query_cache for Õ(1) lookup of promoted hints by extra entry.
        """
        h = self.hints
        if not h.cutoffs:
            raise RuntimeError("Must call generate_hints() before update_hints()")

        if not updates:
            return

        p = self.params

        for u in updates:
            block = p.block_of(u.index)
            offset = p.offset_in_block(u.index)

            # Update hints found via iPRF inverse (Õ(1) expected)
            for hint_idx in self._iprfs[block].inverse(offset):
                if h.cutoffs[hint_idx] == 0:
                    continue

                if self._block_selected(hint_idx, block):
                    h.parities[hint_idx] = xor_bytes(h.parities[hint_idx], u.delta)
                elif hint_idx >= h.next_backup_idx:
                    # Un-promoted backup: block is in high half
                    backup_idx = hint_idx - p.num_reg_hints
                    h.backup_parities_high[backup_idx] = xor_bytes(
                        h.backup_parities_high[backup_idx], u.delta
                    )

            # Update extra entries via query_cache (Õ(1) lookup)
            if u.index in h.query_cache:
                cached_answer, hint_idx = h.query_cache[u.index]
                # Always update cached answer (used for decoy queries)
                h.query_cache[u.index] = (xor_bytes(cached_answer, u.delta), hint_idx)
                # Only update parity if hint still valid
                if h.cutoffs[hint_idx] != 0:
                    h.parities[hint_idx] = xor_bytes(h.parities[hint_idx], u.delta)

    def remaining_queries(self) -> int:
        """Return number of queries remaining before offline phase needed."""
        p = self.params
        h = self.hints
        count = 0
        for hint_idx in range(h.next_backup_idx, p.num_total_hints):
            if h.cutoffs[hint_idx] != 0:
                count += 1
        return count
