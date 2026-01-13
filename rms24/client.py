import secrets
from dataclasses import dataclass
from typing import Iterator

from .params import Params
from .protocol import Query, Response
from .prf import PRF, find_median_cutoff, block_selected
from .hint import RegHint, BackupHint, HintStorage
from .utils import xor_bytes, zero_entry, random_offset


@dataclass
class _QueryState:
    """Internal state from query() needed for extract() and replenish()."""
    real_is_first: bool
    hint: RegHint
    hint_pos: int
    index: int
    entry: bytes = None  # Set by extract()


class Client:
    """
    PIR Client for the single-server RMS24 scheme.

    The client maintains hints that allow sublinear online queries.
    After num_backup_hints queries, the offline phase must be re-run.
    """

    def __init__(self, params: Params):
        """
        Initialize client.

        Args:
            params: PIR parameters
        """
        self.params = params
        self.prf = None
        self.hints = HintStorage()
        self._query_state = None

    def offline(self, db_stream: Iterator[tuple[int, list[bytes]]]) -> None:
        """
        Processes the database block by block and constructs:
        - num_reg_hints regular hints
        - num_backup_hints backup hints

        Args:
            db_stream: Iterator yielding (block_id, entries) tuples
        """
        self.prf = PRF()
        self.hints = HintStorage()
        self._query_state = None

        num_reg_hints = self.params.num_reg_hints
        c = self.params.c  # number of blocks
        w = self.params.w  # entries per block

        # Total hints: num_reg_hints regular + num_backup_hints backup
        num_total_hints = num_reg_hints + self.params.num_backup_hints

        # Step 1: Initialize hints (lines 1-8 of Algorithm 4)
        # For each hint j, compute V_j and find median cutoff
        cutoffs = []
        extra_indices = []  # Only for regular hints (j < num_reg_hints)

        for j in range(num_total_hints):
            # Compute selection vector V_j
            V_j = self.prf.select_vector(j, c)

            # Find median cutoff
            cutoff = find_median_cutoff(V_j)
            cutoffs.append(cutoff)

            # For regular hints, pick an extra index from an unselected block
            if j < num_reg_hints:
                # Find which blocks are NOT selected (v >= cutoff)
                unselected = [k for k in range(c) if V_j[k] >= cutoff]

                # Pick random block and random offset within it
                extra_block = secrets.choice(unselected)
                extra_offset = random_offset(w)
                extra_idx = self.params.index_from_block_offset(
                    extra_block, extra_offset
                )
                extra_indices.append(extra_idx)

        # Initialize parities to zero
        entry_size = self.params.entry_size
        parities = [zero_entry(entry_size) for _ in range(num_total_hints)]
        backup_parities_high = [
            zero_entry(entry_size) for _ in range(self.params.num_backup_hints)
        ]

        # Step 2: Stream database and accumulate parities (lines 9-22)
        for k, block_entries in db_stream:
            for j in range(num_total_hints):
                # Get the PRF values for this hint-block combination
                v_jk = self.prf.select(j, k)
                cutoff = cutoffs[j]

                # Compute r_{j,k} - offset within block
                r_jk = self.prf.offset(j, k) % w

                # Get the entry at this offset
                x = block_entries[r_jk]

                # Check if block k is selected by hint j (v < cutoff)
                if v_jk < cutoff:
                    # Block selected - add to parity
                    parities[j] = xor_bytes(parities[j], x)
                elif j >= num_reg_hints:
                    # Backup hint: also compute parity for unselected half
                    backup_idx = j - num_reg_hints
                    backup_parities_high[backup_idx] = xor_bytes(
                        backup_parities_high[backup_idx], x
                    )

            # For regular hints, check if extra index is in this block
            for j in range(num_reg_hints):
                extra_idx = extra_indices[j]
                extra_block = self.params.block_of(extra_idx)
                if extra_block == k:
                    extra_offset = self.params.offset_in_block(extra_idx)
                    parities[j] = xor_bytes(parities[j], block_entries[extra_offset])

        # Step 3: Create hint objects
        # Regular hints
        for j in range(num_reg_hints):
            hint = RegHint(
                hint_id=j,
                cutoff=cutoffs[j],
                extra_index=extra_indices[j],
                parity=parities[j],
                flip=False,
            )
            self.hints.add_reg_hint(hint)

        # Backup hints
        for j in range(self.params.num_backup_hints):
            backup = BackupHint(
                hint_id=num_reg_hints + j,
                cutoff=cutoffs[num_reg_hints + j],
                parity_low=parities[num_reg_hints + j],
                parity_high=backup_parities_high[j],
            )
            self.hints.add_backup_hint(backup)

    def query(self, index: int) -> Query:
        """
        Algorithm 2 (part 1): Prepare online query.

        Constructs two subsets to send to the server.

        Args:
            index: Database index to retrieve

        Returns:
            Query to send to server
        """
        if not self.hints.reg_hints:
            raise RuntimeError("Must run offline phase before querying")

        if self._query_state is not None:
            raise RuntimeError("Previous query not yet completed")

        # Line 2: Compute block of index i
        ell = self.params.block_of(index)
        c = self.params.c  # number of blocks
        w = self.params.w  # entries per block

        # Line 3: Find a hint whose subset contains index i
        result = self.hints.find_hint_for_index(index, self.prf, self.params)
        if result is None:
            raise RuntimeError(f"No hint found containing index {index}")

        hint_pos, hint = result
        j = hint.hint_id

        # Lines 4-13: Construct real subset S and dummy subset S'
        S = []      # Real subset (will contain hint's indices minus query index)
        S_prime = []  # Dummy subset

        for k in range(c):
            v_jk = self.prf.select(j, k)

            # Check if block k is selected
            if block_selected(v_jk, hint.cutoff, hint.flip):
                # Block selected - add the index from this block
                r_jk = self.prf.offset(j, k) % w
                idx = self.params.index_from_block_offset(k, r_jk)
                S.append(idx)
            elif self.params.block_of(hint.extra_index) == k:
                # This is the extra index's block (unselected but in hint)
                S.append(hint.extra_index)
            else:
                # Unselected block - add random index to dummy subset
                dummy_offset = random_offset(w)
                dummy_idx = self.params.index_from_block_offset(k, dummy_offset)
                S_prime.append(dummy_idx)

        # Line 14: Remove the queried index from the real subset
        if index in S:
            S.remove(index)
        else:
            raise RuntimeError(f"Index {index} not found in hint subset")

        # Line 15: Add a random index from block ell to dummy subset
        dummy_offset = random_offset(w)
        dummy_idx = self.params.index_from_block_offset(ell, dummy_offset)
        S_prime.append(dummy_idx)

        # Line 16: Permute the two subsets randomly
        if secrets.randbelow(2) == 0:
            subset_0, subset_1 = S, S_prime
            real_is_first = True
        else:
            subset_0, subset_1 = S_prime, S
            real_is_first = False

        self._query_state = _QueryState(
            real_is_first=real_is_first,
            hint=hint,
            hint_pos=hint_pos,
            index=index,
        )

        return Query(subset_0=subset_0, subset_1=subset_1)

    def extract(self, response: Response) -> bytes:
        """
        Algorithm 2 (part 2): Extract result from server response.

        Args:
            response: Response from server

        Returns:
            Database entry at the queried index
        """
        if self._query_state is None:
            raise RuntimeError("Must call query() before extract()")

        # Get the parity of the real subset
        P = response.parity_0 if self._query_state.real_is_first else response.parity_1

        # Line 18: Recover DB[i] = P XOR P_j
        entry = xor_bytes(P, self._query_state.hint.parity)

        # Store entry in state for replenish
        self._query_state.entry = entry

        return entry

    def replenish(self) -> None:
        """
        Algorithm 5: Replenish the consumed hint.

        Must be called after extract() to complete the query.
        """
        if self._query_state is None:
            raise RuntimeError("Must call query() before replenish()")

        if self._query_state.entry is None:
            raise RuntimeError("Must call extract() before replenish()")

        # Get next unused backup hint
        backup = self.hints.get_next_backup()
        if backup is None:
            raise RuntimeError("No backup hints remaining")

        queried_index = self._query_state.index
        queried_entry = self._query_state.entry
        ell = self.params.block_of(queried_index)

        # Check which half does NOT select block ell
        v_J_ell = self.prf.select(backup.hint_id, ell)

        if v_J_ell > backup.cutoff:
            # Block ell is NOT in the "low" half, use low half
            parity = backup.parity_low
            flip = False
        else:
            # Block ell IS in the "low" half, use high half
            parity = backup.parity_high
            flip = True

        # Create new regular hint with queried_index as extra index
        new_hint = RegHint(
            hint_id=backup.hint_id,
            cutoff=backup.cutoff,
            extra_index=queried_index,
            parity=xor_bytes(parity, queried_entry),
            flip=flip,
        )

        # Replace consumed hint
        self.hints.replace_hint(self._query_state.hint_pos, new_hint)
        self._query_state = None

    def remaining_queries(self) -> int:
        """Return number of queries remaining before offline phase needed."""
        return self.hints.remaining_queries()
