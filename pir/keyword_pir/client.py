"""
KPIR Client (Section 5, eprint 2019/1483).

Wraps any index-based PIR client with cuckoo hashing for keyword lookups.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType

from .cuckoo import CuckooHash
from .params import KPIRParams
from ..protocols import PIRClient, Query, Response, EntryUpdate


@dataclass
class _KPIRQueryState:
    """Internal state from query() needed for extract()."""

    keywords: list[bytes]


class KPIRClient:
    """KPIR Client. Wraps a PIR client with keyword-to-index translation via cuckoo hashing."""

    def __init__(
        self,
        params: KPIRParams,
        pir_client: PIRClient,
    ):
        """
        Initialize client.

        Args:
            params: KPIR parameters
            pir_client: Underlying PIR client
        """
        self.params = params
        cuckoo = params.cuckoo_params
        self._hasher = CuckooHash(cuckoo.num_hashes, cuckoo.num_buckets, cuckoo.seed)
        self._pir_client = pir_client
        self._query_state: _KPIRQueryState | None = None

    @classmethod
    def create(
        cls,
        params: KPIRParams,
        pir_module: ModuleType,
        **pir_kwargs,
    ) -> "KPIRClient":
        """
        Create a KPIRClient from parameters.

        Args:
            params: KPIR parameters
            pir_module: PIR module (e.g., rms24 or plinko) with Client and create_params
            **pir_kwargs: Scheme-specific options passed to pir_module.create_params()
                (e.g., security_param, block_size, num_backup_hints)

        Returns:
            Configured KPIRClient
        """
        pir_params = pir_module.create_params(
            num_entries=params.num_buckets,
            entry_size=params.entry_size,
            **pir_kwargs,
        )
        pir_client = pir_module.Client(pir_params)
        return cls(params, pir_client)

    def generate_hints(self, db_stream: Iterator[bytes]) -> None:
        """
        Generate hints from database stream.

        Args:
            db_stream: Iterator from server.stream_database()
        """
        self._pir_client.generate_hints(db_stream)

    def query(self, keywords: list[bytes]) -> list[Query]:
        """
        Prepare queries for keywords.

        Each keyword requires num_hashes PIR queries (one per hash function).

        Args:
            keywords: List of keywords to look up

        Returns:
            List of num_hashes * len(keywords) PIR queries
        """
        if self._query_state is not None:
            raise RuntimeError("Previous query batch not yet completed")

        # Compute all hash positions
        indices = []
        for keyword in keywords:
            positions = self._hasher.all_positions(keyword)
            indices.extend(positions)

        self._query_state = _KPIRQueryState(keywords=keywords)

        return self._pir_client.query(indices)

    def _find_value(
        self,
        keyword: bytes,
        candidates: list[bytes],
        stash_dict: dict[bytes, bytes],
    ) -> bytes | None:
        """Find value for keyword in candidates or stash."""
        key_size = self.params.key_size
        for candidate in candidates:
            if candidate[:key_size] == keyword:
                return candidate[key_size:]
        return stash_dict.get(keyword)

    def extract(
        self,
        responses: list[Response],
        stash: list[tuple[bytes, bytes]],
    ) -> list[bytes | None]:
        """
        Extract values from responses.

        Args:
            responses: List of PIR responses
            stash: Stash entries from server

        Returns:
            List of values (None if keyword not found)
        """
        if self._query_state is None:
            raise RuntimeError("Must call query() before extract()")

        entries = self._pir_client.extract(responses)
        num_hashes = self.params.num_hashes
        stash_dict = dict(stash)

        results = []
        for i, keyword in enumerate(self._query_state.keywords):
            candidates = entries[i * num_hashes : (i + 1) * num_hashes]
            results.append(self._find_value(keyword, candidates, stash_dict))

        return results

    def replenish_hints(self) -> None:
        """
        Replenish consumed hints.

        Must be called after extract() to complete the query batch.
        """
        self._pir_client.replenish_hints()
        self._query_state = None

    def update_hints(self, updates: list[EntryUpdate]) -> None:
        """
        Update hints for database changes.

        Args:
            updates: List of EntryUpdate from server
        """
        self._pir_client.update_hints(updates)

    def remaining_queries(self) -> int:
        """
        Return number of keyword queries remaining.

        Note: Each keyword query uses num_hashes PIR queries internally,
        so this is remaining_pir_queries // num_hashes.
        """
        return (
            self._pir_client.remaining_queries()
            // self.params.num_hashes
        )
