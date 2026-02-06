"""
KPIR Server (Section 5, eprint 2019/1483).

Wraps any index-based PIR server with cuckoo hashing for keyword lookups.
"""

from collections.abc import Iterator
from types import ModuleType

from .cuckoo import CuckooTable
from .params import KPIRParams
from ..protocols import PIRServer, Query, Response, EntryUpdate


class KPIRServer:
    """KPIR Server. Wraps a PIR server with cuckoo hashing for keyword-based lookups."""

    def __init__(
        self,
        params: KPIRParams,
        cuckoo_table: CuckooTable,
        pir_server: PIRServer,
    ):
        """
        Initialize server with pre-built components.

        Args:
            params: KPIR parameters
            cuckoo_table: Cuckoo hash table (must match pir_server's database)
            pir_server: Underlying PIR server
        """
        self.params = params
        self._cuckoo_table = cuckoo_table
        self._pir_server = pir_server

    @classmethod
    def create(
        cls,
        kv_pairs: dict[bytes, bytes],
        params: KPIRParams,
        pir_module: ModuleType,
        **pir_kwargs,
    ) -> "KPIRServer":
        """
        Create a KPIRServer from key-value pairs.

        Args:
            kv_pairs: Dictionary mapping keys to values
            params: KPIR parameters
            pir_module: PIR module (e.g., rms24 or plinko) with Server and create_params
            **pir_kwargs: Scheme-specific options passed to pir_module.create_params()
                (e.g., security_param, block_size, num_backup_hints)

        Returns:
            Configured KPIRServer
        """
        pir_params = pir_module.create_params(
            num_entries=params.num_buckets,
            entry_size=params.entry_size,
            **pir_kwargs,
        )
        cuckoo_table = CuckooTable.build(kv_pairs, params.cuckoo_params)
        database = cuckoo_table.to_database()
        pir_server = pir_module.Server(database, pir_params)
        return cls(params, cuckoo_table, pir_server)

    def stream_database(self) -> Iterator[bytes]:
        """
        Stream the database for client hint generation.

        Yields:
            Each entry in order (index 0, 1, 2, ...)
        """
        return self._pir_server.stream_database()

    def answer(
        self, queries: list[Query]
    ) -> tuple[list[Response], list[tuple[bytes, bytes]]]:
        """
        Answer PIR queries.

        Args:
            queries: List of PIR queries

        Returns:
            Tuple of (PIR responses, current stash entries)
        """
        return self._pir_server.answer(queries), self._cuckoo_table.stash

    def update(self, changes: dict[bytes, bytes | None]) -> list[EntryUpdate]:
        """
        Update the database.

        Args:
            changes: Dictionary mapping keys to values.
                - key -> value: insert (if new) or update (if exists)
                - key -> None: delete

        Returns:
            List of EntryUpdate for client hint updates

        Raises:
            KeyError: If deleting a key that doesn't exist
        """
        index_updates: dict[int, bytes] = {}
        empty_entry = bytes(self.params.entry_size)

        for key, value in changes.items():
            if value is None:
                bucket_idx = self._cuckoo_table.delete(key)
                if bucket_idx is not None:
                    index_updates[bucket_idx] = empty_entry
            else:
                bucket_changes = self._cuckoo_table.upsert(key, value)
                for idx, entry in bucket_changes:
                    index_updates[idx] = entry

        return self._pir_server.update_entries(index_updates)
