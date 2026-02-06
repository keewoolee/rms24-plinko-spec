"""
Protocols and messages for Piano-like PIR schemes (client-dependent preprocessing).

This module defines:
1. Message protocols: Query, Response, EntryUpdate
2. Protocol interfaces: PIRClient, PIRServer, PIRParams
3. Module interface: PIRModule (what a PIR module must export)

Piano-like PIR schemes (Piano, RMS24, Plinko, etc.) share these interfaces,
allowing different implementations to be used interchangeably.

Client-dependent preprocessing model:
- Offline phase: Server streams database to client; client generates hints
- Online phase: Client prepares query using a hint, server answers, client
  extracts result, then replenishes the consumed hint from backup hints
- Update: When database entries change, client must update affected hints
"""

from collections.abc import Iterator
from typing import Protocol, Any


# =============================================================================
# Message Protocols
# =============================================================================


class Query(Protocol):
    """
    Query from client to server.

    Concrete implementations define the query structure.
    """

    ...


class Response(Protocol):
    """
    Response from server to client.

    Concrete implementations define the response structure.
    """

    ...


class EntryUpdate(Protocol):
    """
    A single database entry update.

    Concrete implementations define how updates are represented.
    """

    ...


# =============================================================================
# Protocol Interfaces
# =============================================================================


class PIRClient(Protocol):
    """
    Protocol for PIR clients in the client-dependent preprocessing model.

    A PIR client must support:
    1. Offline phase: generate_hints() from database stream
    2. Online phase: query() -> extract() -> replenish_hints()
    3. Updates: update_hints() when database changes
    """

    def generate_hints(self, db_stream: Iterator[bytes]) -> None:
        """
        Offline phase: process database and generate hints.

        Args:
            db_stream: Iterator yielding each entry in order
        """
        ...

    def query(self, indices: list[int]) -> list[Query]:
        """
        Prepare queries for database indices.

        Args:
            indices: List of database indices to retrieve

        Returns:
            List of queries to send to server
        """
        ...

    def extract(self, responses: list[Response]) -> list[bytes]:
        """
        Extract results from server responses.

        Args:
            responses: List of responses from server

        Returns:
            List of database entries at the queried indices
        """
        ...

    def replenish_hints(self) -> None:
        """
        Replenish consumed hints after extraction.

        Must be called after extract() to complete the query batch.
        """
        ...

    def update_hints(self, updates: list[EntryUpdate]) -> None:
        """
        Update hints affected by database changes.

        Args:
            updates: List of EntryUpdate from server
        """
        ...

    def remaining_queries(self) -> int:
        """
        Return number of queries remaining before offline phase needed.
        """
        ...


class PIRServer(Protocol):
    """
    Protocol for PIR servers.

    A PIR server must support:
    1. Streaming database for client's offline phase
    2. Answering online queries
    3. Updating entries and returning update info for client hints
    """

    def stream_database(self) -> Iterator[bytes]:
        """
        Stream the database entry by entry.

        Used during the client's offline phase.

        Yields:
            Each entry in order (index 0, 1, 2, ...)
        """
        ...

    def answer(self, queries: list[Query]) -> list[Response]:
        """
        Answer queries.

        Args:
            queries: List of queries from client

        Returns:
            List of responses
        """
        ...

    def update_entries(self, updates: dict[int, bytes]) -> list[EntryUpdate]:
        """
        Update multiple database entries.

        Args:
            updates: Mapping from index to new value

        Returns:
            List of EntryUpdate for client hint updates
        """
        ...


class PIRParams(Protocol):
    """
    Protocol for PIR parameters.

    Parameters define the structure of the database.
    """

    @property
    def num_entries(self) -> int:
        """Total number of database entries."""
        ...

    @property
    def entry_size(self) -> int:
        """Size of each entry in bytes."""
        ...


# =============================================================================
# Module Interface
# =============================================================================


class PIRModule(Protocol):
    """
    Protocol for PIR modules (e.g., rms24, plinko).

    A PIR module must export:
    - Client: PIR client class that takes (params) as constructor argument
    - Server: PIR server class that takes (database, params) as constructor arguments
    - create_params: Factory function to create PIR parameters

    Example usage:
        from pir import rms24  # or plinko

        params = rms24.create_params(num_entries=1000, entry_size=32, security_param=128)
        client = rms24.Client(params)
        server = rms24.Server(database, params)
    """

    Client: type
    """PIR client class. Constructor: Client(params: PIRParams) -> PIRClient"""

    Server: type
    """PIR server class. Constructor: Server(database: list[bytes], params: PIRParams) -> PIRServer"""

    @staticmethod
    def create_params(num_entries: int, entry_size: int, **kwargs: Any) -> PIRParams:
        """
        Create PIR parameters.

        Args:
            num_entries: Number of database entries
            entry_size: Size of each entry in bytes
            **kwargs: Scheme-specific options (e.g., security_param, block_size)

        Returns:
            Configured PIR parameters
        """
        ...
