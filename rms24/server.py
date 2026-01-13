"""
Server implementation for RMS24 single-server PIR.

The server's role is simple:
1. Store the database
2. Answer online queries by computing XOR parities of requested subsets
3. Stream the database to clients during offline phase
"""

from .params import Params
from .protocol import Query, Response
from .utils import Database, xor_bytes, zero_entry


class Server:
    """
    PIR Server for the single-server RMS24 scheme.

    The server holds the database and answers queries without
    learning which entry the client is interested in.
    """

    def __init__(self, database: Database, params: Params):
        """
        Initialize server with database.

        Args:
            database: The database to serve
            params: PIR parameters
        """
        self.db = database
        self.params = params

    def answer(self, query: Query) -> Response:
        """
        Answer an online query by computing parities of two subsets.

        The client sends two subsets (real and dummy, permuted).
        Server computes XOR of entries in each subset.

        Args:
            query: Query containing two subsets of indices

        Returns:
            Response containing XOR parities of each subset
        """
        parity_0 = self._compute_parity(query.subset_0)
        parity_1 = self._compute_parity(query.subset_1)
        return Response(parity_0=parity_0, parity_1=parity_1)

    def _compute_parity(self, indices: list[int]) -> bytes:
        """
        Compute XOR of database entries at given indices.

        Args:
            indices: List of database indices

        Returns:
            XOR of all entries at the given indices
        """
        if not indices:
            return zero_entry(self.params.entry_size)

        result = self.db[indices[0]]
        for idx in indices[1:]:
            result = xor_bytes(result, self.db[idx])
        return result

    def stream_database(self):
        """
        Stream the database block by block.

        Used during the client's offline phase.

        Yields:
            Tuples of (block_id, entries_in_block)
        """
        yield from self.db.stream_blocks(self.params.w)
