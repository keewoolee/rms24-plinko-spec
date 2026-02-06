"""
Server implementation for Plinko PIR.

The server's role is simple:
1. Store the database
2. Stream the database to clients during offline phase
3. Answer online queries by computing XOR parities of requested subsets

The server sees standard PIR queries and responds accordingly. It is unaware
of the client's iPRF-based hint organization.
"""

from collections.abc import Iterator

from .params import Params
from .messages import Query, Response, EntryUpdate
from .utils import xor_bytes, zero_entry


class Server:
    """
    PIR Server for the Plinko scheme.

    The server holds the database and answers queries without learning
    which entry the client is interested in. Each query specifies two
    subsets of blocks; the server computes XOR parities for both subsets.
    """

    def __init__(self, database: list[bytes], params: Params):
        """
        Initialize server with database.

        Args:
            database: List of database entries
            params: PIR parameters
        """
        # Pad database to full size (num_blocks * block_size)
        full_size = params.num_blocks * params.block_size
        self._database = database + [bytes(params.entry_size)] * (full_size - len(database))
        self.params = params

    def stream_database(self) -> Iterator[bytes]:
        """
        Stream the database entry by entry.

        Used during the client's offline phase for hint generation.

        Yields:
            Each entry in order (index 0, 1, 2, ...)
        """
        for entry in self._database:
            yield entry

    def answer(self, queries: list[Query]) -> list[Response]:
        """
        Answer multiple queries by computing parities of subsets.

        For each query, the server computes XOR parities for two subsets
        of database entries. Each subset contains num_blocks/2 entries
        (one per block in that subset).

        Args:
            queries: List of queries containing mask and offsets

        Returns:
            List of responses containing XOR parities of each subset
        """
        num_blocks = self.params.num_blocks
        block_size = self.params.block_size

        # OPT: Each query iterates over all blocks independently. Could batch across queries.
        responses = []
        for query in queries:
            mask_int = int.from_bytes(query.mask, "little")

            parity_0 = zero_entry(self.params.entry_size)
            parity_1 = zero_entry(self.params.entry_size)
            idx_0 = idx_1 = 0

            for block in range(num_blocks):
                if mask_int & 1:
                    # Block is in subset_0
                    parity_0 = xor_bytes(
                        parity_0,
                        self._database[block * block_size + query.offsets[idx_0]]
                    )
                    idx_0 += 1
                else:
                    # Block is in subset_1
                    parity_1 = xor_bytes(
                        parity_1,
                        self._database[block * block_size + query.offsets[idx_1]]
                    )
                    idx_1 += 1
                mask_int >>= 1

            responses.append(Response(parity_0=parity_0, parity_1=parity_1))

        return responses

    def update_entries(self, updates: dict[int, bytes]) -> list[EntryUpdate]:
        """
        Update multiple database entries.

        Args:
            updates: Mapping from database index to new value

        Returns:
            List of EntryUpdate messages for client hint updates
        """
        result = []
        for index, new_value in updates.items():
            old_value = self._database[index]
            delta = xor_bytes(old_value, new_value)
            self._database[index] = new_value
            result.append(EntryUpdate(index=index, delta=delta))
        return result
