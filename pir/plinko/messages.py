"""
Message types for Plinko PIR scheme.
"""

from dataclasses import dataclass


@dataclass
class Query:
    """
    Query from client to server.

    The query specifies two subsets of blocks via a bitmask. Each subset
    contains exactly num_blocks/2 blocks. The server computes XOR parities
    for both subsets without knowing which contains the real query.

    Attributes:
        mask: Bitmask where bit k indicates block k's subset assignment.
              Bit k = 1 means block k is in subset_0.
        offsets: Shared offset array of length num_blocks/2. The i-th block
                 in each subset uses offsets[i] to select an entry.
    """

    mask: bytes  # num_blocks bits packed into bytes
    offsets: list[int]  # num_blocks/2 offsets shared by both subsets


@dataclass
class Response:
    """
    Response from server to client.

    Contains XOR parities of entries in each subset.
    """

    parity_0: bytes  # XOR of entries in subset_0
    parity_1: bytes  # XOR of entries in subset_1


@dataclass
class EntryUpdate:
    """
    Notification of a database entry update.

    Sent from server to client so the client can update affected hints.
    """

    index: int  # Database index that was updated
    delta: bytes  # XOR of old_value and new_value
