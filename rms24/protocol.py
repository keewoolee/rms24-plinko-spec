"""
Protocol messages for RMS24 PIR.

These are the messages exchanged between client and server.
"""

from dataclasses import dataclass


@dataclass
class Query:
    """Query from client to server."""
    subset_0: list[int]
    subset_1: list[int]


@dataclass
class Response:
    """Response from server to client."""
    parity_0: bytes
    parity_1: bytes
