"""
Plinko PIR scheme implementation, based on the Plinko scheme applied to RMS24.

Plinko uses Invertible PRFs (iPRFs) to achieve:
- Õ(1) hint searching and updating (vs linear scan)
- Optimal online time for any client storage

The key components:
- PRP: Small-domain Pseudorandom Permutation (Sometimes-Recurse Shuffle)
- PMNS: Pseudorandom Multinomial Sampler
- iPRF: Invertible PRF (composition of PRP and PMNS)
- Client: PIR client
- Server: PIR server
"""

from .params import Params
from .messages import Query, Response, EntryUpdate
from .client import Client
from .server import Server


def create_params(num_entries: int, entry_size: int, **kwargs) -> Params:
    """
    Create PIR parameters.

    Args:
        num_entries: Number of database entries
        entry_size: Size of each entry in bytes
        **kwargs: Scheme-specific options (security_param, block_size, num_backup_hints)

    Returns:
        Configured Params
    """
    return Params(num_entries=num_entries, entry_size=entry_size, **kwargs)


__all__ = [
    "Params",
    "Client",
    "Server",
    "Query",
    "Response",
    "EntryUpdate",
    "create_params",
]
