"""
PIR (Private Information Retrieval) library.

This package provides implementations of Piano-like PIR schemes
(client-dependent preprocessing model).

Modules:
- primitives: Cryptographic primitive protocols (PRP, PMNS, iPRF)
- protocols: Protocol interfaces for Piano-like PIR schemes
- keyword_pir: Keyword PIR using cuckoo hashing
- rms24: RMS24 PIR scheme implementation
- plinko: Plinko PIR scheme implementation
"""

from . import primitives
from . import protocols
from . import keyword_pir
from . import rms24
from . import plinko

__all__ = [
    "primitives",
    "protocols",
    "keyword_pir",
    "rms24",
    "plinko",
]
