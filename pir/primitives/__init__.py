"""
Cryptographic primitive interfaces for PIR schemes.

This module defines protocol interfaces for:
- PRP: Pseudorandom Permutation
- PMNS: Pseudorandom Multinomial Sampler
- InvertiblePRF: Invertible Pseudorandom Function

Concrete implementations are in pir/plinko/.
"""

from .prp import PRP
from .pmns import PMNS
from .iprf import InvertiblePRF

__all__ = [
    "PRP",
    "PMNS",
    "InvertiblePRF",
]
