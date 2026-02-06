"""
Pseudorandom Permutation (PRP) protocol.

A PRP is a keyed bijection P: [N] -> [N] that looks random to an observer
without the key. Must support both forward and inverse evaluation.
"""

from typing import Protocol


class PRP(Protocol):
    """
    Pseudorandom Permutation.

    Properties:
    - Bijective: P is a permutation (one-to-one and onto)
    - Pseudorandom: Without key, P(x) is indistinguishable from random permutation
    - Efficiently invertible: Can compute P^{-1}(y) given key
    """

    def __init__(self, domain_size: int, security_param: int = 128) -> None:
        """
        Initialize PRP.

        Args:
            domain_size: Size of the domain [0, N)
            security_param: Security parameter in bits (default: 128)
        """
        ...

    @property
    def domain_size(self) -> int:
        """Size of the domain [0, N)."""
        ...

    def forward(self, x: int) -> int:
        """
        Forward permutation: P(x).

        Args:
            x: Input in [0, N)

        Returns:
            Output in [0, N)
        """
        ...

    def inverse(self, y: int) -> int:
        """
        Inverse permutation: P^{-1}(y).

        Args:
            y: Input in [0, N)

        Returns:
            Output in [0, N) such that P(output) = y
        """
        ...
