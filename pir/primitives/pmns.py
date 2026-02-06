"""
Pseudorandom Multinomial Sampler (PMNS) protocol.

PMNS samples from the multinomial distribution MN(n, m) - throwing n unlabeled
balls into m bins. Balls are unlabeled, so at each split only the count matters;
by convention lower-indexed balls go left, making each bin's contents a contiguous
range in increasing order.

Supports both forward (which bin does ball x land in?) and inverse (which balls
are in bin y?) queries.
"""

from typing import Protocol


class PMNS(Protocol):
    """
    Pseudorandom Multinomial Sampler.

    Properties:
    - Pseudorandom: Distribution of balls looks like uniform multinomial
    - Efficiently invertible: Can enumerate all balls in a bin
    """

    def __init__(self, domain_size: int, range_size: int, security_param: int = 128) -> None:
        """
        Initialize PMNS.

        Args:
            domain_size: Number of balls
            range_size: Number of bins
            security_param: Security parameter in bits (default: 128)
        """
        ...

    @property
    def domain_size(self) -> int:
        """Number of balls (domain size)."""
        ...

    @property
    def range_size(self) -> int:
        """Number of bins (range size)."""
        ...

    def forward(self, x: int) -> int:
        """
        Forward mapping: which bin does ball x land in?

        Args:
            x: Ball index in [0, n)

        Returns:
            Bin index in [0, m)
        """
        ...

    def inverse(self, y: int) -> set[int]:
        """
        Inverse mapping: which balls are in bin y?

        Args:
            y: Bin index in [0, m)

        Returns:
            Set of ball indices that map to bin y
        """
        ...
