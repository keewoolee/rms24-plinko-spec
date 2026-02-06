"""
Invertible Pseudorandom Function (iPRF) protocol.

An iPRF is a function F: [n] -> [m] that looks like a random function
but supports efficient inversion: given y, find all x where F(x) = y.
"""

from typing import Protocol


class InvertiblePRF(Protocol):
    """
    Invertible Pseudorandom Function (iPRF).

    Properties:
    - Pseudorandom: F looks like a random function (each input maps to random output)
    - Efficiently invertible: Can find all preimages of any output
    """

    def __init__(self, domain_size: int, range_size: int, security_param: int = 128) -> None:
        """
        Initialize iPRF.

        Args:
            domain_size: Size of the input domain [0, n)
            range_size: Size of the output range [0, m)
            security_param: Security parameter in bits (default: 128)
        """
        ...

    @property
    def domain_size(self) -> int:
        """Size of the input domain [0, n)."""
        ...

    @property
    def range_size(self) -> int:
        """Size of the output range [0, m)."""
        ...

    def forward(self, x: int) -> int:
        """
        Forward evaluation: F(x).

        Args:
            x: Input in [0, n)

        Returns:
            Output in [0, m)
        """
        ...

    def inverse(self, y: int) -> set[int]:
        """
        Inverse evaluation: F^{-1}(y).

        Returns all x such that F(x) = y.

        Args:
            y: Output value in [0, m)

        Returns:
            Set of all inputs that map to y
        """
        ...
