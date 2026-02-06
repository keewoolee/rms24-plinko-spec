"""
Invertible PRF (iPRF) for Plinko PIR.

Based on Plinko.pdf Theorem 4.4.

The iPRF is a composition of:
- PRP: Small-domain Pseudorandom Permutation (Sometimes-Recurse Shuffle)
- PMNS: Pseudorandom Multinomial Sampler

Composition:
    F(x) = S(P(x))                        // Forward: PRP then PMNS
    F^{-1}(y) = {P^{-1}(z) : z ∈ S^{-1}(y)}  // Inverse: PMNS^{-1} then PRP^{-1}

Why this works:
1. PRP permutes inputs randomly (like shuffling cards)
2. PMNS assigns permuted inputs to bins (like throwing balls)
3. Result looks like random function: each input goes to random bin
4. Inversion: find all inputs in a bin via PMNS^{-1}, then un-permute via PRP^{-1}
"""


from .prp import PRP
from .pmns import PMNS


class InvertiblePRF:
    """Invertible PRF (iPRF) for Plinko PIR."""

    def __init__(
        self,
        domain_size: int,
        range_size: int,
        security_param: int = 128,
    ):
        """
        Initialize iPRF.

        Args:
            domain_size: Size of the input domain [0, n)
            range_size: Size of the output range [0, m)
            security_param: Security parameter in bits (default: 128)
        """
        if domain_size <= 0:
            raise ValueError("Domain size must be positive")
        if range_size <= 0:
            raise ValueError("Range size must be positive")

        self._domain_size = domain_size
        self._range_size = range_size

        # Initialize components (they generate their own keys)
        self._prp = PRP(
            domain_size=domain_size,
            security_param=security_param
        )
        self._pmns = PMNS(
            domain_size=domain_size,
            range_size=range_size,
            security_param=security_param,
        )

    @property
    def domain_size(self) -> int:
        """Size of the input domain [0, n)."""
        return self._domain_size

    @property
    def range_size(self) -> int:
        """Size of the output range [0, m)."""
        return self._range_size

    def forward(self, x: int) -> int:
        """
        Forward evaluation: F(x).

        Computes S(P(x)) - first apply PRP, then PMNS.

        Args:
            x: Input in [0, domain_size)

        Returns:
            Output in [0, range_size)
        """
        if x < 0 or x >= self._domain_size:
            raise ValueError(f"Input {x} out of range [0, {self._domain_size})")

        # Apply PRP first, then PMNS
        return self._pmns.forward(self._prp.forward(x))

    def inverse(self, y: int) -> set[int]:
        """
        Inverse evaluation: F^{-1}(y).

        Returns all x such that F(x) = y.

        Computes {P^{-1}(z) : z ∈ S^{-1}(y)} - first get all balls in bin y,
        then un-permute each one.

        Args:
            y: Output value in [0, range_size)

        Returns:
            Set of all inputs that map to y
        """
        if y < 0 or y >= self._range_size:
            raise ValueError(f"Output {y} out of range [0, {self._range_size})")

        # Get all balls in bin y, then un-permute each
        return {self._prp.inverse(z) for z in self._pmns.inverse(y)}
