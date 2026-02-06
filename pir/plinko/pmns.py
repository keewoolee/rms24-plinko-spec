"""
Pseudorandom Multinomial Sampler (PMNS) for Plinko PIR.

Based on Plinko.pdf Figure 4.

PMNS samples from the multinomial distribution MN(n, m) - throwing n balls into m bins.
Uses a binary tree with binomial sampling at each node.

The key operations:
- Forward S(x): which bin does ball x land in?
- Inverse S^{-1}(y): which balls are in bin y?
"""

import secrets
from dataclasses import dataclass

from .utils import PRG, derive_seed


@dataclass
class PMNSNode:
    """
    A node in the PMNS binary tree.

    Represents a range of balls [start, start+count) mapped to bins [low, high].
    """
    start: int   # First ball index in this subtree
    count: int   # Number of balls in this subtree
    low: int     # First bin index in this subtree
    high: int    # Last bin index in this subtree

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (single bin)."""
        return self.low == self.high


class PMNS:
    """
    Pseudorandom Multinomial Sampler.

    Implements the PMNS from Plinko Figure 4:
    - S(x): Forward mapping - which bin does ball x land in?
    - S^{-1}(y): Inverse mapping - which balls are in bin y?

    The construction uses a binary tree where each internal node
    splits balls between left and right children using binomial sampling.
    """

    def __init__(self, domain_size: int, range_size: int, security_param: int = 128):
        """
        Initialize PMNS.

        Args:
            domain_size: Number of balls (domain size)
            range_size: Number of bins (range size)
            security_param: Security parameter in bits (default: 128)
        """
        if domain_size < 0:
            raise ValueError("domain_size must be non-negative")
        if range_size <= 0:
            raise ValueError("range_size must be positive")

        self._domain_size = domain_size
        self._range_size = range_size
        # SEC: Construction fixes 128-bit security (32-byte key, SHAKE-256)
        # regardless of security_param.
        self._key = secrets.token_bytes(32)
        self._security_param = security_param

    @property
    def domain_size(self) -> int:
        """Number of balls (domain size)."""
        return self._domain_size

    @property
    def range_size(self) -> int:
        """Number of bins (range size)."""
        return self._range_size

    def _node_to_seed(self, node: PMNSNode) -> bytes:
        """
        Derive a deterministic seed for a node.

        The seed is derived from the key and node parameters,
        ensuring the same split is computed each time.
        """
        # Create a unique label for this node
        label = f"pmns:{node.start}:{node.count}:{node.low}:{node.high}".encode()
        return derive_seed(self._key, label)

    def _children(self, node: PMNSNode) -> tuple[PMNSNode, PMNSNode, int]:
        """
        Compute children of a node.

        Args:
            node: Parent node

        Returns:
            (left_child, right_child, s) where s is the number of balls
            going to the left child.
        """
        if node.is_leaf():
            raise ValueError("Cannot compute children of leaf node")

        mid = (node.low + node.high) // 2

        # Probability of going left = (number of left bins) / (total bins)
        num_left_bins = mid - node.low + 1
        num_total_bins = node.high - node.low + 1
        p = num_left_bins / num_total_bins

        # Sample how many balls go left
        seed = self._node_to_seed(node)
        prg = PRG(seed)
        s = prg.binomial(node.count, p)

        # Create child nodes
        left = PMNSNode(
            start=node.start,
            count=s,
            low=node.low,
            high=mid
        )
        right = PMNSNode(
            start=node.start + s,
            count=node.count - s,
            low=mid + 1,
            high=node.high
        )

        return left, right, s

    def forward(self, x: int) -> int:
        """
        Forward mapping: which bin does ball x land in?

        S(x) from Plinko Figure 4.

        Args:
            x: Ball index in [0, domain_size)

        Returns:
            Bin index in [0, range_size)
        """
        if x < 0 or x >= self._domain_size:
            raise ValueError(f"Ball index {x} out of range [0, {self._domain_size})")

        if self._domain_size == 0:
            raise ValueError("Cannot map ball when n=0")

        # Start at root
        node = PMNSNode(start=0, count=self._domain_size, low=0, high=self._range_size - 1)

        while not node.is_leaf():
            left, right, s = self._children(node)

            # Which child contains ball x?
            if x < node.start + s:
                node = left
            else:
                node = right

        return node.low

    def inverse(self, y: int) -> set[int]:
        """
        Inverse mapping: which balls are in bin y?

        S^{-1}(y) from Plinko Figure 4.

        Args:
            y: Bin index in [0, range_size)

        Returns:
            Set of ball indices that map to bin y
        """
        if y < 0 or y >= self._range_size:
            raise ValueError(f"Bin index {y} out of range [0, {self._range_size})")

        if self._domain_size == 0:
            raise ValueError("Cannot compute inverse when domain_size is 0")

        # Start at root
        node = PMNSNode(start=0, count=self._domain_size, low=0, high=self._range_size - 1)

        while not node.is_leaf():
            left, right, s = self._children(node)

            mid = (node.low + node.high) // 2

            # Which child contains bin y?
            if y <= mid:
                node = left
            else:
                node = right

        # Return all balls in this leaf node
        return set(range(node.start, node.start + node.count))
