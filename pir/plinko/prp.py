"""
Small-Domain Pseudorandom Permutation (PRP) for Plinko PIR.

Based on the Sometimes-Recurse (SR) Shuffle from MR14.pdf Figure 1.

The SR shuffle builds a PRP on domain [N] using:
1. Swap-or-Not (SN) inner shuffle for mixing
2. Recursive structure: shuffle, cut, recurse on one half

This gives a full-security PRP on arbitrary domains [N].
"""

import hashlib
import math
import secrets


class PRP:
    """
    Small-domain Pseudorandom Permutation using Sometimes-Recurse Shuffle.

    Based on MR14 Figure 1 (SR Shuffle):

    procedure E^N_KF(X):           // X ∈ [N]
        if N = 1: return X

        // Swap-or-Not shuffle for t_N rounds
        for i = 1 to t_N:
            X' = K_i - X (mod N)       // partner of X
            X̂ = max(X, X')             // canonical name
            if F(i, X̂) = 1: X = X'     // maybe swap

        // Recurse on first pile only
        if X < p_N: return E^{p_N}_KF(X)
        else: return X

    Parameters:
    - p_N = ⌊N/2⌋ (split point)
    - t_N = number of SN rounds for domain size N
    - K_i = round constants (derived from PRF)
    - F(i, x̂) = round function returning 0 or 1 (PRF)
    """

    def __init__(self, domain_size: int, security_param: int = 128):
        """
        Initialize PRP.

        Args:
            domain_size: Size of the domain [0, N)
            security_param: Security parameter for round count computation (default: 128)
        """
        if domain_size <= 0:
            raise ValueError("Domain size must be positive")

        self._domain_size = domain_size
        self._key = secrets.token_bytes(32)
        self._security_param = security_param

    @property
    def domain_size(self) -> int:
        """Size of the domain [0, N)."""
        return self._domain_size

    def _num_sn_rounds(self, n: int) -> int:
        """
        Get number of Swap-or-Not rounds for domain size n.

        From MR14 Section 5 (equation 3), using strategy 1 (split error equally):
        - Total error budget ε = 2^{-security_param}
        - Number of stages ≈ lg(N₀) where N₀ is initial domain size
        - Each stage gets error budget ε / lg(N₀)
        - For domain size n: t_n ≥ 7.23 lg(n) + 4.82 * security_param + 4.82 * lg(lg(N₀))
        """
        if n <= 1:
            return 0

        lg_n = math.log2(n)
        lg_n0 = math.log2(self._domain_size)
        lg_lg_n0 = math.log2(lg_n0) if lg_n0 > 1 else 0

        return math.ceil(7.23 * lg_n + 4.82 * self._security_param + 4.82 * lg_lg_n0)

    def _round_constant(self, n: int, round_idx: int) -> int:
        """
        Derive round constant K_i for domain size n and round index i.

        Returns a value in [0, n).
        """
        label = f"K:{n}:{round_idx}".encode()
        digest = hashlib.shake_256(self._key + label).digest(32)
        # SEC: Modular bias is at most n/2^256, negligible for 256-bit digest.
        value = int.from_bytes(digest, "little")
        return value % n

    def _round_function(self, n: int, round_idx: int, x_hat: int) -> bool:
        """
        Round function F(i, x̂) returning True (swap) or False (no swap).

        This is the PRF that determines whether to swap x with its partner.
        """
        label = f"F:{n}:{round_idx}:{x_hat}".encode()
        digest = hashlib.shake_256(self._key + label).digest(1)
        # Use least significant bit
        return (digest[0] & 1) == 1

    def _swap_or_not(self, n: int, x: int) -> int:
        """
        Apply Swap-or-Not shuffle for domain size n.

        OPT: Each round calls SHAKE-256 twice (round constant + round function).
        Could use a faster PRF (e.g., AES-based) for significant speedup.

        For each round:
        1. Compute partner x' = K_i - x (mod n)
        2. Compute canonical name x̂ = max(x, x')
        3. If F(i, x̂) = 1, swap x with x'
        """
        t_n = self._num_sn_rounds(n)

        for i in range(t_n):
            k_i = self._round_constant(n, i)
            x_prime = (k_i - x) % n
            x_hat = max(x, x_prime)

            if self._round_function(n, i, x_hat):
                x = x_prime

        return x

    def _inverse_swap_or_not(self, n: int, y: int) -> int:
        """
        Inverse Swap-or-Not shuffle.

        Same as forward but rounds are applied in reverse order.
        (Swap-or-Not is self-inverse per round, so we just reverse order)
        """
        t_n = self._num_sn_rounds(n)

        for i in range(t_n - 1, -1, -1):
            k_i = self._round_constant(n, i)
            y_prime = (k_i - y) % n
            y_hat = max(y, y_prime)

            if self._round_function(n, i, y_hat):
                y = y_prime

        return y

    def forward(self, x: int) -> int:
        """
        Forward permutation: P(x).

        E^N_KF(X) from MR14 Figure 1.

        Args:
            x: Input in [0, N)

        Returns:
            Output in [0, N)
        """
        if x < 0 or x >= self._domain_size:
            raise ValueError(f"Input {x} out of range [0, {self._domain_size})")

        return self._forward_recursive(self._domain_size, x)

    def _forward_recursive(self, n: int, x: int) -> int:
        """Recursive forward permutation for domain size n."""
        if n <= 1:
            return x

        # Apply Swap-or-Not shuffle
        x = self._swap_or_not(n, x)

        # Split point
        p_n = n // 2

        # Recurse on first pile only
        if x < p_n:
            return self._forward_recursive(p_n, x)
        else:
            return x

    def inverse(self, y: int) -> int:
        """
        Inverse permutation: P^{-1}(y).

        D^N_KF(Y) - inverse of the SR shuffle.

        Args:
            y: Input in [0, N)

        Returns:
            Output in [0, N)
        """
        if y < 0 or y >= self._domain_size:
            raise ValueError(f"Input {y} out of range [0, {self._domain_size})")

        return self._inverse_recursive(self._domain_size, y)

    def _inverse_recursive(self, n: int, y: int) -> int:
        """Recursive inverse permutation for domain size n."""
        if n <= 1:
            return y

        # Split point
        p_n = n // 2

        # Inverse: recurse first if in first pile
        if y < p_n:
            y = self._inverse_recursive(p_n, y)

        # Apply inverse Swap-or-Not shuffle
        y = self._inverse_swap_or_not(n, y)

        return y

