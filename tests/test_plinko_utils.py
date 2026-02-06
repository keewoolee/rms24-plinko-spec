"""Tests for Plinko utility functions."""

import pytest
from pir.plinko.utils import (
    PRG,
    derive_seed,
)


class TestPRG:
    """Tests for PRG class."""

    def test_deterministic(self):
        """Same seed should give same sequence."""
        seed = b"test_seed_bytes_1234567890123456"
        rng1 = PRG(seed)
        rng2 = PRG(seed)

        for _ in range(10):
            assert rng1.random_bytes(16) == rng2.random_bytes(16)

    def test_binomial(self):
        """Binomial should give reasonable results."""
        seed = b"test_seed_bytes_1234567890123456"
        rng = PRG(seed)
        n, p = 100, 0.5

        results = [rng.binomial(n, p) for _ in range(100)]
        mean = sum(results) / len(results)

        # Mean should be close to n*p = 50
        assert 40 < mean < 60


class TestDeriveSeed:
    """Tests for derive_seed."""

    def test_deterministic(self):
        """Same inputs should give same seed."""
        key = b"test_key_12345678901234567890123"
        label = b"label1"

        seed1 = derive_seed(key, label)
        seed2 = derive_seed(key, label)

        assert seed1 == seed2

    def test_different_labels(self):
        """Different labels should give different seeds."""
        key = b"test_key_12345678901234567890123"

        seed1 = derive_seed(key, b"label1")
        seed2 = derive_seed(key, b"label2")

        assert seed1 != seed2

    def test_returns_bytes(self):
        """derive_seed should return bytes."""
        key = b"test_key_12345678901234567890123"
        label = b"label1"

        seed = derive_seed(key, label)

        assert isinstance(seed, bytes)
        assert len(seed) == 32  # SHA-256 output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
