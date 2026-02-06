"""Tests for Sometimes-Recurse PRP."""

import pytest
from pir.plinko.prp import PRP


class TestPRP:
    """Tests for PRP."""

    def test_basic_permutation(self):
        """Test that forward is a permutation."""
        prp = PRP(domain_size=10)

        outputs = set()
        for x in range(10):
            y = prp.forward(x)
            assert 0 <= y < 10, f"Output {y} out of range"
            outputs.add(y)

        # Should be a bijection
        assert outputs == set(range(10))

    def test_inverse_correctness(self):
        """Test that inverse correctly inverts forward."""
        prp = PRP(domain_size=20)

        for x in range(20):
            y = prp.forward(x)
            x_recovered = prp.inverse(y)
            assert x_recovered == x, f"Inverse failed: {x} -> {y} -> {x_recovered}"

    def test_edge_cases(self):
        """Test edge cases."""
        # Domain size 1
        prp = PRP(domain_size=1)
        assert prp.forward(0) == 0
        assert prp.inverse(0) == 0

        # Domain size 2
        prp = PRP(domain_size=2)
        outputs = {prp.forward(0), prp.forward(1)}
        assert outputs == {0, 1}

    def test_non_power_of_two(self):
        """Test with non-power-of-two domain sizes."""
        for n in [7, 13, 25, 99, 127]:
            prp = PRP(domain_size=n)

            # Check bijection
            outputs = set()
            for x in range(n):
                y = prp.forward(x)
                assert 0 <= y < n
                outputs.add(y)
            assert outputs == set(range(n))

            # Check inverse
            for x in range(n):
                assert prp.inverse(prp.forward(x)) == x

    def test_larger_domain(self):
        """Test with larger domain size."""
        prp = PRP(domain_size=1000)

        # Spot check some values
        for x in [0, 100, 500, 999]:
            y = prp.forward(x)
            assert 0 <= y < 1000
            assert prp.inverse(y) == x

    def test_pseudorandom_distribution(self):
        """Output should look pseudorandom."""
        n = 100
        prp = PRP(domain_size=n)

        # Check that not too many elements stay in place
        fixed_points = sum(1 for x in range(n) if prp.forward(x) == x)
        # Expected ~1 fixed point on average for random permutation
        # Allow up to 15 for reasonable tolerance
        assert fixed_points < 15, f"Too many fixed points: {fixed_points}"


class TestPRPErrors:
    """Test error handling."""

    def test_invalid_domain_size(self):
        """Non-positive domain size should raise error."""
        with pytest.raises(ValueError):
            PRP(domain_size=0)
        with pytest.raises(ValueError):
            PRP(domain_size=-1)

    def test_forward_out_of_range(self):
        """Input out of range should raise error."""
        prp = PRP(domain_size=10)
        with pytest.raises(ValueError):
            prp.forward(-1)
        with pytest.raises(ValueError):
            prp.forward(10)

    def test_inverse_out_of_range(self):
        """Input out of range should raise error."""
        prp = PRP(domain_size=10)
        with pytest.raises(ValueError):
            prp.inverse(-1)
        with pytest.raises(ValueError):
            prp.inverse(10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
