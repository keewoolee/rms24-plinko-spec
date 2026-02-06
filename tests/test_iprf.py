"""Tests for Invertible PRF (iPRF)."""

import pytest
from pir.plinko.iprf import InvertiblePRF


class TestInvertiblePRF:
    """Tests for InvertiblePRF."""

    def test_forward_basic(self):
        """Test basic forward evaluation."""
        iprf = InvertiblePRF(domain_size=100, range_size=10)

        # Each input should map to a valid output
        for x in range(100):
            y = iprf.forward(x)
            assert 0 <= y < 10, f"Output {y} out of range"

    def test_inverse_basic(self):
        """Test basic inverse evaluation."""
        iprf = InvertiblePRF(domain_size=100, range_size=10)

        # Inverse should return valid inputs
        all_preimages = set()
        for y in range(10):
            preimages = iprf.inverse(y)
            for x in preimages:
                assert 0 <= x < 100, f"Preimage {x} out of range"
            all_preimages.update(preimages)

        # All inputs should be accounted for
        assert all_preimages == set(range(100))

    def test_forward_inverse_consistency(self):
        """Test that forward(x) ∈ inverse for all x."""
        iprf = InvertiblePRF(domain_size=50, range_size=10)

        for x in range(50):
            y = iprf.forward(x)
            preimages = iprf.inverse(y)
            assert x in preimages, f"x={x} not in inverse of y={y}"

    def test_inverse_forward_consistency(self):
        """Test that all preimages of y map to y."""
        iprf = InvertiblePRF(domain_size=50, range_size=10)

        for y in range(10):
            preimages = iprf.inverse(y)
            for x in preimages:
                assert iprf.forward(x) == y, f"forward({x}) != {y}"

    def test_different_instances(self):
        """Different instances should produce different mappings."""
        iprf1 = InvertiblePRF(domain_size=100, range_size=10)
        iprf2 = InvertiblePRF(domain_size=100, range_size=10)

        # Mappings should be different (different random keys)
        different_count = sum(
            1 for x in range(100) if iprf1.forward(x) != iprf2.forward(x)
        )
        assert different_count > 0

    def test_consistency(self):
        """Test forward/inverse consistency."""
        for n, m in [(10, 4), (50, 10), (100, 20), (20, 20)]:
            iprf = InvertiblePRF(domain_size=n, range_size=m)
            # For each x, x should be in inverse(forward(x))
            for x in range(n):
                y = iprf.forward(x)
                assert x in iprf.inverse(y), f"x={x} not in inverse({y})"

    def test_distribution(self):
        """Outputs should be roughly uniformly distributed."""
        n, m = 1000, 10
        iprf = InvertiblePRF(domain_size=n, range_size=m)

        # Count outputs
        bin_counts = [0] * m
        for x in range(n):
            y = iprf.forward(x)
            bin_counts[y] += 1

        # Each output should appear roughly n/m = 100 times
        expected = n / m
        for count in bin_counts:
            assert expected * 0.5 < count < expected * 1.5

    def test_edge_cases(self):
        """Test edge cases."""
        # n == m (like random permutation)
        iprf = InvertiblePRF(domain_size=10, range_size=10)
        for x in range(10):
            assert x in iprf.inverse(iprf.forward(x))

        # n < m (some outputs may be empty)
        iprf = InvertiblePRF(domain_size=5, range_size=10)
        for x in range(5):
            assert x in iprf.inverse(iprf.forward(x))

        # n > m (multiple inputs per output)
        iprf = InvertiblePRF(domain_size=20, range_size=5)
        for x in range(20):
            assert x in iprf.inverse(iprf.forward(x))


class TestIPRFErrors:
    """Test error handling."""

    def test_invalid_domain_size(self):
        """Non-positive domain size should raise error."""
        with pytest.raises(ValueError):
            InvertiblePRF(domain_size=0, range_size=10)
        with pytest.raises(ValueError):
            InvertiblePRF(domain_size=-1, range_size=10)

    def test_invalid_range_size(self):
        """Non-positive range size should raise error."""
        with pytest.raises(ValueError):
            InvertiblePRF(domain_size=10, range_size=0)
        with pytest.raises(ValueError):
            InvertiblePRF(domain_size=10, range_size=-1)

    def test_forward_out_of_range(self):
        """Input out of range should raise error."""
        iprf = InvertiblePRF(domain_size=10, range_size=5)
        with pytest.raises(ValueError):
            iprf.forward(-1)
        with pytest.raises(ValueError):
            iprf.forward(10)

    def test_inverse_out_of_range(self):
        """Output out of range should raise error."""
        iprf = InvertiblePRF(domain_size=10, range_size=5)
        with pytest.raises(ValueError):
            iprf.inverse(-1)
        with pytest.raises(ValueError):
            iprf.inverse(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
