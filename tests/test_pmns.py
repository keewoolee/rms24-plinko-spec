"""Tests for PMNS (Pseudorandom Multinomial Sampler)."""

import pytest
from pir.plinko.pmns import PMNS, PMNSNode


class TestPMNSNode:
    """Tests for PMNSNode."""

    def test_is_leaf(self):
        """Test leaf detection."""
        leaf = PMNSNode(start=0, count=5, low=3, high=3)
        assert leaf.is_leaf()

        internal = PMNSNode(start=0, count=5, low=0, high=3)
        assert not internal.is_leaf()


class TestPMNS:
    """Tests for PMNS."""

    def test_forward_basic(self):
        """Test basic forward mapping."""
        pmns = PMNS(domain_size=10, range_size=4)

        # Each ball should map to a valid bin
        for x in range(10):
            y = pmns.forward(x)
            assert 0 <= y < 4

    def test_inverse_basic(self):
        """Test basic inverse mapping."""
        pmns = PMNS(domain_size=10, range_size=4)

        # Inverse should return valid ball indices
        all_balls = set()
        for y in range(4):
            balls = pmns.inverse(y)
            for x in balls:
                assert 0 <= x < 10
            all_balls.update(balls)

        # All balls should be accounted for
        assert all_balls == set(range(10))

    def test_forward_inverse_consistency(self):
        """Test that forward and inverse are consistent."""
        pmns = PMNS(domain_size=20, range_size=5)

        # For each ball, check that it appears in the inverse of its bin
        for x in range(20):
            y = pmns.forward(x)
            balls_in_bin = pmns.inverse(y)
            assert x in balls_in_bin

    def test_distribution(self):
        """Balls should be roughly uniformly distributed."""
        n, m = 1000, 10
        pmns = PMNS(domain_size=n, range_size=m)

        # Count balls in each bin
        bin_counts = [0] * m
        for x in range(n):
            y = pmns.forward(x)
            bin_counts[y] += 1

        # Each bin should have roughly n/m = 100 balls
        # Allow 50% deviation
        expected = n / m
        for count in bin_counts:
            assert expected * 0.5 < count < expected * 1.5

    def test_edge_cases(self):
        """Test edge cases."""
        # Single ball
        pmns = PMNS(domain_size=1, range_size=5)
        y = pmns.forward(0)
        assert 0 <= y < 5
        assert pmns.inverse(y) == {0}

        # Single bin
        pmns = PMNS(domain_size=10, range_size=1)
        for x in range(10):
            assert pmns.forward(x) == 0
        assert pmns.inverse(0) == set(range(10))

    def test_empty_bins(self):
        """Some bins may be empty, and that's OK."""
        pmns = PMNS(domain_size=3, range_size=10)

        # With 3 balls and 10 bins, most bins will be empty
        non_empty = 0
        for y in range(10):
            balls = pmns.inverse(y)
            if balls:
                non_empty += 1

        # Should have exactly 3 non-empty bins (one per ball) or fewer if collisions
        assert 1 <= non_empty <= 3

    def test_large_n(self):
        """Test with larger n."""
        pmns = PMNS(domain_size=10000, range_size=100)

        # Spot check some values
        for x in [0, 100, 5000, 9999]:
            y = pmns.forward(x)
            assert 0 <= y < 100
            assert x in pmns.inverse(y)


class TestPMNSErrors:
    """Test error handling."""

    def test_invalid_domain_size(self):
        """Negative domain_size should raise error."""
        with pytest.raises(ValueError):
            PMNS(domain_size=-1, range_size=4)

    def test_invalid_range_size(self):
        """Non-positive range_size should raise error."""
        with pytest.raises(ValueError):
            PMNS(domain_size=10, range_size=0)
        with pytest.raises(ValueError):
            PMNS(domain_size=10, range_size=-1)

    def test_forward_out_of_range(self):
        """Ball index out of range should raise error."""
        pmns = PMNS(domain_size=10, range_size=4)
        with pytest.raises(ValueError):
            pmns.forward(-1)
        with pytest.raises(ValueError):
            pmns.forward(10)

    def test_inverse_out_of_range(self):
        """Bin index out of range should raise error."""
        pmns = PMNS(domain_size=10, range_size=4)
        with pytest.raises(ValueError):
            pmns.inverse(-1)
        with pytest.raises(ValueError):
            pmns.inverse(4)

    def test_inverse_empty_domain(self):
        """Inverse with domain_size=0 should raise error."""
        pmns = PMNS(domain_size=0, range_size=4)
        with pytest.raises(ValueError):
            pmns.inverse(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
