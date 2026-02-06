"""Tests for Plinko PIR scheme."""

import pytest
import secrets
from pir.plinko import Params, Client, Server


def create_random_database(num_entries: int, entry_size: int) -> list[bytes]:
    """Create a random database for testing."""
    return [secrets.token_bytes(entry_size) for _ in range(num_entries)]


class TestParams:
    """Tests for Params."""

    def test_basic(self):
        """Test basic parameter creation."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        assert params.num_entries == 100
        assert params.entry_size == 32
        assert params.num_blocks % 2 == 0
        assert params.num_reg_hints > 0

    def test_block_of(self):
        """Test block_of method."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        for i in range(100):
            block = params.block_of(i)
            assert 0 <= block < params.num_blocks

    def test_offset_in_block(self):
        """Test offset_in_block method."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        for i in range(100):
            offset = params.offset_in_block(i)
            assert 0 <= offset < params.block_size


class TestPlinkoPIR:
    """Tests for the full Plinko PIR workflow."""

    def test_basic_query(self):
        """Test basic single query."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)

        # Offline phase
        client.generate_hints(server.stream_database())

        # Query a random entry
        target_index = 42
        queries = client.query([target_index])
        responses = server.answer(queries)
        results = client.extract(responses)

        assert len(results) == 1
        assert results[0] == database[target_index]

        # Cleanup
        client.replenish_hints()

    def test_multiple_queries(self):
        """Test multiple queries in one batch."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)

        client.generate_hints(server.stream_database())

        # Query multiple entries
        target_indices = [10, 25, 50, 75, 99]
        queries = client.query(target_indices)
        responses = server.answer(queries)
        results = client.extract(responses)

        assert len(results) == len(target_indices)
        for i, idx in enumerate(target_indices):
            assert results[i] == database[idx]

        client.replenish_hints()

    def test_repeated_queries(self):
        """Test multiple query rounds."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)

        client.generate_hints(server.stream_database())

        # Multiple query rounds
        for round_num in range(5):
            target_index = (round_num * 17) % params.num_entries
            queries = client.query([target_index])
            responses = server.answer(queries)
            results = client.extract(responses)

            assert results[0] == database[target_index]
            client.replenish_hints()

    def test_query_all_entries(self):
        """Test querying every entry in a small database."""
        # Need more hints to guarantee coverage of all entries
        params = Params(num_entries=20, entry_size=16, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)

        client.generate_hints(server.stream_database())

        # Query each entry
        for target_index in range(params.num_entries):
            if client.remaining_queries() == 0:
                # Need to regenerate hints
                client.generate_hints(server.stream_database())

            queries = client.query([target_index])
            responses = server.answer(queries)
            results = client.extract(responses)

            assert results[0] == database[target_index], f"Failed at index {target_index}"
            client.replenish_hints()

        # Query all indices again to verify continued operation
        # Must regenerate hints to clear cache (all entries are now cached)
        client.generate_hints(server.stream_database())

        for target_index in range(params.num_entries):
            if client.remaining_queries() == 0:
                client.generate_hints(server.stream_database())

            queries = client.query([target_index])
            responses = server.answer(queries)
            results = client.extract(responses)
            assert results[0] == database[target_index]
            client.replenish_hints()

    def test_database_update(self):
        """Test hint updates when database changes."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)

        client.generate_hints(server.stream_database())

        # Update some entries
        update_index = 42
        new_value = secrets.token_bytes(params.entry_size)
        updates = server.update_entries({update_index: new_value})

        # Update client hints
        client.update_hints(updates)

        # Query the updated entry
        queries = client.query([update_index])
        responses = server.answer(queries)
        results = client.extract(responses)

        assert results[0] == new_value
        client.replenish_hints()

    def test_remaining_queries(self):
        """Test remaining_queries tracking."""
        params = Params(num_entries=100, entry_size=32, security_param=40, num_backup_hints=10)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)

        client.generate_hints(server.stream_database())

        initial_remaining = client.remaining_queries()
        assert initial_remaining == 10  # num_backup_hints

        # Query and check remaining decreases
        for i in range(5):
            queries = client.query([i])
            responses = server.answer(queries)
            client.extract(responses)
            client.replenish_hints()

        assert client.remaining_queries() == initial_remaining - 5


class TestPlinkoPIREdgeCases:
    """Test edge cases and error handling."""

    def test_query_before_hints(self):
        """Query before generate_hints should raise error."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        client = Client(params)

        with pytest.raises(RuntimeError):
            client.query([0])

    def test_extract_before_query(self):
        """Extract before query should raise error."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)
        client.generate_hints(server.stream_database())

        with pytest.raises(RuntimeError):
            client.extract([])

    def test_replenish_before_extract(self):
        """Replenish before extract should raise error."""
        params = Params(num_entries=100, entry_size=32, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)
        client.generate_hints(server.stream_database())

        client.query([0])
        with pytest.raises(RuntimeError):
            client.replenish_hints()

    def test_small_database(self):
        """Test with very small database."""
        params = Params(num_entries=4, entry_size=8, security_param=40)
        database = create_random_database(params.num_entries, params.entry_size)

        server = Server(database, params)
        client = Client(params)
        client.generate_hints(server.stream_database())

        for idx in range(4):
            if client.remaining_queries() == 0:
                client.generate_hints(server.stream_database())

            queries = client.query([idx])
            responses = server.answer(queries)
            results = client.extract(responses)
            assert results[0] == database[idx]
            client.replenish_hints()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
