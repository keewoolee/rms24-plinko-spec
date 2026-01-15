#!/usr/bin/env python3
"""
Demo of RMS24 Single-Server PIR scheme.

This demonstrates the basic usage of the PIR protocol:
1. Create a database
2. Run the offline phase
3. Make private queries
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rms24.params import Params
from rms24.client import Client
from rms24.server import Server
from rms24.utils import create_sequential_database


def main():
    print("=" * 60)
    print("RMS24 Single-Server PIR Demo")
    print("=" * 60)

    # Parameters: 4096 entries, 32-byte entries, lambda=10 (small for demo)
    n = 4096
    entry_size = 32
    lambda_ = 10

    params = Params(n=n, entry_size=entry_size, lambda_=lambda_)
    print(f"\nParameters: {params}")
    print(f"  - Database entries (n): {params.n}")
    print(f"  - Block size (w): {params.w}")
    print(f"  - Blocks (c): {params.c}")
    print(f"  - Regular hints: {params.num_reg_hints}")
    print(f"  - Backup hints: {params.num_backup_hints}")

    # Create database with sequential values (for easy verification)
    print("\n[1] Creating database...")
    db = create_sequential_database(params.n, entry_size)
    print(f"    Database created with {len(db)} entries")

    # Create server and client
    server = Server(db, params)
    client = Client(params)

    # Generate hints
    print("\n[2] Generating hints (streaming database)...")
    start = time.time()
    client.generate_hints(server.stream_database())
    offline_time = time.time() - start
    print(f"    Offline phase completed in {offline_time:.3f}s")

    # Make some queries
    print("\n[3] Making private queries...")
    test_indices = [0, 1, 100, 1000, params.n - 1]
    num_queries = min(10, client.remaining_queries())

    query_times = []
    for i, idx in enumerate(test_indices[:num_queries]):
        start = time.time()
        query = client.query(idx)
        response = server.answer(query)
        result = client.extract(response)
        client.replenish_hint()
        query_time = time.time() - start
        query_times.append(query_time)

        # Verify correctness
        expected = db[idx]
        correct = result == expected

        # Display result (first 8 bytes as int for sequential DB)
        result_val = int.from_bytes(result[:8], "little")
        print(f"    Query {i+1}: index={idx}, result={result_val}, "
              f"correct={correct}, time={query_time*1000:.2f}ms")

    # Summary
    print("\n[4] Summary")
    print(f"    Remaining queries: {client.remaining_queries()}")
    if query_times:
        avg_time = sum(query_times) / len(query_times)
        print(f"    Average query time: {avg_time*1000:.2f}ms")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
