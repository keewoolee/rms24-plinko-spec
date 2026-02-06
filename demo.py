#!/usr/bin/env python3
"""
Demo and benchmarks for RMS24 PIR, Plinko PIR, and Keyword PIR.

Usage:
    python3 demo.py --rms24       # Run RMS24 demo
    python3 demo.py --plinko      # Run Plinko demo
    python3 demo.py --kpir-rms24  # Run Keyword PIR demo (RMS24 backend)
    python3 demo.py --kpir-plinko # Run Keyword PIR demo (Plinko backend)
    python3 demo.py --benchmark   # Benchmark PIR and KPIR (RMS24 vs Plinko)
"""

import argparse
import time

from pir import rms24, plinko
from pir.rms24 import Params, Client, Server
from pir.plinko import Params as PlinkoParams, Client as PlinkoClient, Server as PlinkoServer
from pir.keyword_pir import KPIRParams, KPIRClient, KPIRServer


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_count(n: int) -> str:
    """Format number with K/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def format_bytes(n: int) -> str:
    """Format bytes with KiB/MiB/GiB suffix."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GiB"
    if n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def format_time(seconds: float) -> str:
    """Format time with appropriate unit."""
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 0.001:
        return f"{seconds*1000:.2f}ms"
    return f"{seconds*1_000_000:.1f}us"


# =============================================================================
# RMS24 PIR Demo
# =============================================================================


def create_database(num_entries: int, entry_size: int) -> list[bytes]:
    """Create a database with sequential values."""
    return [
        i.to_bytes(min(entry_size, 8), "little").ljust(entry_size, b"\x00")
        for i in range(num_entries)
    ]


def compute_query_size(params: Params) -> int:
    """Compute size of a single query in bytes."""
    mask_size = (params.num_blocks + 7) // 8
    # offsets: num_blocks/2 integers, assume 4 bytes each (could be 2 for small DBs)
    offsets_size = (params.num_blocks // 2) * 4
    return mask_size + offsets_size


def compute_response_size(params: Params) -> int:
    """Compute size of a single response in bytes."""
    return 2 * params.entry_size  # parity_0 + parity_1


def compute_plinko_query_size(params: PlinkoParams) -> int:
    """Compute size of a single Plinko query in bytes."""
    mask_size = (params.num_blocks + 7) // 8
    offsets_size = (params.num_blocks // 2) * 4
    return mask_size + offsets_size


def compute_plinko_response_size(params: PlinkoParams) -> int:
    """Compute size of a single Plinko response in bytes."""
    return 2 * params.entry_size  # parity_0 + parity_1


def run_rms24_demo(num_entries: int, entry_size: int, security_param: int, num_queries: int):
    """Run the RMS24 PIR demo with detailed metrics."""
    print("=" * 70)
    print("RMS24 Single-Server PIR - Demo & Benchmarks")
    print("=" * 70)

    # Setup parameters
    params = Params(
        num_entries=num_entries,
        entry_size=entry_size,
        security_param=security_param,
    )

    db_size = num_entries * entry_size
    query_size = compute_query_size(params)
    response_size = compute_response_size(params)

    print(f"\n{'Parameters':─^70}")
    print(f"  Database size:      {format_bytes(db_size):>12}  ({format_count(num_entries)} × {entry_size}B)")
    print(f"  Block size (w):     {params.block_size:>12}")
    print(f"  Num blocks (c):     {params.num_blocks:>12}")
    print(f"  Regular hints (M):  {format_count(params.num_reg_hints):>12}")
    print(f"  Backup hints:       {format_count(params.num_backup_hints):>12}")
    print(f"  Security param (λ): {security_param:>12}")

    # Create database
    print(f"\n{'Database Creation':─^70}")
    start = time.perf_counter()
    db = create_database(num_entries, entry_size)
    db_time = time.perf_counter() - start
    print(f"  Time: {format_time(db_time)}")

    server = Server(db, params)
    client = Client(params)

    # Offline phase
    print(f"\n{'Offline Phase (Hint Generation)':─^70}")
    start = time.perf_counter()
    client.generate_hints(server.stream_database())
    offline_time = time.perf_counter() - start

    print(f"  Time:           {format_time(offline_time):>12}")
    print(f"  Throughput:     {num_entries / offline_time:>12,.0f} entries/s")
    print(f"  Download:       {format_bytes(db_size):>12}  (full database)")

    # Online phase
    num_queries = min(num_queries, client.remaining_queries())
    print(f"\n{'Online Phase (' + str(num_queries) + ' queries)':─^70}")

    test_indices = [(i * num_entries) // num_queries for i in range(num_queries)]

    # Detailed timing breakdown
    query_gen_times = []
    server_answer_times = []
    extract_times = []
    replenish_times = []
    all_correct = True

    for idx in test_indices:
        # Query generation (client)
        start = time.perf_counter()
        queries = client.query([idx])
        query_gen_times.append(time.perf_counter() - start)

        # Server answer
        start = time.perf_counter()
        responses = server.answer(queries)
        server_answer_times.append(time.perf_counter() - start)

        # Extract result (client)
        start = time.perf_counter()
        [result] = client.extract(responses)
        extract_times.append(time.perf_counter() - start)

        # Replenish hints (client)
        start = time.perf_counter()
        client.replenish_hints()
        replenish_times.append(time.perf_counter() - start)

        if result != db[idx]:
            all_correct = False
            print(f"  ERROR: index {idx} returned wrong result!")

    avg_query_gen = sum(query_gen_times) / len(query_gen_times)
    avg_server = sum(server_answer_times) / len(server_answer_times)
    avg_extract = sum(extract_times) / len(extract_times)
    avg_replenish = sum(replenish_times) / len(replenish_times)
    avg_total = avg_query_gen + avg_server + avg_extract + avg_replenish

    print(f"  Correctness:    {'PASS' if all_correct else 'FAIL':>12}")
    print(f"\n  Timing breakdown (avg per query):")
    print(f"    Client query():      {format_time(avg_query_gen):>10}")
    print(f"    Server answer():     {format_time(avg_server):>10}")
    print(f"    Client extract():    {format_time(avg_extract):>10}")
    print(f"    Client replenish():  {format_time(avg_replenish):>10}")
    print(f"    ─────────────────────────────────")
    print(f"    Total:               {format_time(avg_total):>10}")

    # Communication costs
    print(f"\n{'Communication Costs':─^70}")
    print(f"  Offline download:   {format_bytes(db_size):>12}")
    print(f"  Query size:         {format_bytes(query_size):>12}  (mask + offsets)")
    print(f"  Response size:      {format_bytes(response_size):>12}  (2 × entry_size)")
    print(f"  Online total:       {format_bytes(query_size + response_size):>12}  per query")

    # Update phase
    print(f"\n{'Update Phase (10 updates)':─^70}")
    update_server_times = []
    update_client_times = []

    for i in range(10):
        idx = (i * num_entries) // 10
        new_value = bytes([i] * entry_size)

        start = time.perf_counter()
        updates = server.update_entries({idx: new_value})
        update_server_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        client.update_hints(updates)
        update_client_times.append(time.perf_counter() - start)

    avg_server_update = sum(update_server_times) / len(update_server_times)
    avg_client_update = sum(update_client_times) / len(update_client_times)
    update_msg_size = 4 + entry_size  # index (4 bytes) + delta

    print(f"  Timing breakdown (avg per update):")
    print(f"    Server update_entries(): {format_time(avg_server_update):>10}")
    print(f"    Client update_hints():   {format_time(avg_client_update):>10}")
    print(f"  Update message size:   {format_bytes(update_msg_size):>12}  (index + delta)")

    # Summary
    print(f"\n{'Summary':─^70}")
    print(f"  Offline:  {format_time(offline_time)} to download {format_bytes(db_size)}")
    print(f"  Online:   {format_time(avg_total)}/query, {format_bytes(query_size + response_size)} communication")
    print(f"  Capacity: {client.remaining_queries()} queries remaining")
    print("=" * 70)


# =============================================================================
# Plinko PIR Demo
# =============================================================================


def run_plinko_demo(num_entries: int, entry_size: int, security_param: int, num_queries: int):
    """Run the Plinko PIR demo with detailed metrics."""
    print("=" * 70)
    print("Plinko Single-Server PIR - Demo & Benchmarks")
    print("=" * 70)

    # Note about performance
    print(f"\n{'Note':─^70}")
    print("  The current implementation is not optimized for performance.")
    print("  Production use would need hardware-accelerated cryptography.")

    # Warn if parameters are large
    if num_entries > 64:
        print(f"\n  Warning: num_entries={num_entries} may be slow. Consider --entries 64")

    # Setup parameters
    params = PlinkoParams(
        num_entries=num_entries,
        entry_size=entry_size,
        security_param=security_param,
    )

    db_size = num_entries * entry_size
    query_size = compute_plinko_query_size(params)
    response_size = compute_plinko_response_size(params)

    print(f"\n{'Parameters':─^70}")
    print(f"  Database size:      {format_bytes(db_size):>12}  ({format_count(num_entries)} × {entry_size}B)")
    print(f"  Block size (w):     {params.block_size:>12}")
    print(f"  Num blocks (c):     {params.num_blocks:>12}")
    print(f"  Regular hints:      {format_count(params.num_reg_hints):>12}")
    print(f"  Backup hints:       {format_count(params.num_backup_hints):>12}")
    print(f"  Security param (λ): {security_param:>12}")

    # Create database
    print(f"\n{'Database Creation':─^70}")
    start = time.perf_counter()
    db = create_database(num_entries, entry_size)
    db_time = time.perf_counter() - start
    print(f"  Time: {format_time(db_time)}")

    server = PlinkoServer(db, params)
    client = PlinkoClient(params)

    # Offline phase
    print(f"\n{'Offline Phase (Hint Generation)':─^70}")
    start = time.perf_counter()
    client.generate_hints(server.stream_database())
    offline_time = time.perf_counter() - start

    print(f"  Time:           {format_time(offline_time):>12}")
    print(f"  Throughput:     {num_entries / offline_time:>12,.0f} entries/s")
    print(f"  Download:       {format_bytes(db_size):>12}  (full database)")

    # Online phase
    num_queries = min(num_queries, client.remaining_queries())
    print(f"\n{'Online Phase (' + str(num_queries) + ' queries)':─^70}")

    test_indices = [(i * num_entries) // num_queries for i in range(num_queries)]

    # Detailed timing breakdown
    query_gen_times = []
    server_answer_times = []
    extract_times = []
    replenish_times = []
    all_correct = True

    for idx in test_indices:
        # Query generation (client) - includes O(1) hint search!
        start = time.perf_counter()
        queries = client.query([idx])
        query_gen_times.append(time.perf_counter() - start)

        # Server answer
        start = time.perf_counter()
        responses = server.answer(queries)
        server_answer_times.append(time.perf_counter() - start)

        # Extract result (client)
        start = time.perf_counter()
        [result] = client.extract(responses)
        extract_times.append(time.perf_counter() - start)

        # Replenish hints (client)
        start = time.perf_counter()
        client.replenish_hints()
        replenish_times.append(time.perf_counter() - start)

        if result != db[idx]:
            all_correct = False
            print(f"  ERROR: index {idx} returned wrong result!")

    avg_query_gen = sum(query_gen_times) / len(query_gen_times)
    avg_server = sum(server_answer_times) / len(server_answer_times)
    avg_extract = sum(extract_times) / len(extract_times)
    avg_replenish = sum(replenish_times) / len(replenish_times)
    avg_total = avg_query_gen + avg_server + avg_extract + avg_replenish

    print(f"  Correctness:    {'PASS' if all_correct else 'FAIL':>12}")
    print(f"\n  Timing breakdown (avg per query):")
    print(f"    Client query():      {format_time(avg_query_gen):>10}  (includes O(1) hint search)")
    print(f"    Server answer():     {format_time(avg_server):>10}")
    print(f"    Client extract():    {format_time(avg_extract):>10}")
    print(f"    Client replenish():  {format_time(avg_replenish):>10}")
    print(f"    ─────────────────────────────────")
    print(f"    Total:               {format_time(avg_total):>10}")

    # Communication costs
    print(f"\n{'Communication Costs':─^70}")
    print(f"  Offline download:   {format_bytes(db_size):>12}")
    print(f"  Query size:         {format_bytes(query_size):>12}  (mask + offsets)")
    print(f"  Response size:      {format_bytes(response_size):>12}  (2 × entry_size)")
    print(f"  Online total:       {format_bytes(query_size + response_size):>12}  per query")

    # Update phase
    print(f"\n{'Update Phase (10 updates)':─^70}")
    update_server_times = []
    update_client_times = []

    for i in range(10):
        idx = (i * num_entries) // 10
        new_value = bytes([i] * entry_size)

        start = time.perf_counter()
        updates = server.update_entries({idx: new_value})
        update_server_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        client.update_hints(updates)
        update_client_times.append(time.perf_counter() - start)

    avg_server_update = sum(update_server_times) / len(update_server_times)
    avg_client_update = sum(update_client_times) / len(update_client_times)
    update_msg_size = 4 + entry_size  # index (4 bytes) + delta

    print(f"  Timing breakdown (avg per update):")
    print(f"    Server update_entries(): {format_time(avg_server_update):>10}")
    print(f"    Client update_hints():   {format_time(avg_client_update):>10}  (O(1) via iPRF)")
    print(f"  Update message size:   {format_bytes(update_msg_size):>12}  (index + delta)")

    # Summary
    print(f"\n{'Summary':─^70}")
    print(f"  Offline:  {format_time(offline_time)} to download {format_bytes(db_size)}")
    print(f"  Online:   {format_time(avg_total)}/query, {format_bytes(query_size + response_size)} communication")
    print(f"  Capacity: {client.remaining_queries()} queries remaining")
    print("=" * 70)


# =============================================================================
# Benchmark: PIR and KPIR Comparison
# =============================================================================


def run_benchmark(num_entries: int, entry_size: int, key_size: int, security_param: int, num_queries: int):
    """Benchmark RMS24 vs Plinko for both PIR and KPIR."""
    print("=" * 70)
    print("PIR & KPIR Benchmark: RMS24 vs Plinko")
    print("=" * 70)

    # Note about performance
    print(f"\n{'Note':─^70}")
    print("  The current Plinko implementation uses SHAKE-256 for the PRP,")
    print("  which is slower than optimized implementations. Production use")
    print("  would require AES-based or hardware-accelerated cryptography.")

    # Warn if parameters are large
    if num_entries > 64:
        print(f"\n  Warning: num_entries={num_entries} may be slow. Consider --entries 64")

    # ==========================================================================
    # Part 1: PIR Benchmark
    # ==========================================================================
    print(f"\n{'═' * 70}")
    print(f"{'PART 1: Index PIR Benchmark':^70}")
    print(f"{'═' * 70}")

    rms_params = Params(num_entries=num_entries, entry_size=entry_size, security_param=security_param)
    plinko_params = PlinkoParams(num_entries=num_entries, entry_size=entry_size, security_param=security_param)

    db_size = num_entries * entry_size
    print(f"\n{'Configuration':─^70}")
    print(f"  Database size:      {format_bytes(db_size):>12}  ({format_count(num_entries)} × {entry_size}B)")
    print(f"  Block size:         {rms_params.block_size:>12}")
    print(f"  Num blocks:         {rms_params.num_blocks:>12}")
    print(f"  Hints (RMS24):      {format_count(rms_params.num_reg_hints):>12}")
    print(f"  Hints (Plinko):     {format_count(plinko_params.num_reg_hints):>12}")

    db = create_database(num_entries, entry_size)

    # RMS24 setup
    rms_server = Server(db, rms_params)
    rms_client = Client(rms_params)
    start = time.perf_counter()
    rms_client.generate_hints(rms_server.stream_database())
    rms_offline_time = time.perf_counter() - start

    # Plinko setup
    plinko_server = PlinkoServer(db, plinko_params)
    plinko_client = PlinkoClient(plinko_params)
    start = time.perf_counter()
    plinko_client.generate_hints(plinko_server.stream_database())
    plinko_offline_time = time.perf_counter() - start

    # Benchmark queries
    test_indices = [(i * num_entries) // num_queries for i in range(num_queries)]
    rms_query_times = []
    plinko_query_times = []

    for idx in test_indices:
        start = time.perf_counter()
        queries = rms_client.query([idx])
        rms_query_times.append(time.perf_counter() - start)
        responses = rms_server.answer(queries)
        rms_client.extract(responses)
        rms_client.replenish_hints()

        start = time.perf_counter()
        queries = plinko_client.query([idx])
        plinko_query_times.append(time.perf_counter() - start)
        responses = plinko_server.answer(queries)
        plinko_client.extract(responses)
        plinko_client.replenish_hints()

    avg_rms_query = sum(rms_query_times) / len(rms_query_times)
    avg_plinko_query = sum(plinko_query_times) / len(plinko_query_times)

    # Benchmark updates
    rms_update_times = []
    plinko_update_times = []
    num_updates = min(10, num_queries)

    for i in range(num_updates):
        idx = (i * num_entries) // num_updates
        new_value = bytes([i % 256] * entry_size)

        updates_rms = rms_server.update_entries({idx: new_value})
        start = time.perf_counter()
        rms_client.update_hints(updates_rms)
        rms_update_times.append(time.perf_counter() - start)

        updates_plinko = plinko_server.update_entries({idx: new_value})
        start = time.perf_counter()
        plinko_client.update_hints(updates_plinko)
        plinko_update_times.append(time.perf_counter() - start)

    avg_rms_update = sum(rms_update_times) / len(rms_update_times)
    avg_plinko_update = sum(plinko_update_times) / len(plinko_update_times)

    # PIR Summary
    print(f"\n{'PIR Results':─^70}")
    print(f"  {'Operation':<25} {'RMS24':>12} {'Plinko':>12} {'Ratio':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Hint generation':<25} {format_time(rms_offline_time):>12} {format_time(plinko_offline_time):>12} {rms_offline_time/plinko_offline_time:>11.2f}x")
    print(f"  {'Query (hint search)':<25} {format_time(avg_rms_query):>12} {format_time(avg_plinko_query):>12} {avg_rms_query/avg_plinko_query if avg_plinko_query > 0 else float('inf'):>11.2f}x")
    print(f"  {'Update (hint update)':<25} {format_time(avg_rms_update):>12} {format_time(avg_plinko_update):>12} {avg_rms_update/avg_plinko_update if avg_plinko_update > 0 else float('inf'):>11.2f}x")

    # ==========================================================================
    # Part 2: KPIR Benchmark
    # ==========================================================================
    print(f"\n{'═' * 70}")
    print(f"{'PART 2: Keyword PIR Benchmark':^70}")
    print(f"{'═' * 70}")

    kw_params = KPIRParams(
        num_items_expected=num_entries,
        key_size=key_size,
        value_size=entry_size,
    )

    print(f"\n{'Configuration':─^70}")
    print(f"  Num items:          {format_count(num_entries):>12}")
    print(f"  Key size:           {key_size:>12} bytes")
    print(f"  Value size:         {entry_size:>12} bytes")
    print(f"  Cuckoo buckets:     {format_count(kw_params.num_buckets):>12}")

    kv_store = create_kv_store(num_entries, key_size, entry_size)

    # KPIR with RMS24 backend
    kpir_rms_server = KPIRServer.create(kv_store, kw_params, rms24, security_param=security_param)
    kpir_rms_client = KPIRClient.create(kw_params, rms24, security_param=security_param)
    start = time.perf_counter()
    kpir_rms_client.generate_hints(kpir_rms_server.stream_database())
    kpir_rms_offline_time = time.perf_counter() - start

    # KPIR with Plinko backend
    kpir_plinko_server = KPIRServer.create(kv_store, kw_params, plinko, security_param=security_param)
    kpir_plinko_client = KPIRClient.create(kw_params, plinko, security_param=security_param)
    start = time.perf_counter()
    kpir_plinko_client.generate_hints(kpir_plinko_server.stream_database())
    kpir_plinko_offline_time = time.perf_counter() - start

    # Benchmark KPIR queries
    test_keys = list(kv_store.keys())[:num_queries]
    kpir_rms_query_times = []
    kpir_plinko_query_times = []

    for key in test_keys:
        start = time.perf_counter()
        queries = kpir_rms_client.query([key])
        kpir_rms_query_times.append(time.perf_counter() - start)
        responses, stash = kpir_rms_server.answer(queries)
        kpir_rms_client.extract(responses, stash)
        kpir_rms_client.replenish_hints()

        start = time.perf_counter()
        queries = kpir_plinko_client.query([key])
        kpir_plinko_query_times.append(time.perf_counter() - start)
        responses, stash = kpir_plinko_server.answer(queries)
        kpir_plinko_client.extract(responses, stash)
        kpir_plinko_client.replenish_hints()

    avg_kpir_rms_query = sum(kpir_rms_query_times) / len(kpir_rms_query_times)
    avg_kpir_plinko_query = sum(kpir_plinko_query_times) / len(kpir_plinko_query_times)

    # Benchmark KPIR updates
    kpir_rms_update_times = []
    kpir_plinko_update_times = []
    update_keys = list(kv_store.keys())[:num_updates]

    for i, key in enumerate(update_keys):
        new_value = f"updated-{i}".encode().ljust(entry_size, b"\x00")

        updates = kpir_rms_server.update({key: new_value})
        start = time.perf_counter()
        kpir_rms_client.update_hints(updates)
        kpir_rms_update_times.append(time.perf_counter() - start)

        updates = kpir_plinko_server.update({key: new_value})
        start = time.perf_counter()
        kpir_plinko_client.update_hints(updates)
        kpir_plinko_update_times.append(time.perf_counter() - start)

    avg_kpir_rms_update = sum(kpir_rms_update_times) / len(kpir_rms_update_times)
    avg_kpir_plinko_update = sum(kpir_plinko_update_times) / len(kpir_plinko_update_times)

    # KPIR Summary
    print(f"\n{'KPIR Results':─^70}")
    print(f"  {'Operation':<25} {'RMS24':>12} {'Plinko':>12} {'Ratio':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Hint generation':<25} {format_time(kpir_rms_offline_time):>12} {format_time(kpir_plinko_offline_time):>12} {kpir_rms_offline_time/kpir_plinko_offline_time:>11.2f}x")
    print(f"  {'Query (hint search)':<25} {format_time(avg_kpir_rms_query):>12} {format_time(avg_kpir_plinko_query):>12} {avg_kpir_rms_query/avg_kpir_plinko_query if avg_kpir_plinko_query > 0 else float('inf'):>11.2f}x")
    print(f"  {'Update (hint update)':<25} {format_time(avg_kpir_rms_update):>12} {format_time(avg_kpir_plinko_update):>12} {avg_kpir_rms_update/avg_kpir_plinko_update if avg_kpir_plinko_update > 0 else float('inf'):>11.2f}x")

    # ==========================================================================
    # Overall Summary
    # ==========================================================================
    print(f"\n{'═' * 70}")
    print(f"{'Overall Summary':^70}")
    print(f"{'═' * 70}")
    print(f"\n  Theoretical advantage: Plinko uses iPRF inverse for O(1) hint")
    print(f"  operations while RMS24 requires O(r) linear scan.")
    print(f"")
    print(f"  Current implementation: Plinko may be slower due to unoptimized")
    print(f"  iPRF (uses SHAKE-256). With optimized PRF (AES-based), the O(1)")
    print(f"  advantage would manifest for large hint counts.")
    print("=" * 70)


# =============================================================================
# Keyword PIR Demo
# =============================================================================


def create_kv_store(num_items: int, key_size: int, value_size: int) -> dict[bytes, bytes]:
    """Create a key-value store with sequential keys and values."""
    return {
        f"key_{i:0{key_size-4}d}".encode()[:key_size]: f"val_{i:0{value_size-4}d}".encode()[:value_size]
        for i in range(num_items)
    }


def run_kpir_demo(num_items: int, key_size: int, value_size: int, security_param: int, num_queries: int):
    """Run the Keyword PIR demo with detailed metrics."""
    print("=" * 70)
    print("Keyword PIR (KPIR) - Demo & Benchmarks")
    print("=" * 70)

    # Setup parameters
    kw_params = KPIRParams(
        num_items_expected=num_items,
        key_size=key_size,
        value_size=value_size,
    )

    pir_params = Params(
        num_entries=kw_params.num_buckets,
        entry_size=kw_params.entry_size,
        security_param=security_param,
    )

    logical_db_size = num_items * (key_size + value_size)
    pir_db_size = kw_params.num_buckets * kw_params.entry_size
    query_size = compute_query_size(pir_params) * kw_params.num_hashes
    response_size = compute_response_size(pir_params) * kw_params.num_hashes

    print(f"\n{'Parameters':─^70}")
    print(f"  Logical DB size:    {format_bytes(logical_db_size):>12}  ({format_count(num_items)} items)")
    print(f"  Key size:           {key_size:>12} bytes")
    print(f"  Value size:         {value_size:>12} bytes")
    print(f"  Cuckoo buckets:     {format_count(kw_params.num_buckets):>12}")
    print(f"  Cuckoo hashes:      {kw_params.num_hashes:>12}")
    print(f"  PIR entry size:     {kw_params.entry_size:>12} bytes")
    print(f"  PIR DB size:        {format_bytes(pir_db_size):>12}")

    # Create key-value store
    print(f"\n{'Key-Value Store Creation':─^70}")
    start = time.perf_counter()
    kv_store = create_kv_store(num_items, key_size, value_size)
    kv_time = time.perf_counter() - start
    print(f"  Time: {format_time(kv_time)}")

    # Build cuckoo table
    print(f"\n{'Cuckoo Table Construction':─^70}")
    start = time.perf_counter()
    server = KPIRServer.create(kv_store, kw_params, rms24, security_param=security_param)
    build_time = time.perf_counter() - start
    print(f"  Time: {format_time(build_time)}")

    # Create client
    client = KPIRClient.create(kw_params, rms24, security_param=security_param)

    # Offline phase
    print(f"\n{'Offline Phase (Hint Generation)':─^70}")
    start = time.perf_counter()
    client.generate_hints(server.stream_database())
    offline_time = time.perf_counter() - start

    print(f"  Time:           {format_time(offline_time):>12}")
    print(f"  Download:       {format_bytes(pir_db_size):>12}  (PIR database)")

    # Online phase
    num_queries = min(num_queries, client.remaining_queries())
    print(f"\n{'Online Phase (' + str(num_queries) + ' keyword queries)':─^70}")

    test_keys = list(kv_store.keys())[:num_queries]

    # Detailed timing breakdown
    query_gen_times = []
    server_answer_times = []
    extract_times = []
    replenish_times = []
    all_correct = True

    for key in test_keys:
        # Query generation (client)
        start = time.perf_counter()
        queries = client.query([key])
        query_gen_times.append(time.perf_counter() - start)

        # Server answer
        start = time.perf_counter()
        responses, stash = server.answer(queries)
        server_answer_times.append(time.perf_counter() - start)

        # Extract result (client)
        start = time.perf_counter()
        [result] = client.extract(responses, stash)
        extract_times.append(time.perf_counter() - start)

        # Replenish hints (client)
        start = time.perf_counter()
        client.replenish_hints()
        replenish_times.append(time.perf_counter() - start)

        expected = kv_store[key]
        if result != expected:
            all_correct = False
            print(f"  ERROR: key {key} returned wrong value!")

    avg_query_gen = sum(query_gen_times) / len(query_gen_times)
    avg_server = sum(server_answer_times) / len(server_answer_times)
    avg_extract = sum(extract_times) / len(extract_times)
    avg_replenish = sum(replenish_times) / len(replenish_times)
    avg_total = avg_query_gen + avg_server + avg_extract + avg_replenish

    print(f"  Correctness:    {'PASS' if all_correct else 'FAIL':>12}")
    print(f"\n  Timing breakdown (avg per query):")
    print(f"    Client query():      {format_time(avg_query_gen):>10}  ({kw_params.num_hashes} PIR queries)")
    print(f"    Server answer():     {format_time(avg_server):>10}")
    print(f"    Client extract():    {format_time(avg_extract):>10}")
    print(f"    Client replenish():  {format_time(avg_replenish):>10}")
    print(f"    ─────────────────────────────────")
    print(f"    Total:               {format_time(avg_total):>10}")

    # Communication costs
    print(f"\n{'Communication Costs':─^70}")
    print(f"  Offline download:   {format_bytes(pir_db_size):>12}")
    print(f"  Query size:         {format_bytes(query_size):>12}  ({kw_params.num_hashes} PIR queries)")
    print(f"  Response size:      {format_bytes(response_size):>12}  ({kw_params.num_hashes} PIR responses)")
    print(f"  Online total:       {format_bytes(query_size + response_size):>12}  per keyword query")

    # Update phase
    print(f"\n{'Update Phase (10 updates)':─^70}")
    update_server_times = []
    update_client_times = []

    update_keys = list(kv_store.keys())[:10]
    for i, key in enumerate(update_keys):
        new_value = f"updated-{i}".encode().ljust(value_size, b"\x00")

        start = time.perf_counter()
        updates = server.update({key: new_value})
        update_server_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        client.update_hints(updates)
        update_client_times.append(time.perf_counter() - start)

    avg_server_update = sum(update_server_times) / len(update_server_times)
    avg_client_update = sum(update_client_times) / len(update_client_times)
    update_msg_size = 4 + kw_params.entry_size  # index (4 bytes) + delta

    print(f"  Timing breakdown (avg per update):")
    print(f"    Server update():         {format_time(avg_server_update):>10}")
    print(f"    Client update_hints():   {format_time(avg_client_update):>10}")
    print(f"  Update message size:   {format_bytes(update_msg_size):>12}  (index + delta)")

    # Summary
    print(f"\n{'Summary':─^70}")
    print(f"  Offline:  {format_time(offline_time)} to download {format_bytes(pir_db_size)}")
    print(f"  Online:   {format_time(avg_total)}/query, {format_bytes(query_size + response_size)} communication")
    print(f"  Capacity: {client.remaining_queries()} queries remaining")
    print("=" * 70)


# =============================================================================
# Keyword PIR with Plinko Backend Demo
# =============================================================================


def run_kpir_plinko_demo(num_items: int, key_size: int, value_size: int, security_param: int, num_queries: int):
    """Run the Keyword PIR demo with Plinko backend."""
    print("=" * 70)
    print("Keyword PIR with Plinko Backend - Demo & Benchmarks")
    print("=" * 70)

    # Note about performance
    print(f"\n{'Note':─^70}")
    print("  This demo uses Plinko as the underlying PIR scheme.")
    print("  Plinko provides O(1) hint operations via invertible PRF.")

    # Warn if parameters are large
    if num_items > 64:
        print(f"\n  Warning: num_items={num_items} may be slow. Consider --entries 64")

    # Setup parameters
    kw_params = KPIRParams(
        num_items_expected=num_items,
        key_size=key_size,
        value_size=value_size,
    )

    pir_params = PlinkoParams(
        num_entries=kw_params.num_buckets,
        entry_size=kw_params.entry_size,
        security_param=security_param,
    )

    logical_db_size = num_items * (key_size + value_size)
    pir_db_size = kw_params.num_buckets * kw_params.entry_size
    query_size = compute_plinko_query_size(pir_params) * kw_params.num_hashes
    response_size = compute_plinko_response_size(pir_params) * kw_params.num_hashes

    print(f"\n{'Parameters':─^70}")
    print(f"  Logical DB size:    {format_bytes(logical_db_size):>12}  ({format_count(num_items)} items)")
    print(f"  Key size:           {key_size:>12} bytes")
    print(f"  Value size:         {value_size:>12} bytes")
    print(f"  Cuckoo buckets:     {format_count(kw_params.num_buckets):>12}")
    print(f"  Cuckoo hashes:      {kw_params.num_hashes:>12}")
    print(f"  PIR entry size:     {kw_params.entry_size:>12} bytes")
    print(f"  PIR DB size:        {format_bytes(pir_db_size):>12}")
    print(f"  PIR backend:        {'Plinko':>12}")

    # Create key-value store
    print(f"\n{'Key-Value Store Creation':─^70}")
    start = time.perf_counter()
    kv_store = create_kv_store(num_items, key_size, value_size)
    kv_time = time.perf_counter() - start
    print(f"  Time: {format_time(kv_time)}")

    # Build cuckoo table
    print(f"\n{'Cuckoo Table Construction':─^70}")
    start = time.perf_counter()
    server = KPIRServer.create(kv_store, kw_params, plinko, security_param=security_param)
    build_time = time.perf_counter() - start
    print(f"  Time: {format_time(build_time)}")

    # Create client
    client = KPIRClient.create(kw_params, plinko, security_param=security_param)

    # Offline phase
    print(f"\n{'Offline Phase (Hint Generation)':─^70}")
    start = time.perf_counter()
    client.generate_hints(server.stream_database())
    offline_time = time.perf_counter() - start

    print(f"  Time:           {format_time(offline_time):>12}")
    print(f"  Download:       {format_bytes(pir_db_size):>12}  (PIR database)")

    # Online phase
    num_queries = min(num_queries, client.remaining_queries())
    print(f"\n{'Online Phase (' + str(num_queries) + ' keyword queries)':─^70}")

    test_keys = list(kv_store.keys())[:num_queries]

    # Detailed timing breakdown
    query_gen_times = []
    server_answer_times = []
    extract_times = []
    replenish_times = []
    all_correct = True

    for key in test_keys:
        # Query generation (client)
        start = time.perf_counter()
        queries = client.query([key])
        query_gen_times.append(time.perf_counter() - start)

        # Server answer
        start = time.perf_counter()
        responses, stash = server.answer(queries)
        server_answer_times.append(time.perf_counter() - start)

        # Extract result (client)
        start = time.perf_counter()
        [result] = client.extract(responses, stash)
        extract_times.append(time.perf_counter() - start)

        # Replenish hints (client)
        start = time.perf_counter()
        client.replenish_hints()
        replenish_times.append(time.perf_counter() - start)

        expected = kv_store[key]
        if result != expected:
            all_correct = False
            print(f"  ERROR: key {key} returned wrong value!")

    avg_query_gen = sum(query_gen_times) / len(query_gen_times)
    avg_server = sum(server_answer_times) / len(server_answer_times)
    avg_extract = sum(extract_times) / len(extract_times)
    avg_replenish = sum(replenish_times) / len(replenish_times)
    avg_total = avg_query_gen + avg_server + avg_extract + avg_replenish

    print(f"  Correctness:    {'PASS' if all_correct else 'FAIL':>12}")
    print(f"\n  Timing breakdown (avg per query):")
    print(f"    Client query():      {format_time(avg_query_gen):>10}  ({kw_params.num_hashes} PIR queries, O(1) hint search)")
    print(f"    Server answer():     {format_time(avg_server):>10}")
    print(f"    Client extract():    {format_time(avg_extract):>10}")
    print(f"    Client replenish():  {format_time(avg_replenish):>10}")
    print(f"    ─────────────────────────────────")
    print(f"    Total:               {format_time(avg_total):>10}")

    # Communication costs
    print(f"\n{'Communication Costs':─^70}")
    print(f"  Offline download:   {format_bytes(pir_db_size):>12}")
    print(f"  Query size:         {format_bytes(query_size):>12}  ({kw_params.num_hashes} PIR queries)")
    print(f"  Response size:      {format_bytes(response_size):>12}  ({kw_params.num_hashes} PIR responses)")
    print(f"  Online total:       {format_bytes(query_size + response_size):>12}  per keyword query")

    # Update phase
    print(f"\n{'Update Phase (10 updates)':─^70}")
    update_server_times = []
    update_client_times = []

    update_keys = list(kv_store.keys())[:10]
    for i, key in enumerate(update_keys):
        new_value = f"updated-{i}".encode().ljust(value_size, b"\x00")

        start = time.perf_counter()
        updates = server.update({key: new_value})
        update_server_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        client.update_hints(updates)
        update_client_times.append(time.perf_counter() - start)

    avg_server_update = sum(update_server_times) / len(update_server_times)
    avg_client_update = sum(update_client_times) / len(update_client_times)
    update_msg_size = 4 + kw_params.entry_size  # index (4 bytes) + delta

    print(f"  Timing breakdown (avg per update):")
    print(f"    Server update():         {format_time(avg_server_update):>10}")
    print(f"    Client update_hints():   {format_time(avg_client_update):>10}  (O(1) via iPRF)")
    print(f"  Update message size:   {format_bytes(update_msg_size):>12}  (index + delta)")

    # Summary
    print(f"\n{'Summary':─^70}")
    print(f"  Offline:  {format_time(offline_time)} to download {format_bytes(pir_db_size)}")
    print(f"  Online:   {format_time(avg_total)}/query, {format_bytes(query_size + response_size)} communication")
    print(f"  Capacity: {client.remaining_queries()} queries remaining")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


# Default parameters
DEFAULT_NUM_ENTRIES = 64
DEFAULT_ENTRY_SIZE = 32
DEFAULT_KEY_SIZE = 32
DEFAULT_SECURITY_PARAM = 128
DEFAULT_NUM_QUERIES = 10


def main():
    parser = argparse.ArgumentParser(
        description="RMS24, Plinko PIR & KPIR Demo with detailed benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo.py --rms24            # RMS24 PIR demo
  python3 demo.py --plinko           # Plinko PIR demo
  python3 demo.py --kpir-rms24       # Keyword PIR demo (RMS24 backend)
  python3 demo.py --kpir-plinko      # Keyword PIR demo (Plinko backend)
  python3 demo.py --benchmark        # PIR & KPIR benchmark (RMS24 vs Plinko)
  python3 demo.py --entries 128      # Custom entry count
        """,
    )
    parser.add_argument("--rms24", action="store_true", help="Run RMS24 PIR demo")
    parser.add_argument("--plinko", action="store_true", help="Run Plinko PIR demo")
    parser.add_argument("--benchmark", action="store_true", help="Run PIR & KPIR benchmark (RMS24 vs Plinko)")
    parser.add_argument("--kpir-rms24", action="store_true", help="Run Keyword PIR demo (RMS24 backend)")
    parser.add_argument("--kpir-plinko", action="store_true", help="Run Keyword PIR demo (Plinko backend)")
    parser.add_argument("--entries", type=int, default=None, help="Number of entries/items")
    parser.add_argument("--entry-size", type=int, default=DEFAULT_ENTRY_SIZE, help=f"Entry/value size (default: {DEFAULT_ENTRY_SIZE})")
    parser.add_argument("--key-size", type=int, default=DEFAULT_KEY_SIZE, help=f"Key size for KPIR (default: {DEFAULT_KEY_SIZE})")
    parser.add_argument("--security", type=int, default=DEFAULT_SECURITY_PARAM, help=f"Security parameter (default: {DEFAULT_SECURITY_PARAM})")
    parser.add_argument("--queries", type=int, default=DEFAULT_NUM_QUERIES, help=f"Number of queries (default: {DEFAULT_NUM_QUERIES})")
    args = parser.parse_args()

    # Determine number of entries
    num_entries = args.entries if args.entries else DEFAULT_NUM_ENTRIES

    # Run demos
    if args.benchmark:
        run_benchmark(num_entries, args.entry_size, args.key_size, args.security, args.queries)
    elif args.plinko:
        run_plinko_demo(num_entries, args.entry_size, args.security, args.queries)
    elif args.kpir_rms24:
        run_kpir_demo(num_entries, args.key_size, args.entry_size, args.security, args.queries)
    elif args.kpir_plinko:
        run_kpir_plinko_demo(num_entries, args.key_size, args.entry_size, args.security, args.queries)
    elif args.rms24:
        run_rms24_demo(num_entries, args.entry_size, args.security, args.queries)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
