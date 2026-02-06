# RMS24 & Plinko: A Runnable Specification

A readable, executable specification of single-server PIR schemes with Keyword PIR support.

## Purpose

This repository serves as:

- **Runnable Spec**: A readable, executable specification prioritizing clarity over performance
- **Anchor for AI-assisted development**: A reference implementation for LLM-assisted porting to other languages or optimization

> **Note**: `demo.py` and `tests/` are AI-generated and have not been reviewed by humans. The core implementation in `pir/` has been reviewed.

## Features

- **Single-server PIR** with client-dependent preprocessing
  - **RMS24**: Based on [RMS24](https://eprint.iacr.org/2023/1072)
  - **Plinko**: Based on [Plinko](https://eprint.iacr.org/2024/318)
    - Hint updates cost Õ(1) via invertible PRF (iPRF), but with significant concrete overhead
  - O(√num_entries) online communication and server computation
  - Supports O(√num_entries) queries before re-running offline phase

- **Keyword PIR** via cuckoo hashing (works with both RMS24 and Plinko)
  - Based on [ALPRSSY21](https://eprint.iacr.org/2019/1483)

- **Updatability** without re-running offline phase

- **Batch operations** for queries and updates

- **Configurable tradeoffs** via `block_size` and `num_backup_hints`

- **Vitalik's optimization** for reduced query communication
  - Original: send offsets for both subsets (c offsets total)
  - Optimized: share offsets between subsets (c/2 offsets total)
  - 50% reduction in query size

## Quick Start

```bash
# Run RMS24 PIR demo
python3 demo.py --rms24

# Run Plinko PIR demo
python3 demo.py --plinko

# Run Keyword PIR demo
python3 demo.py --kpir-rms24
```

## Usage

### Index PIR

```python
from pir.rms24 import Params, Client, Server  # or pir.plinko

database = [bytes(32) for _ in range(1000)]  # 1000 entries, 32 bytes each

params = Params(num_entries=len(database), entry_size=32)
server = Server(database, params)
client = Client(params)

client.generate_hints(server.stream_database())  # Offline

queries = client.query([42, 100, 999])           # Online (batch)
responses = server.answer(queries)
results = client.extract(responses)
client.replenish_hints()

updates = server.update_entries({0: bytes(32), 5: bytes(32)})  # Update (batch)
client.update_hints(updates)
```

### Keyword PIR

```python
from pir.rms24 import Client, Server  # or pir.plinko
from pir.keyword_pir import KPIRParams, KPIRClient, KPIRServer

kv_store = {b"key1".ljust(32): b"value1".ljust(32)}  # 32-byte keys and values

kw_params = KPIRParams(num_items_expected=len(kv_store), key_size=32, value_size=32)
server = KPIRServer.create(kv_store, kw_params, Server)
client = KPIRClient.create(kw_params, Client)

client.generate_hints(server.stream_database())  # Offline

key1, key2 = b"key-0".ljust(32), b"key-1".ljust(32)
queries = client.query([key1, key2])             # Online (batch)
responses, stash = server.answer(queries)
results = client.extract(responses, stash)
client.replenish_hints()

updates = server.update({b"key1".ljust(32): b"new_val".ljust(32)})  # Update
client.update_hints(updates)
```

## Benchmarks

**Setup**: 64 entries × 32 bytes, security_param=128

*Note: With only 64 entries, this benchmark does not demonstrate Plinko's asymptotic Õ(1) advantage. However, it illustrates the significant concrete overhead of the iPRF construction.*

### Index PIR

| Operation | RMS24 | Plinko | Communication |
|-----------|-------|--------|---------------|
| Hint generation | 57ms | 33s | 2 KiB (download) |
| Query (hint search) | 41μs | 614ms | 17 B |
| Server answer | 12μs | 12μs | 64 B |
| Update (hint update) | 2ms | 602ms | 36 B |

### Keyword PIR

| Operation | RMS24 | Plinko | Communication |
|-----------|-------|--------|---------------|
| Hint generation | 198ms | 99s | 12 KiB (download) |
| Query (hint search) | 114μs | 1.2s | 52 B |
| Server answer | 53μs | 53μs | 256 B |
| Update (hint update) | 4ms | 590ms | 68 B |

Run benchmarks with:

```bash
python3 demo.py --rms24         # RMS24 PIR
python3 demo.py --plinko        # Plinko PIR
python3 demo.py --kpir-rms24    # Keyword PIR (RMS24 backend)
python3 demo.py --kpir-plinko   # Keyword PIR (Plinko backend)
python3 demo.py --benchmark     # Compare RMS24 vs Plinko
```

Use `--entries N` to adjust database size (default: 64 entries).

## Project Structure

```
rms24/
├── demo.py                 # Benchmarks and usage examples
├── pir/
│   ├── protocols.py        # PIR client/server interfaces
│   ├── rms24/              # RMS24 scheme
│   │   ├── params.py       # Parameter configuration
│   │   ├── client.py       # Client (hint generation, queries)
│   │   ├── server.py       # Server (database, responses)
│   │   ├── messages.py     # Query/Response types
│   │   └── utils.py        # PRF and XOR utilities
│   ├── plinko/             # Plinko scheme
│   │   ├── params.py       # Parameter configuration
│   │   ├── client.py       # Client (hint generation, queries)
│   │   ├── server.py       # Server (database, responses)
│   │   ├── messages.py     # Query/Response types
│   │   ├── utils.py        # PRF, PRG, and XOR utilities
│   │   ├── iprf.py         # Invertible PRF (PRP + PMNS)
│   │   ├── prp.py          # Small-domain PRP (MR14)
│   │   └── pmns.py         # Pseudorandom multinomial sampler
│   ├── primitives/         # Primitive API definitions
│   │   ├── prp.py          # PRP interface
│   │   ├── pmns.py         # PMNS interface
│   │   └── iprf.py         # iPRF interface
│   └── keyword_pir/        # Keyword PIR layer
│       ├── params.py       # KPIR parameters
│       ├── client.py       # KPIR client
│       ├── server.py       # KPIR server
│       └── cuckoo.py       # Cuckoo hashing
└── tests/                  # Test suite
```

## Implementation Notes

This is an unoptimized reference implementation prioritizing clarity over performance.

- `# OPT:` comments mark parts that could be optimized
- `# SEC:` comments mark security-related notes

## References

- **RMS24**: Ling Ren, Muhammad Haris Mughees, I Sun. [Simple and Practical Amortized Sublinear Private Information Retrieval](https://eprint.iacr.org/2023/1072). CCS 2024.
- **Plinko**: Alexander Hoover, Sarvar Patel, Giuseppe Persiano, Kevin Yeo. [Plinko: Single-Server PIR with Efficient Updates via Invertible PRFs](https://eprint.iacr.org/2024/318). Eurocrypt 2025.
- **S3PIR**: [github.com/renling/S3PIR](https://github.com/renling/S3PIR) - Accompanying C++ PoC for RMS24 with optimizations (batched AES-PRF).
- **Keyword PIR**: Asra Ali, Tancrède Lepoint, Sarvar Patel, Mariana Raykova, Phillipp Schoppmann, Karn Seth, Kevin Yeo. [Communication-Computation Trade-offs in PIR](https://eprint.iacr.org/2019/1483). USENIX Security 2021.

## License

Apache 2.0
