# Plinko Single-Server PIR

Implementation of the Plinko scheme from ["Plinko: Single-Server PIR with Efficient Updates via Invertible PRFs"](https://eprint.iacr.org/2024/318).

## Overview

Plinko achieves Õ(1) hint operations using Invertible PRFs (iPRFs):

- **Offline**: Client streams database to build hints
- **Online**: Each query uses O(√num_entries) communication and server computation
- **Hint search**: Õ(1) via iPRF inverse (vs O(num_hints) linear scan in RMS24)
- **Hint update**: Õ(1) via iPRF inverse (vs O(num_hints) linear scan in RMS24)
- **Capacity**: Supports O(√num_entries) queries before re-running offline phase

**Trade-off**: The iPRF construction has significant concrete overhead. See benchmarks in main README.

## Parameters

```python
Params(
    num_entries=1000,        # Database size
    entry_size=32,           # Bytes per entry
    security_param=128,      # Failure probability ≈ 2^{-security_param}
    block_size=None,         # Default: ⌈√num_entries⌉
    num_backup_hints=None,   # Default: security_param × block_size
)
```

**Derived values:**
- `num_blocks` = ⌈num_entries / block_size⌉, rounded to even
- `num_reg_hints` = security_param × block_size
- `num_backup_hints` = number of queries supported per offline phase

## Files

| File | Description |
|------|-------------|
| `params.py` | Parameter configuration and validation |
| `client.py` | Hint generation, query preparation, result extraction |
| `server.py` | Database storage, query answering |
| `messages.py` | Query and Response dataclasses |
| `utils.py` | SHAKE-256 based PRF/PRG and XOR utilities |
| `iprf.py` | Invertible PRF (iPRF) composition |
| `prp.py` | Small-domain PRP via Sometimes-Recurse Shuffle (MR14) |
| `pmns.py` | Pseudorandom Multinomial Sampler |

## Invertible PRF (iPRF)

The key innovation enabling Õ(1) operations. For each block α, an iPRF maps hint indices to offsets:

```
F_α: [num_total_hints] → [block_size]
```

**Composition** (Theorem 4.4 in Plinko paper):
```
F(x) = S(P(x))                           // Forward: PRP then PMNS
F⁻¹(y) = {P⁻¹(z) : z ∈ S⁻¹(y)}          // Inverse: PMNS⁻¹ then PRP⁻¹
```

Where:
- **PRP**: Fully-secure small-domain pseudorandom permutation (Sometimes-Recurse Shuffle from MR14)
- **PMNS**: Pseudorandom Multinomial Sampler

**Note**: Security requires a fully-secure small-domain PRP. We use MR14's Sometimes-Recurse Shuffle, which appears to be the only known construction.

## Query Cache

In Plinko, the query cache is a **required mechanism** (not optional):

- Extra entries of promoted hints cannot be tracked by iPRF inverse
- Without the cache, updating extra entries would require O(num_hints) scan, nullifying the Õ(1) benefit
- The cache stores `index → (answer, promoted_hint_idx)` for:
  1. **Update by extra**: Õ(1) lookup of hints whose extra entry changed
  2. **Repeated queries**: Return cached answer, query a decoy index to maintain privacy

## Hint Structure

Hints are stored in `HintState` using parallel arrays (index = hint_id):

**Regular hints** (indices 0 to num_reg_hints-1):
- `cutoffs[i]`: Median value splitting blocks into two halves (0 = consumed)
- `parities[i]`: XOR of entries in the hint's subset
- `flips[i]`: Selection direction (inverted after promotion from backup)

**Backup hints** (indices num_reg_hints to num_total_hints-1):
- `cutoffs[i]`: Same as regular hints
- `extras[i - num_reg_hints]`: Extra entry index (set during promotion)
- `parities[i]`: XOR of entries from blocks where `PRF < cutoff`
- `backup_parities_high[i - num_reg_hints]`: XOR of entries from blocks where `PRF >= cutoff`

## Protocol

### Offline Phase

```
Client                                  Server
  |                                       |
  |  <------- blocks[0..num_blocks-1] --- |
  |                                       |
  [generate_hints: build hints using      |
   iPRF to map hints to offsets]          |
```

### Online Phase (Query)

```
Client                                  Server
  |                                       |
  [Õ(1) hint search via iPRF inverse]     |
  [build query hiding index in subset]    |
  |                                       |
  |  -------- Query(mask, offsets) -----> |
  |                                       |
  |           [compute parity_0, parity_1]|
  |                                       |
  |  <------- Response(p0, p1) ---------- |
  |                                       |
  [extract: result = parity XOR hint.parity]
  [replenish + update query cache]        |
```

## Security

The query hides which entry is accessed:
1. Both subsets in Query have exactly `num_blocks/2` blocks
2. The mask randomly assigns the real subset to position 0 or 1
3. Offsets are shared between subsets, revealing no information
4. Server sees two equal-sized subsets and cannot distinguish which is real

The PRF/iPRF keys are client-secret. All hint data must remain private.

## Communication Costs

| Message | Size |
|---------|------|
| Query | ⌈num_blocks/8⌉ bytes (mask) + num_blocks/2 offsets |
| Response | 2 × entry_size bytes |

Communication is identical to RMS24. The Õ(1) advantage is in client computation, not communication.

## Example

```python
from pir.plinko import Params, Client, Server

# 1000 entries × 32 bytes
params = Params(num_entries=1000, entry_size=32)
# block_size=32, num_blocks=32, num_reg_hints=4096

database = [bytes(32) for _ in range(1000)]
server = Server(database, params)
client = Client(params)

client.generate_hints(server.stream_database())  # Offline

queries = client.query([42, 100, 999])           # Online (Õ(1) hint search)
responses = server.answer(queries)
results = client.extract(responses)
client.replenish_hints()

print(f"Remaining queries: {client.remaining_queries()}")
```

## Updates

Database updates use Õ(1) hint updates via iPRF inverse:

```python
# Server updates entries
updates = server.update_entries({0: new_value, 5: another_value})

# Client updates affected hints in Õ(1) time
client.update_hints(updates)
```

The `delta` in each update is `old_value XOR new_value`, allowing efficient hint adjustment.
