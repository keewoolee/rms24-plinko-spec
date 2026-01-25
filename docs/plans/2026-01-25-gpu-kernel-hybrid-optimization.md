# GPU Kernel Hybrid Optimization Design

## Overview

Optimize RMS24 GPU hint generation from 98 hints/sec to target ~100K hints/sec using:
1. **Precomputed subset lists** - CPU generates sorted block arrays per hint
2. **Warp-level parallelism** - 32 threads cooperatively process blocks
3. **Vectorized XOR reduction** - Use warp shuffles for parallel accumulation

## Current Bottleneck Analysis

Current kernel iterates all 42K blocks per hint:
```cuda
for (uint64_t block = 0; block < params.num_blocks; block++) {
    uint32_t select_value = chacha_prf_select(key, hint_idx, block);
    // ... ~42K iterations with ChaCha12 per hint
}
```

**Problem**: Each hint does 42K ChaCha12 calls (12 rounds each) = 504K quarter-rounds per hint.

## Hybrid Approach

### Phase 1 (CPU): Precompute Compressed Subset Lists

Instead of bitmasks (42K bits = 5.25KB per hint), store sorted block indices:
- Regular hints: ~21K blocks in subset → ~84KB per hint as u32 array
- Also precompute offset for each block (avoids PRF call on GPU)

**New data structure:**
```rust
struct HintSubset {
    blocks: Vec<u32>,      // Sorted block indices in subset
    offsets: Vec<u32>,     // Precomputed offset for each block
    extra_block: u32,      // For regular hints
    extra_offset: u32,
}
```

### Phase 2 (GPU): Warp-Cooperative Processing

Each warp (32 threads) processes one hint:
1. Thread 0 loads hint metadata
2. All 32 threads divide subset blocks: `blocks[lane], blocks[lane+32], ...`
3. Each thread XORs entries for its assigned blocks into local parity
4. Warp-level reduction via `__shfl_xor_sync` to combine parities

```cuda
__global__ void rms24_hint_gen_warp_kernel(
    const Rms24Params params,
    const uint32_t* subset_blocks,    // Flattened: hint 0 blocks, hint 1 blocks, ...
    const uint32_t* subset_offsets,   // Matching offsets
    const uint32_t* subset_starts,    // Start index for each hint
    const uint32_t* subset_sizes,     // Number of blocks per hint
    const uint8_t* entries,
    HintOutput* output
) {
    uint32_t hint_idx = blockIdx.x;
    uint32_t lane = threadIdx.x & 31;  // Warp lane (0-31)
    
    uint32_t start = subset_starts[hint_idx];
    uint32_t size = subset_sizes[hint_idx];
    
    // Each lane accumulates its share of blocks
    uint64_t parity[5] = {0};
    for (uint32_t i = lane; i < size; i += WARP_SIZE) {
        uint32_t block = subset_blocks[start + i];
        uint32_t offset = subset_offsets[start + i];
        uint64_t entry_idx = block * params.block_size + offset;
        
        const uint64_t* entry = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
        parity[0] ^= entry[0];
        parity[1] ^= entry[1];
        parity[2] ^= entry[2];
        parity[3] ^= entry[3];
        parity[4] ^= entry[4];
    }
    
    // Warp reduction: XOR across all 32 lanes
    for (int offset = 16; offset > 0; offset /= 2) {
        parity[0] ^= __shfl_xor_sync(0xFFFFFFFF, parity[0], offset);
        parity[1] ^= __shfl_xor_sync(0xFFFFFFFF, parity[1], offset);
        parity[2] ^= __shfl_xor_sync(0xFFFFFFFF, parity[2], offset);
        parity[3] ^= __shfl_xor_sync(0xFFFFFFFF, parity[3], offset);
        parity[4] ^= __shfl_xor_sync(0xFFFFFFFF, parity[4], offset);
    }
    
    // Lane 0 writes result
    if (lane == 0) {
        uint64_t* out = (uint64_t*)output[hint_idx].parity;
        out[0] = parity[0]; out[1] = parity[1]; out[2] = parity[2];
        out[3] = parity[3]; out[4] = parity[4]; out[5] = 0;
    }
}
```

### Memory Layout

**Flattened subset arrays:**
```
subset_blocks:  [hint0_block0, hint0_block1, ..., hint1_block0, hint1_block1, ...]
subset_offsets: [hint0_off0,   hint0_off1,   ..., hint1_off0,   hint1_off1,   ...]
subset_starts:  [0, 21000, 42000, ...]  // Cumulative sizes
subset_sizes:   [21000, 21000, ...]     // Blocks per hint
```

**GPU memory estimate for 1K hints:**
- 1K hints × 21K blocks × 8 bytes (block + offset) = ~168MB
- Fits easily in H200's 141GB HBM3e

## Implementation Plan

### Task 1: Add `HintSubset` struct and CPU precomputation

File: `src/hints.rs`
- Add `HintSubset` struct
- Add `SubsetBuilder` to generate subsets during Phase 1

### Task 2: Update `client.rs` for subset generation

File: `src/client.rs`  
- Modify `generate_hints()` to populate `HintSubset` per hint
- Precompute offsets alongside block selection

### Task 3: Create warp-optimized CUDA kernel

File: `cuda/hint_kernel.cu`
- Add `rms24_hint_gen_warp_kernel` 
- Keep old kernel for comparison
- Handle regular vs backup hints separately

### Task 4: Update GPU bindings

File: `src/gpu.rs`
- Add `SubsetData` struct for flattened arrays
- Add `generate_hints_warp()` method
- Update `hints_to_subset_data()` conversion

### Task 5: Update benchmark binary

File: `src/bin/bench_gpu_hints.rs`
- Use new warp kernel
- Add comparison mode (--kernel old|warp)

### Task 6: Verify correctness

- Add test comparing CPU vs GPU parities
- Verify with deterministic PRF key

## Expected Performance

| Metric | Current | Target |
|--------|---------|--------|
| GPU kernel time | 10.2s/1K hints | <100ms/1K hints |
| Throughput | 98 hints/sec | ~100K hints/sec |
| Memory bandwidth | Low (PRF-bound) | High (entry-bound) |

**Key improvements:**
1. No ChaCha12 on GPU (precomputed)
2. Coalesced memory access (sorted blocks)
3. Warp-level parallelism (32x parallel XOR)
4. Reduced warp divergence (uniform block counts)
