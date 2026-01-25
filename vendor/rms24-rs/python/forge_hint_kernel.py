"""
RMS24 Hint Generation Kernel for Forge Optimization.

Drop this file into Forge to generate optimized CUDA/Triton kernels.

Operation: For each hint, gather database entries at precomputed indices
and XOR them together to produce a parity value.

Memory pattern: Scatter-gather with segment reduction (XOR).
"""

import torch
import torch.nn as nn


class HintGenKernel(nn.Module):
    """
    Batched hint generation kernel.
    
    Input shapes (all on GPU):
    - entries: [num_entries, 5] int64 - Database as 5 x 8-byte chunks
    - padded_indices: [num_hints, max_subset_size] int64 - Entry indices per hint
    - valid_mask: [num_hints, max_subset_size] bool - Which indices are valid
    
    Output:
    - parities: [num_hints, 5] int64 - XOR parity per hint
    
    The core operation is:
        parities[h] = XOR over all entries[padded_indices[h, i]] where valid_mask[h, i]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        entries: torch.Tensor,        # [num_entries, 5] int64
        padded_indices: torch.Tensor, # [num_hints, max_subset_size] int64
        valid_mask: torch.Tensor,     # [num_hints, max_subset_size] bool
    ) -> torch.Tensor:
        # Gather: [num_hints, max_subset_size, 5]
        gathered = entries[padded_indices]
        
        # Zero out invalid entries (XOR identity)
        gathered = gathered * valid_mask.unsqueeze(-1).to(gathered.dtype)
        
        # XOR reduction across subset dimension
        # Start with first element, XOR rest
        parity = gathered[:, 0, :].clone()
        for i in range(1, gathered.shape[1]):
            parity = torch.bitwise_xor(parity, gathered[:, i, :])
        
        return parity


def create_test_data(
    num_entries: int = 262144,  # ~10MB database
    num_hints: int = 100,
    max_subset_size: int = 512,
    device: str = "cuda",
):
    """Create synthetic test data for benchmarking."""
    dev = torch.device(device)
    
    entries = torch.randint(
        0, 2**60, (num_entries, 5), 
        dtype=torch.int64, device=dev
    )
    
    # Random indices, padded
    padded_indices = torch.randint(
        0, num_entries, (num_hints, max_subset_size),
        dtype=torch.int64, device=dev
    )
    
    # Random valid mask (80% valid on average)
    valid_mask = torch.rand(num_hints, max_subset_size, device=dev) < 0.8
    
    return entries, padded_indices, valid_mask


def benchmark(warmup: int = 10, iters: int = 100):
    """Benchmark the kernel."""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    entries, padded_indices, valid_mask = create_test_data(device=device)
    kernel = HintGenKernel().to(device)
    
    print(f"Entries: {entries.shape}")
    print(f"Padded indices: {padded_indices.shape}")
    print(f"Valid mask: {valid_mask.shape}")
    
    # Warmup
    for _ in range(warmup):
        _ = kernel(entries, padded_indices, valid_mask)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = kernel(entries, padded_indices, valid_mask)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    ms_per_iter = elapsed / iters * 1000
    
    print(f"\nLatency: {ms_per_iter:.3f} ms")
    print(f"Throughput: {padded_indices.shape[0] / (ms_per_iter/1000):.0f} hints/sec")
    
    return ms_per_iter


if __name__ == "__main__":
    benchmark()
