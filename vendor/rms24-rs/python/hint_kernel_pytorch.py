"""
RMS24 Hint Generation Kernel in PyTorch.

This is a reference implementation for optimization via Forge.
The kernel computes XOR parities over indexed database entries.

Entry format: 40 bytes = 5 x int64 (little-endian)
"""

import torch
import torch.nn as nn


class HintGenKernel(nn.Module):
    """
    Compute hint parities via indexed gather + XOR reduction.
    
    For each hint, gathers entries at specified indices and XORs them together.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        entries: torch.Tensor,      # [num_entries, 5] int64 - database
        subset_blocks: torch.Tensor, # [total_blocks] int32 - flattened block indices
        subset_offsets: torch.Tensor, # [total_blocks] int32 - offsets within blocks
        subset_starts: torch.Tensor,  # [num_hints] int32 - start index per hint
        subset_sizes: torch.Tensor,   # [num_hints] int32 - blocks per hint
        block_size: int,
    ) -> torch.Tensor:
        """
        Returns: [num_hints, 5] int64 - XOR parities
        """
        num_hints = subset_starts.shape[0]
        device = entries.device
        
        # Output parities
        parities = torch.zeros(num_hints, 5, dtype=torch.int64, device=device)
        
        # Process each hint
        for hint_idx in range(num_hints):
            start = subset_starts[hint_idx].item()
            size = subset_sizes[hint_idx].item()
            
            if size == 0:
                continue
            
            # Get block indices and offsets for this hint
            blocks = subset_blocks[start:start+size]
            offsets = subset_offsets[start:start+size]
            
            # Compute entry indices (clamp to valid range)
            entry_indices = blocks.long() * block_size + offsets.long()
            entry_indices = entry_indices.clamp(0, entries.shape[0] - 1)
            
            # Gather entries
            selected = entries[entry_indices]  # [size, 5]
            
            # XOR reduction
            parity = selected[0].clone()
            for i in range(1, size):
                parity ^= selected[i]
            
            parities[hint_idx] = parity
        
        return parities


class HintGenKernelBatched(nn.Module):
    """
    Batched version using padding for Forge optimization.
    
    All hints padded to max_subset_size for vectorization.
    """
    
    def __init__(self, max_subset_size: int = 2048):
        super().__init__()
        self.max_subset_size = max_subset_size
    
    def forward(
        self,
        entries: torch.Tensor,        # [num_entries, 5] int64
        padded_indices: torch.Tensor, # [num_hints, max_subset_size] int64
        valid_mask: torch.Tensor,     # [num_hints, max_subset_size] bool
    ) -> torch.Tensor:
        """
        Batched gather + masked XOR reduction.
        
        Returns: [num_hints, 5] int64
        """
        # Gather all entries at once: [num_hints, max_subset_size, 5]
        gathered = entries[padded_indices]
        
        # Mask invalid entries (set to 0 so XOR is identity)
        gathered = gathered * valid_mask.unsqueeze(-1)
        
        # XOR reduction across subset dimension
        # Note: torch doesn't have bitwise_xor reduce, so we use a loop
        # Forge should optimize this into a proper reduction
        parity = gathered[:, 0, :]
        for i in range(1, gathered.shape[1]):
            parity = parity ^ gathered[:, i, :]
        
        return parity


class HintGenKernelVectorized(nn.Module):
    """
    Fully vectorized version for maximum Forge optimization potential.
    
    Uses scatter_add trick: XOR is equivalent to sum mod 2 for each bit,
    but we can't easily express this. Instead, we'll use a different approach:
    pack hints into a single large gather + use segment reduction.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        entries: torch.Tensor,        # [num_entries, 5] int64
        entry_indices: torch.Tensor,  # [total_blocks] int64 - precomputed
        hint_ids: torch.Tensor,       # [total_blocks] int64 - which hint each block belongs to
        num_hints: int,
    ) -> torch.Tensor:
        """
        Segment XOR reduction using scatter.
        
        This is the most Forge-friendly formulation.
        """
        # Gather all relevant entries
        gathered = entries[entry_indices]  # [total_blocks, 5]
        
        # Initialize output
        parities = torch.zeros(num_hints, 5, dtype=torch.int64, device=entries.device)
        
        # Segment XOR via loop (Forge should fuse this)
        # Group by hint_id and XOR
        for i in range(gathered.shape[0]):
            hint_id = hint_ids[i]
            parities[hint_id] ^= gathered[i]
        
        return parities


def prepare_batched_inputs(
    subset_blocks: torch.Tensor,
    subset_offsets: torch.Tensor, 
    subset_starts: torch.Tensor,
    subset_sizes: torch.Tensor,
    block_size: int,
    max_subset_size: int,
    num_entries: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert variable-length subsets to padded format.
    """
    num_hints = subset_starts.shape[0]
    
    padded_indices = torch.zeros(num_hints, max_subset_size, dtype=torch.int64, device=device)
    valid_mask = torch.zeros(num_hints, max_subset_size, dtype=torch.bool, device=device)
    
    for hint_idx in range(num_hints):
        start = subset_starts[hint_idx].item()
        size = min(subset_sizes[hint_idx].item(), max_subset_size)
        
        blocks = subset_blocks[start:start+size].long()
        offsets = subset_offsets[start:start+size].long()
        entry_indices = blocks * block_size + offsets
        entry_indices = entry_indices.clamp(0, num_entries - 1)
        
        padded_indices[hint_idx, :size] = entry_indices
        valid_mask[hint_idx, :size] = True
    
    return padded_indices, valid_mask


def benchmark_kernel(kernel: nn.Module, *args, warmup: int = 10, iters: int = 100):
    """Benchmark a kernel."""
    import time
    
    # Warmup
    for _ in range(warmup):
        _ = kernel(*args)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iters):
        _ = kernel(*args)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iters * 1000  # ms


if __name__ == "__main__":
    # Test with synthetic data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Parameters matching 100MB synthetic DB
    num_entries = 2621440  # 100MB / 40 bytes
    block_size = 1620
    num_blocks = (num_entries + block_size - 1) // block_size
    num_hints = 1000
    blocks_per_hint = num_blocks // 2  # ~half blocks per hint
    
    print(f"Entries: {num_entries:,}")
    print(f"Block size: {block_size}")
    print(f"Num blocks: {num_blocks}")
    print(f"Hints: {num_hints}")
    print(f"Blocks per hint: {blocks_per_hint}")
    
    # Create synthetic data
    entries = torch.randint(0, 2**60, (num_entries, 5), dtype=torch.int64, device=device)
    
    # Create subset data (each hint has ~half the blocks)
    total_blocks = num_hints * blocks_per_hint
    subset_blocks = torch.randint(0, num_blocks, (total_blocks,), dtype=torch.int32, device=device)
    subset_offsets = torch.randint(0, block_size, (total_blocks,), dtype=torch.int32, device=device)
    subset_starts = torch.arange(0, total_blocks, blocks_per_hint, dtype=torch.int32, device=device)
    subset_sizes = torch.full((num_hints,), blocks_per_hint, dtype=torch.int32, device=device)
    
    print(f"\nTotal blocks in subsets: {total_blocks:,}")
    print(f"Subset data size: {total_blocks * 8 / 1e6:.2f} MB")
    
    # Test basic kernel
    kernel = HintGenKernel().to(device)
    print("\nRunning basic kernel...")
    
    if device.type == "cuda":
        ms = benchmark_kernel(kernel, entries, subset_blocks, subset_offsets, 
                             subset_starts, subset_sizes, block_size,
                             warmup=2, iters=5)
        print(f"Basic kernel: {ms:.2f} ms ({num_hints / (ms/1000):.0f} hints/sec)")
    else:
        # CPU test - just run once
        parities = kernel(entries, subset_blocks, subset_offsets, 
                         subset_starts, subset_sizes, block_size)
        print(f"Output shape: {parities.shape}")
        print(f"Non-zero parities: {(parities != 0).any(dim=1).sum().item()}")
