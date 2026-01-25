//! GPU-accelerated hint generation using CUDA.

#[cfg(feature = "cuda")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::hints::SubsetData;
use crate::params::ENTRY_SIZE;

/// RMS24 parameters for GPU kernel
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Rms24Params {
    pub num_entries: u64,
    pub block_size: u64,
    pub num_blocks: u64,
    pub num_reg_hints: u32,
    pub num_backup_hints: u32,
    pub total_hints: u32,
    pub _padding: u32,  // Align to 8 bytes
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for Rms24Params {}

/// Precomputed hint metadata (matches CUDA struct)
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct HintMeta {
    pub cutoff: u32,
    pub extra_block: u32,
    pub extra_offset: u32,
    pub _padding: u32,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for HintMeta {}

/// Hint output (48-byte parity for alignment)
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct HintOutput {
    pub parity: [u8; 48],
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for HintOutput {}

/// GPU hint generator using CUDA
#[cfg(feature = "cuda")]
pub struct GpuHintGenerator {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
    warp_kernel: CudaFunction,
}

#[cfg(feature = "cuda")]
impl GpuHintGenerator {
    /// Create a new GPU hint generator on the specified device.
    pub fn new(device_ord: usize) -> Result<Self, cudarc::driver::DriverError> {
        let device = CudaDevice::new(device_ord)?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/hint_kernel.ptx"));
        device.load_ptx(
            ptx.into(),
            "rms24",
            &["rms24_hint_gen_kernel", "rms24_hint_gen_warp_kernel"],
        )?;

        let kernel = device
            .get_func("rms24", "rms24_hint_gen_kernel")
            .expect("Failed to get rms24_hint_gen_kernel");

        let warp_kernel = device
            .get_func("rms24", "rms24_hint_gen_warp_kernel")
            .expect("Failed to get rms24_hint_gen_warp_kernel");

        Ok(Self {
            device,
            kernel,
            warp_kernel,
        })
    }

    /// Generate hints using GPU.
    ///
    /// Phase 1 (cutoffs, extras) must be done on CPU first.
    /// This function executes Phase 2 (parity accumulation) on GPU.
    ///
    /// # Arguments
    ///
    /// * `entries` - Database entries (N × ENTRY_SIZE bytes)
    /// * `prf_key` - 256-bit PRF key as 8 × u32
    /// * `hint_meta` - Precomputed hint metadata from Phase 1
    /// * `params` - RMS24 parameters
    ///
    /// # Returns
    ///
    /// Tuple of (main parities, backup high parities)
    pub fn generate_hints(
        &self,
        entries: &[u8],
        prf_key: &[u32; 8],
        hint_meta: &[HintMeta],
        params: Rms24Params,
    ) -> Result<(Vec<HintOutput>, Vec<HintOutput>), cudarc::driver::DriverError> {
        // Validate input
        let expected_size = params.num_entries as usize * ENTRY_SIZE;
        assert_eq!(entries.len(), expected_size, "entries size mismatch");
        assert_eq!(
            hint_meta.len(),
            params.total_hints as usize,
            "hint_meta size mismatch"
        );

        // Copy data to GPU
        let d_entries = self.device.htod_sync_copy(entries)?;
        let d_prf_key = self.device.htod_sync_copy(prf_key)?;
        let d_hint_meta: CudaSlice<HintMeta> = self.device.htod_sync_copy(hint_meta)?;

        // Allocate output buffers
        let total_hints = params.total_hints as usize;
        let num_backup = params.num_backup_hints as usize;
        let mut d_output: CudaSlice<HintOutput> = unsafe { self.device.alloc(total_hints)? };
        let mut d_backup_high: CudaSlice<HintOutput> = unsafe { self.device.alloc(num_backup)? };

        // Launch configuration
        let threads_per_block = 256u32;
        let num_blocks = (params.total_hints + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            self.kernel.clone().launch(
                cfg,
                (
                    params,
                    &d_prf_key,
                    &d_hint_meta,
                    &d_entries,
                    &mut d_output,
                    &mut d_backup_high,
                ),
            )?;
        }

        // Copy results back
        let output = self.device.dtoh_sync_copy(&d_output)?;
        let backup_high = self.device.dtoh_sync_copy(&d_backup_high)?;

        Ok((output, backup_high))
    }

    /// Generate hints using the warp-based kernel.
    ///
    /// Uses precomputed subset data instead of PRF-based selection.
    /// One warp (32 threads) per hint for coalesced memory access.
    pub fn generate_hints_warp(
        &self,
        entries: &[u8],
        subset_data: &SubsetData,
        params: Rms24Params,
    ) -> Result<Vec<HintOutput>, cudarc::driver::DriverError> {
        let expected_size = params.num_entries as usize * ENTRY_SIZE;
        assert_eq!(entries.len(), expected_size, "entries size mismatch");

        let num_hints = subset_data.starts.len();

        // Copy subset data arrays to GPU
        let d_entries = self.device.htod_sync_copy(entries)?;
        let d_blocks = self.device.htod_sync_copy(&subset_data.blocks)?;
        let d_offsets = self.device.htod_sync_copy(&subset_data.offsets)?;
        let d_starts = self.device.htod_sync_copy(&subset_data.starts)?;
        let d_sizes = self.device.htod_sync_copy(&subset_data.sizes)?;
        let d_extra_blocks = self.device.htod_sync_copy(&subset_data.extra_blocks)?;
        let d_extra_offsets = self.device.htod_sync_copy(&subset_data.extra_offsets)?;

        // Allocate output buffer
        let mut d_output: CudaSlice<HintOutput> = unsafe { self.device.alloc(num_hints)? };

        // For backup_high_output, pass empty slice (TODO: not implemented yet)
        let d_backup_high_empty: CudaSlice<HintOutput> = unsafe { self.device.alloc(0)? };

        // Launch config: one warp (32 threads) per hint
        let cfg = LaunchConfig {
            grid_dim: (num_hints as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            self.warp_kernel.clone().launch(
                cfg,
                (
                    params,
                    &d_blocks,
                    &d_offsets,
                    &d_starts,
                    &d_sizes,
                    &d_extra_blocks,
                    &d_extra_offsets,
                    &d_entries,
                    &mut d_output,
                    &d_backup_high_empty,
                ),
            )?;
        }

        // Copy results back
        let output = self.device.dtoh_sync_copy(&d_output)?;

        Ok(output)
    }
}

/// Convert SubsetData to GPU-compatible arrays.
/// Returns (blocks, offsets, starts, sizes, extra_blocks, extra_offsets).
#[cfg(feature = "cuda")]
pub fn subset_data_to_gpu_arrays(
    data: &SubsetData,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    (
        data.blocks.clone(),
        data.offsets.clone(),
        data.starts.clone(),
        data.sizes.clone(),
        data.extra_blocks.clone(),
        data.extra_offsets.clone(),
    )
}

/// Convert Client's HintState to GPU-compatible HintMeta array
#[cfg(feature = "cuda")]
pub fn hints_to_meta(hints: &crate::hints::HintState) -> Vec<HintMeta> {
    let total = hints.cutoffs.len();
    (0..total)
        .map(|i| HintMeta {
            cutoff: hints.cutoffs[i],
            extra_block: hints.extra_blocks[i],
            extra_offset: hints.extra_offsets[i],
            _padding: 0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_struct_sizes() {
        #[cfg(feature = "cuda")]
        {
            use super::*;
            assert_eq!(std::mem::size_of::<Rms24Params>(), 40);
            assert_eq!(std::mem::size_of::<HintMeta>(), 16);
            assert_eq!(std::mem::size_of::<HintOutput>(), 48);
        }
    }
}
