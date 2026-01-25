//! Phase 2: Run GPU kernel with precomputed subsets.
//!
//! Usage:
//!   cargo run --release --features cuda --bin run_gpu_kernel -- \
//!     --db data/database.bin --subsets subsets.bin --hint-start 0 --hint-count 1000

use clap::Parser;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "run_gpu_kernel")]
#[command(about = "Run GPU hint generation kernel with precomputed subsets")]
struct Args {
    /// Path to database file (or "synthetic:SIZE_MB")
    #[arg(long)]
    db: String,

    /// Path to precomputed subset file
    #[arg(long)]
    subsets: PathBuf,

    /// Starting hint index
    #[arg(long, default_value = "0")]
    hint_start: u32,

    /// Number of hints to process
    #[arg(long)]
    hint_count: u32,

    /// Security parameter (for params calculation)
    #[arg(long, default_value = "80")]
    lambda: u32,

    /// Entry size in bytes
    #[arg(long, default_value = "40")]
    entry_size: usize,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    println!("RMS24 GPU Kernel Runner (Phase 2)");
    println!("==================================");

    let db_data = if args.db.starts_with("synthetic:") {
        let size_mb: usize = args.db.strip_prefix("synthetic:").unwrap().parse().unwrap();
        vec![0xABu8; size_mb * 1024 * 1024]
    } else {
        std::fs::read(&args.db)?
    };

    let num_entries = db_data.len() / args.entry_size;
    println!("Database: {} ({:.2} GB)", args.db, db_data.len() as f64 / 1e9);

    println!("Loading subsets from {}...", args.subsets.display());
    let file = File::open(&args.subsets)?;
    let mut reader = BufReader::new(file);

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let total_hints_in_file = u32::from_le_bytes(buf4);
    reader.read_exact(&mut buf4)?;
    let total_blocks_in_file = u32::from_le_bytes(buf4);

    println!("Subset file: {} hints, {} total blocks", total_hints_in_file, total_blocks_in_file);

    fn read_u32_vec(reader: &mut BufReader<File>, count: usize) -> io::Result<Vec<u32>> {
        let mut data = vec![0u32; count];
        for i in 0..count {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            data[i] = u32::from_le_bytes(buf);
        }
        Ok(data)
    }

    let all_blocks = read_u32_vec(&mut reader, total_blocks_in_file as usize)?;
    let all_offsets = read_u32_vec(&mut reader, total_blocks_in_file as usize)?;
    let all_starts = read_u32_vec(&mut reader, total_hints_in_file as usize)?;
    let all_sizes = read_u32_vec(&mut reader, total_hints_in_file as usize)?;
    let all_extra_blocks = read_u32_vec(&mut reader, total_hints_in_file as usize)?;
    let all_extra_offsets = read_u32_vec(&mut reader, total_hints_in_file as usize)?;

    let hint_start = args.hint_start as usize;
    let hint_end = (hint_start + args.hint_count as usize).min(total_hints_in_file as usize);
    let hint_count = hint_end - hint_start;

    println!("Processing hints {}..{} ({} hints)", hint_start, hint_end, hint_count);

    let block_start = all_starts[hint_start] as usize;
    let block_end = if hint_end < total_hints_in_file as usize {
        all_starts[hint_end] as usize
    } else {
        total_blocks_in_file as usize
    };

    let subset_data = rms24::hints::SubsetData {
        blocks: all_blocks[block_start..block_end].to_vec(),
        offsets: all_offsets[block_start..block_end].to_vec(),
        starts: all_starts[hint_start..hint_end]
            .iter()
            .map(|&s| s - block_start as u32)
            .collect(),
        sizes: all_sizes[hint_start..hint_end].to_vec(),
        extra_blocks: all_extra_blocks[hint_start..hint_end].to_vec(),
        extra_offsets: all_extra_offsets[hint_start..hint_end].to_vec(),
        is_regular: vec![true; hint_count],
    };

    println!("Subset shard: {} blocks", subset_data.blocks.len());
    io::stdout().flush()?;

    #[cfg(feature = "cuda")]
    {
        use rms24::gpu::{GpuHintGenerator, Rms24Params};

        let params = rms24::Params::new(num_entries as u64, args.entry_size, args.lambda);

        let gpu_params = Rms24Params {
            num_entries: params.num_entries,
            block_size: params.block_size,
            num_blocks: params.num_blocks,
            num_reg_hints: params.num_reg_hints as u32,
            num_backup_hints: params.num_backup_hints as u32,
            total_hints: hint_count as u32,
            _padding: 0,
        };

        println!("Initializing GPU...");
        io::stdout().flush()?;
        let generator = GpuHintGenerator::new(0).expect("Failed to init GPU");

        println!("Running GPU kernel...");
        io::stdout().flush()?;
        let start = Instant::now();
        let output = generator
            .generate_hints_warp(&db_data, &subset_data, gpu_params)
            .expect("GPU kernel failed");
        let elapsed = start.elapsed();

        println!("GPU time: {:.2}ms", elapsed.as_millis());
        println!("Throughput: {:.0} hints/sec", hint_count as f64 / elapsed.as_secs_f64());
        println!("Generated {} parities", output.len());
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: CUDA feature not enabled");
        std::process::exit(1);
    }

    Ok(())
}
