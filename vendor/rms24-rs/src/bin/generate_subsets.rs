//! Phase 1: Generate subsets on CPU and save to file.
//!
//! Usage:
//!   cargo run --release --bin generate_subsets -- \
//!     --db data/database.bin --lambda 80 --max-hints 10000 --output subsets.bin

use clap::Parser;
use rms24::client::Client;
use rms24::hints::SubsetData;
use rms24::Params;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "generate_subsets")]
#[command(about = "Generate hint subsets on CPU for GPU processing")]
struct Args {
    /// Path to database file (or "synthetic:SIZE_MB")
    #[arg(long)]
    db: String,

    /// Security parameter (lambda)
    #[arg(long, default_value = "80")]
    lambda: u32,

    /// Maximum hints to generate
    #[arg(long)]
    max_hints: u32,

    /// Output file for subset data
    #[arg(long)]
    output: PathBuf,

    /// Entry size in bytes
    #[arg(long, default_value = "40")]
    entry_size: usize,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    println!("RMS24 Subset Generator (Phase 1)");
    println!("=================================");

    let db_size = if args.db.starts_with("synthetic:") {
        let size_mb: usize = args.db.strip_prefix("synthetic:").unwrap().parse().unwrap();
        size_mb * 1024 * 1024
    } else {
        std::fs::metadata(&args.db)?.len() as usize
    };

    let num_entries = db_size / args.entry_size;
    println!("Database: {} ({:.2} GB, {} entries)", args.db, db_size as f64 / 1e9, num_entries);

    let params = Params::new(num_entries as u64, args.entry_size, args.lambda);
    println!("Block size: {}", params.block_size);
    println!("Num blocks: {}", params.num_blocks);

    let total_hints = args.max_hints.min(params.total_hints() as u32);
    println!("Generating: {} subsets", total_hints);
    io::stdout().flush()?;

    let start = Instant::now();
    let client = Client::new(params);
    let subsets = client.generate_subsets_range(0, total_hints as usize);
    let phase1_time = start.elapsed();
    println!("Subset generation: {:.2}s", phase1_time.as_secs_f64());

    let subset_data = SubsetData::from_subsets(&subsets);
    println!("Total blocks in subsets: {}", subset_data.blocks.len());

    println!("Saving to {}...", args.output.display());
    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);

    // Write header
    let num_hints = subset_data.starts.len() as u32;
    let total_blocks = subset_data.blocks.len() as u32;
    writer.write_all(&num_hints.to_le_bytes())?;
    writer.write_all(&total_blocks.to_le_bytes())?;

    // Write arrays
    for &v in &subset_data.blocks {
        writer.write_all(&v.to_le_bytes())?;
    }
    for &v in &subset_data.offsets {
        writer.write_all(&v.to_le_bytes())?;
    }
    for &v in &subset_data.starts {
        writer.write_all(&v.to_le_bytes())?;
    }
    for &v in &subset_data.sizes {
        writer.write_all(&v.to_le_bytes())?;
    }
    for &v in &subset_data.extra_blocks {
        writer.write_all(&v.to_le_bytes())?;
    }
    for &v in &subset_data.extra_offsets {
        writer.write_all(&v.to_le_bytes())?;
    }

    writer.flush()?;
    let file_size = std::fs::metadata(&args.output)?.len();
    println!("Saved: {:.2} MB", file_size as f64 / 1e6);

    Ok(())
}
