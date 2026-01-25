"""
Distributed RMS24 hint generation with shared Phase 1.

Phase 1 (CPU): Compute all subsets once, save to volume
Phase 2 (GPU): Each GPU loads its shard and runs kernel

Usage:
    modal run scripts/modal_distributed.py --num-gpus 10 --max-hints 10000
"""

import modal
import struct

app = modal.App("rms24-distributed")
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=["target", ".git", ".jj", "__pycache__", ".DS_Store", "data"],
    )
)


@app.function(image=image, cpu=32, memory=65536, volumes={"/data": volume}, timeout=7200)
def phase1_compute_subsets(
    db_path: str,
    total_hints: int,
    lambda_param: int,
    entry_size: int = 40,
) -> str:
    """
    Phase 1: Compute all subsets on CPU, save to volume.
    Returns path to saved subset data.
    """
    import os
    import subprocess
    import time

    os.chdir("/app")

    env = os.environ.copy()
    env["PATH"] = f"/root/.cargo/bin:{env.get('PATH', '')}"

    print("=== Phase 1: Building subset generator ===")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "generate_subsets"],
        capture_output=True,
        text=True,
        env=env,
    )
    if build_result.returncode != 0:
        print(build_result.stderr[-2000:])
        raise RuntimeError("Build failed")

    output_path = "/data/rms24_subsets.bin"
    
    print(f"=== Generating {total_hints} subsets ===")
    start = time.time()
    result = subprocess.run(
        [
            "./target/release/generate_subsets",
            "--db", db_path,
            "--lambda", str(lambda_param),
            "--max-hints", str(total_hints),
            "--output", output_path,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("Subset generation failed")
    
    print(result.stdout)
    print(f"Phase 1 complete: {elapsed:.2f}s")
    
    volume.commit()
    return output_path


@app.function(image=image, gpu="H200", volumes={"/data": volume}, timeout=7200)
def phase2_gpu_kernel(
    gpu_id: int,
    num_gpus: int,
    db_path: str,
    subset_path: str,
    total_hints: int,
    lambda_param: int,
) -> dict:
    """
    Phase 2: Load precomputed subsets and run GPU kernel.
    """
    import os
    import subprocess
    import time

    os.chdir("/app")

    hints_per_gpu = total_hints // num_gpus
    hint_start = gpu_id * hints_per_gpu
    hint_count = hints_per_gpu if gpu_id < num_gpus - 1 else total_hints - hint_start

    env = os.environ.copy()
    env["PATH"] = f"/root/.cargo/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_ROOT"] = "/usr/local/cuda"
    env["CUDA_PATH"] = "/usr/local/cuda"
    env["CUDA_ARCH"] = "sm_90"

    build_result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "run_gpu_kernel", "--features", "cuda"],
        capture_output=True,
        text=True,
        env=env,
    )
    if build_result.returncode != 0:
        return {"gpu_id": gpu_id, "error": build_result.stderr[-1000:]}

    start_time = time.time()
    result = subprocess.run(
        [
            "./target/release/run_gpu_kernel",
            "--db", db_path,
            "--subsets", subset_path,
            "--hint-start", str(hint_start),
            "--hint-count", str(hint_count),
            "--lambda", str(lambda_param),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start_time

    return {
        "gpu_id": gpu_id,
        "hint_start": hint_start,
        "hint_count": hint_count,
        "elapsed_sec": elapsed,
        "output": result.stdout[-1500:] if result.returncode == 0 else result.stderr[-1500:],
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main(
    num_gpus: int = 4,
    db: str = "synthetic:100",
    lambda_param: int = 8,
    max_hints: int = 1000,
):
    """
    Distributed RMS24 with shared Phase 1.

    1. Phase 1 (CPU): Compute all subsets once
    2. Phase 2 (GPU): Each GPU loads its shard
    """
    print(f"=== RMS24 DISTRIBUTED HINT GENERATION ===")
    print(f"Phase 1: 1x CPU (32 cores)")
    print(f"Phase 2: {num_gpus}x H200 GPU")
    print(f"Database: {db}")
    print(f"Total hints: {max_hints:,}")
    print()

    print(">>> Starting Phase 1 (CPU subset computation)...")
    subset_path = phase1_compute_subsets.remote(db, max_hints, lambda_param)
    print(f"Subsets saved to: {subset_path}")
    print()

    print(f">>> Starting Phase 2 ({num_gpus} GPUs)...")
    futures = []
    for gpu_id in range(num_gpus):
        future = phase2_gpu_kernel.spawn(
            gpu_id, num_gpus, db, subset_path, max_hints, lambda_param
        )
        futures.append(future)

    results = [f.get() for f in futures]

    print("\n=== RESULTS ===")
    total_time = 0
    total_hints_processed = 0
    for r in sorted(results, key=lambda x: x["gpu_id"]):
        if r.get("error"):
            print(f"GPU {r['gpu_id']}: ERROR - {r['error'][:200]}")
        else:
            print(f"GPU {r['gpu_id']}: hints {r['hint_start']}..{r['hint_start']+r['hint_count']}, {r['elapsed_sec']:.2f}s")
            total_time = max(total_time, r['elapsed_sec'])
            total_hints_processed += r['hint_count']

    if total_hints_processed > 0:
        print(f"\nPhase 2 wall time: {total_time:.2f}s")
        print(f"Total hints: {total_hints_processed:,}")
        print(f"GPU throughput: {total_hints_processed / total_time:.0f} hints/sec")
