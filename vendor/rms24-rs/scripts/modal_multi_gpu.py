"""
Run RMS24 GPU hint generation across multiple GPUs in parallel.

Usage:
    modal run scripts/modal_multi_gpu.py --num-gpus 10 --max-hints 10000

Each GPU processes a shard of hints: GPU i handles [i*N/G, (i+1)*N/G).
"""

import modal

app = modal.App("rms24-multi-gpu")
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


@app.function(image=image, gpu="H200", volumes={"/data": volume}, timeout=7200)
def run_shard(
    gpu_id: int,
    num_gpus: int,
    db_path: str,
    total_hints: int,
    lambda_param: int,
) -> dict:
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
        ["cargo", "build", "--release", "--bin", "bench_gpu_hints", "--features", "cuda"],
        capture_output=True,
        text=True,
        env=env,
    )
    if build_result.returncode != 0:
        return {"gpu_id": gpu_id, "error": build_result.stderr[-1000:]}

    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", db_path,
        "--lambda", str(lambda_param),
        "--iterations", "1",
        "--warmup", "1",
        "--max-hints", str(hint_count),
        "--hint-start", str(hint_start),
    ]

    start_time = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start_time

    return {
        "gpu_id": gpu_id,
        "hint_start": hint_start,
        "hint_count": hint_count,
        "elapsed_sec": elapsed,
        "output": result.stdout[-2000:] if result.returncode == 0 else result.stderr[-2000:],
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main(
    num_gpus: int = 4,
    db: str = "synthetic:1000",
    lambda_param: int = 40,
    max_hints: int = 1000,
):
    """
    Run RMS24 across multiple H200 GPUs in parallel.

    Args:
        num_gpus: Number of H200 GPUs to use
        db: Database path or "synthetic:SIZE_MB"
        lambda_param: Security parameter
        max_hints: Total hints to generate (split across GPUs)
    """
    print(f"=== RMS24 MULTI-GPU BENCHMARK ===")
    print(f"GPUs: {num_gpus} x H200")
    print(f"Database: {db}")
    print(f"Total hints: {max_hints:,}")
    print(f"Hints per GPU: ~{max_hints // num_gpus:,}")
    print()

    futures = []
    for gpu_id in range(num_gpus):
        future = run_shard.spawn(gpu_id, num_gpus, db, max_hints, lambda_param)
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
        print(f"\nWall time: {total_time:.2f}s")
        print(f"Total hints: {total_hints_processed:,}")
        print(f"Aggregate throughput: {total_hints_processed / total_time:.0f} hints/sec")
