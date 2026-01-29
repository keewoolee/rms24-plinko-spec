# R2 Data Slice Distribution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a 1M-entry mainnet-v3 slice, upload to R2, and add download/validation helpers + tests for real-data slice usage.

**Architecture:** Add a small Python utility module for slicing/filtering + checksum metadata, a CLI to build the slice from `/mnt/mainnet-pir-data-v3`, a download helper that validates checksums, and optional integration tests gated by `RMS24_DATA_DIR`.

**Tech Stack:** Python 3, pytest, stdlib (hashlib/json/urllib), optional boto3 for upload.

---

### Task 1: Add data-slice utilities + unit tests

**Files:**
- Create: `scripts/data_slice.py`
- Create: `scripts/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data_slice.py`

**Step 1: Write the failing test**

```python
# tests/test_data_slice.py
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from scripts import data_slice
from scripts.make_mainnet_slice import ENTRY_SIZE, copy_db_slice


def test_filter_account_mapping_by_index(tmp_path: Path):
    # 3 records: idx 5, idx 10, and idx 12
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    data = record(b"a" * 20, 5) + record(b"b" * 20, 10) + record(b"c" * 20, 12)
    out = tmp_path / "account.bin"

    data_slice.filter_account_mapping_bytes(data, max_index=10, out_path=out)

    assert out.read_bytes() == record(b"a" * 20, 5)


def test_filter_account_mapping_excludes_max_index(tmp_path: Path):
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    data = record(b"a" * 20, 10)
    out = tmp_path / "account.bin"

    data_slice.filter_account_mapping_bytes(data, max_index=10, out_path=out)

    assert out.read_bytes() == b""


def test_filter_storage_mapping_by_index(tmp_path: Path):
    record = lambda addr, slot, idx: addr + slot + idx.to_bytes(4, "little")
    data = record(b"a" * 20, b"s" * 32, 3) + record(b"b" * 20, b"t" * 32, 10)
    data += record(b"c" * 20, b"u" * 32, 99)
    out = tmp_path / "storage.bin"

    data_slice.filter_storage_mapping_bytes(data, max_index=10, out_path=out)

    assert out.read_bytes() == record(b"a" * 20, b"s" * 32, 3)


def test_filter_account_mapping_invalid_length_raises(tmp_path: Path):
    data = b"a" * 23
    out = tmp_path / "account.bin"

    with pytest.raises(ValueError):
        data_slice.filter_account_mapping_bytes(data, max_index=10, out_path=out)


def test_filter_storage_mapping_invalid_length_raises(tmp_path: Path):
    data = b"a" * 55
    out = tmp_path / "storage.bin"

    with pytest.raises(ValueError):
        data_slice.filter_storage_mapping_bytes(data, max_index=10, out_path=out)


def test_filter_account_mapping_file(tmp_path: Path):
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    data = record(b"a" * 20, 3) + record(b"b" * 20, 10) + record(b"c" * 20, 11)
    src = tmp_path / "account.bin"
    out = tmp_path / "account_out.bin"
    src.write_bytes(data)

    data_slice.filter_account_mapping_file(src, max_index=10, out_path=out, chunk_records=2)

    assert out.read_bytes() == record(b"a" * 20, 3)


def test_filter_storage_mapping_file(tmp_path: Path):
    record = lambda addr, slot, idx: addr + slot + idx.to_bytes(4, "little")
    data = record(b"a" * 20, b"s" * 32, 3) + record(b"b" * 20, b"t" * 32, 10)
    data += record(b"c" * 20, b"u" * 32, 11)
    src = tmp_path / "storage.bin"
    out = tmp_path / "storage_out.bin"
    src.write_bytes(data)

    data_slice.filter_storage_mapping_file(src, max_index=10, out_path=out, chunk_records=2)

    assert out.read_bytes() == record(b"a" * 20, b"s" * 32, 3)


def test_filter_account_mapping_file_invalid_length_raises(tmp_path: Path):
    src = tmp_path / "account.bin"
    out = tmp_path / "account_out.bin"
    src.write_bytes(b"a" * 23)

    with pytest.raises(ValueError):
        data_slice.filter_account_mapping_file(src, max_index=10, out_path=out)


def test_filter_storage_mapping_file_invalid_length_raises(tmp_path: Path):
    src = tmp_path / "storage.bin"
    out = tmp_path / "storage_out.bin"
    src.write_bytes(b"a" * 55)

    with pytest.raises(ValueError):
        data_slice.filter_storage_mapping_file(src, max_index=10, out_path=out)


def test_filter_account_mapping_file_invalid_chunk_records_raises(tmp_path: Path):
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    src = tmp_path / "account.bin"
    out = tmp_path / "account_out.bin"
    src.write_bytes(record(b"a" * 20, 3))

    with pytest.raises(ValueError):
        data_slice.filter_account_mapping_file(src, max_index=10, out_path=out, chunk_records=0)


def test_filter_storage_mapping_file_invalid_chunk_records_raises(tmp_path: Path):
    record = lambda addr, slot, idx: addr + slot + idx.to_bytes(4, "little")
    src = tmp_path / "storage.bin"
    out = tmp_path / "storage_out.bin"
    src.write_bytes(record(b"a" * 20, b"s" * 32, 3))

    with pytest.raises(ValueError):
        data_slice.filter_storage_mapping_file(src, max_index=10, out_path=out, chunk_records=0)


def test_write_metadata(tmp_path: Path):
    db = tmp_path / "database.bin"
    payload_bytes = b"x" * 40
    db.write_bytes(payload_bytes)
    meta = tmp_path / "metadata.json"

    data_slice.write_metadata(
        meta_path=meta,
        entries=1,
        entry_size=40,
        files={"database.bin": db},
        source_tag="mainnet-v3",
    )

    payload = json.loads(meta.read_text())
    assert payload["entries"] == 1
    assert payload["entry_size"] == 40
    assert payload["source"] == "mainnet-v3"
    assert "database.bin" in payload["files"]
    assert payload["files"]["database.bin"]["bytes"] == 40
    assert payload["files"]["database.bin"]["sha256"] == hashlib.sha256(payload_bytes).hexdigest()


def test_copy_db_slice_requires_full_bytes(tmp_path: Path):
    source_db = tmp_path / "database.bin"
    source_db.write_bytes(b"a" * ENTRY_SIZE)
    out_db = tmp_path / "out.bin"

    with pytest.raises(ValueError):
        copy_db_slice(source_db, out_db, entries=2)

    assert not out_db.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_slice.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.data_slice'` (or missing functions).

**Step 3: Write minimal implementation**

```python
# scripts/data_slice.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ACCOUNT_RECORD_SIZE = 24
STORAGE_RECORD_SIZE = 56


def filter_account_mapping_bytes(data: bytes, max_index: int, out_path: Path) -> None:
    if len(data) % ACCOUNT_RECORD_SIZE != 0:
        raise ValueError("account mapping data length is not aligned to record size")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=out_path.parent) as out:
            tmp_path = Path(out.name)
            for i in range(0, len(data), ACCOUNT_RECORD_SIZE):
                record = data[i : i + ACCOUNT_RECORD_SIZE]
                idx = int.from_bytes(record[20:24], "little")
                if idx < max_index:
                    out.write(record)
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def filter_storage_mapping_bytes(data: bytes, max_index: int, out_path: Path) -> None:
    if len(data) % STORAGE_RECORD_SIZE != 0:
        raise ValueError("storage mapping data length is not aligned to record size")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=out_path.parent) as out:
            tmp_path = Path(out.name)
            for i in range(0, len(data), STORAGE_RECORD_SIZE):
                record = data[i : i + STORAGE_RECORD_SIZE]
                idx = int.from_bytes(record[52:56], "little")
                if idx < max_index:
                    out.write(record)
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def filter_account_mapping_file(
    path: Path, max_index: int, out_path: Path, *, chunk_records: int = 4096
) -> None:
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive")
    size = path.stat().st_size
    if size % ACCOUNT_RECORD_SIZE != 0:
        raise ValueError("account mapping data length is not aligned to record size")
    chunk_size = ACCOUNT_RECORD_SIZE * chunk_records
    tmp_path = None
    try:
        with path.open("rb") as src, tempfile.NamedTemporaryFile(
            "wb", delete=False, dir=out_path.parent
        ) as out:
            tmp_path = Path(out.name)
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                if len(chunk) % ACCOUNT_RECORD_SIZE != 0:
                    raise ValueError("account mapping data length is not aligned to record size")
                for i in range(0, len(chunk), ACCOUNT_RECORD_SIZE):
                    record = chunk[i : i + ACCOUNT_RECORD_SIZE]
                    idx = int.from_bytes(record[20:24], "little")
                    if idx < max_index:
                        out.write(record)
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def filter_storage_mapping_file(
    path: Path, max_index: int, out_path: Path, *, chunk_records: int = 4096
) -> None:
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive")
    size = path.stat().st_size
    if size % STORAGE_RECORD_SIZE != 0:
        raise ValueError("storage mapping data length is not aligned to record size")
    chunk_size = STORAGE_RECORD_SIZE * chunk_records
    tmp_path = None
    try:
        with path.open("rb") as src, tempfile.NamedTemporaryFile(
            "wb", delete=False, dir=out_path.parent
        ) as out:
            tmp_path = Path(out.name)
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                if len(chunk) % STORAGE_RECORD_SIZE != 0:
                    raise ValueError("storage mapping data length is not aligned to record size")
                for i in range(0, len(chunk), STORAGE_RECORD_SIZE):
                    record = chunk[i : i + STORAGE_RECORD_SIZE]
                    idx = int.from_bytes(record[52:56], "little")
                    if idx < max_index:
                        out.write(record)
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(meta_path: Path, entries: int, entry_size: int, files: dict[str, Path], source_tag: str) -> None:
    meta = {
        "entries": entries,
        "entry_size": entry_size,
        "source": source_tag,
        "files": {
            name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_slice.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/data_slice.py tests/test_data_slice.py
git commit -m "test: add data slice utilities" 
```

---

### Task 2: Create slice builder CLI

**Files:**
- Create: `scripts/make_mainnet_slice.py`
- Test: `tests/test_data_slice.py`

**Step 1: Write the failing test**

```python
# tests/test_data_slice.py (append)
import subprocess
import sys


def test_make_slice_cli(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Create tiny mock DB (3 entries)
    (source_dir / "database.bin").write_bytes(b"a" * 40 + b"b" * 40 + b"c" * 40)
    # Account mapping: two records with indices 0 and 2
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little") + b"y" * 20 + (2).to_bytes(4, "little"))
    # Storage mapping: one record with index 1
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (1).to_bytes(4, "little"))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source", str(source_dir),
        "--out", str(out_dir),
        "--entries", "2",
    ]
    subprocess.check_call(cmd)

    assert (out_dir / "database.bin").stat().st_size == 2 * 40
    assert (out_dir / "account-mapping.bin").stat().st_size == 24  # idx 0 only
    assert (out_dir / "storage-mapping.bin").stat().st_size == 56  # idx 1 only
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["entries"] == 2
    assert meta["entry_size"] == 40
    assert meta["files"]["database.bin"]["bytes"] == 2 * 40


def test_make_slice_cli_rejects_non_positive_entries(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "entries must be positive" in proc.stderr


def test_make_slice_cli_rejects_too_small_database(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "database.bin size" in proc.stderr


def test_make_slice_cli_rejects_non_integer_entries(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "abc",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "entries must be an integer" in proc.stderr


def test_make_slice_cli_rejects_missing_mapping_file(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "storage-mapping.bin" in proc.stderr


def test_make_slice_cli_rejects_missing_database(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "database.bin" in proc.stderr


def test_make_slice_cli_rejects_unaligned_account_mapping(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 23)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 56)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "account-mapping.bin size" in proc.stderr


def test_make_slice_cli_rejects_unaligned_storage_mapping(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 24)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 55)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "storage-mapping.bin size" in proc.stderr


def test_make_slice_cli_cleans_outputs_on_failure(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 23)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 56)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert not (out_dir / "database.bin").exists()
    assert not (out_dir / "account-mapping.bin").exists()
    assert not (out_dir / "storage-mapping.bin").exists()
    assert not (out_dir / "metadata.json").exists()


def test_make_slice_cli_rejects_out_inside_source(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 24)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 56)
    out_dir = source_dir / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "outside source directory" in proc.stderr
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_slice.py -v`
Expected: FAIL with `No such file or directory: scripts/make_mainnet_slice.py`.

**Step 3: Write minimal implementation**

```python
# scripts/make_mainnet_slice.py
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import data_slice

ENTRY_SIZE = 40
COPY_CHUNK_SIZE = 1024 * 1024


def positive_int(value: str) -> int:
    try:
        entries = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entries must be an integer") from exc
    if entries <= 0:
        raise argparse.ArgumentTypeError("entries must be positive")
    return entries


def require_file(parser: argparse.ArgumentParser, path: Path) -> None:
    if not path.is_file():
        parser.error(f"missing required file: {path.name}")


def require_aligned(path: Path, record_size: int) -> None:
    size = path.stat().st_size
    if size % record_size != 0:
        raise ValueError(
            f"{path.name} size {size} is not aligned to record size {record_size}"
        )


def copy_db_slice(source_db: Path, out_db: Path, entries: int) -> None:
    remaining = entries * ENTRY_SIZE
    tmp_path = None
    try:
        with source_db.open("rb") as src, tempfile.NamedTemporaryFile(
            "wb", delete=False, dir=out_db.parent
        ) as dst:
            tmp_path = Path(dst.name)
            while remaining > 0:
                chunk = src.read(min(COPY_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                dst.write(chunk)
                remaining -= len(chunk)
        if remaining != 0:
            raise ValueError("database.bin truncated while copying slice")
        tmp_path.replace(out_db)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def cleanup_outputs(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source directory with database.bin and mappings")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--entries", type=positive_int, default=1_000_000)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source_db = source / "database.bin"
    account_mapping = source / "account-mapping.bin"
    storage_mapping = source / "storage-mapping.bin"
    require_file(parser, source_db)
    require_file(parser, account_mapping)
    require_file(parser, storage_mapping)

    source_resolved = source.resolve()
    out_resolved = out.resolve()
    try:
        out_resolved.relative_to(source_resolved)
        parser.error("output directory must be outside source directory")
    except ValueError:
        pass

    required_bytes = args.entries * ENTRY_SIZE
    db_size = source_db.stat().st_size
    if db_size < required_bytes:
        parser.error(
            f"database.bin size {db_size} is smaller than required {required_bytes}"
        )

    try:
        copy_db_slice(source_db, out / "database.bin", args.entries)
        require_aligned(account_mapping, data_slice.ACCOUNT_RECORD_SIZE)
        require_aligned(storage_mapping, data_slice.STORAGE_RECORD_SIZE)
        data_slice.filter_account_mapping_file(
            account_mapping,
            max_index=args.entries,
            out_path=out / "account-mapping.bin",
        )
        data_slice.filter_storage_mapping_file(
            storage_mapping,
            max_index=args.entries,
            out_path=out / "storage-mapping.bin",
        )
        data_slice.write_metadata(
            meta_path=out / "metadata.json",
            entries=args.entries,
            entry_size=ENTRY_SIZE,
            files={
                "database.bin": out / "database.bin",
                "account-mapping.bin": out / "account-mapping.bin",
                "storage-mapping.bin": out / "storage-mapping.bin",
            },
            source_tag="mainnet-v3",
        )
    except Exception as exc:
        cleanup_outputs(
            [
                out / "database.bin",
                out / "account-mapping.bin",
                out / "storage-mapping.bin",
                out / "metadata.json",
            ]
        )
        parser.error(str(exc))


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_slice.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/make_mainnet_slice.py tests/test_data_slice.py
git commit -m "feat: add mainnet slice builder"
```

---

### Task 3: Add download helper + tests

**Files:**
- Create: `scripts/download_slice.py`
- Modify: `tests/test_data_slice.py`

**Step 1: Write the failing test**

```python
# tests/test_data_slice.py (append)
import json


def test_download_helper_validates_checksum(tmp_path: Path):
    # Fake metadata and file
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    data = b"abc"
    file_path = out_dir / "database.bin"
    file_path.write_bytes(data)

    meta = {
        "entries": 1,
        "entry_size": 40,
        "files": {"database.bin": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": 3}},
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta))

    from scripts import download_slice

    download_slice.verify_files(out_dir, meta)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_slice.py -v`
Expected: FAIL with missing `scripts.download_slice`.

**Step 3: Write minimal implementation**

```python
# scripts/download_slice.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://pir.53627.org/mainnet-v3-slice-1m"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(out_dir: Path, meta: dict) -> None:
    for name, info in meta["files"].items():
        path = out_dir / name
        if path.stat().st_size != info["bytes"]:
            raise ValueError(f"Size mismatch for {name}")
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")


def download_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"{BASE_URL}/metadata.json", out_dir / "metadata.json")
    meta = json.loads((out_dir / "metadata.json").read_text())
    for name in meta["files"].keys():
        urlretrieve(f"{BASE_URL}/{name}", out_dir / name)
    verify_files(out_dir, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    download_all(Path(args.out))


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_slice.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/download_slice.py tests/test_data_slice.py
git commit -m "feat: add data slice download helper"
```

---

### Task 4: Add R2 upload helper (optional) + tests

**Files:**
- Create: `scripts/upload_slice_r2.py`
- Modify: `tests/test_data_slice.py`

**Step 1: Write the failing test**

```python
# tests/test_data_slice.py (append)
from scripts import upload_slice_r2


def test_build_r2_keys():
    files = ["database.bin", "metadata.json"]
    keys = upload_slice_r2.build_object_keys(prefix="mainnet-v3-slice-1m", files=files)
    assert keys["database.bin"].endswith("mainnet-v3-slice-1m/database.bin")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_slice.py -v`
Expected: FAIL with missing module.

**Step 3: Write minimal implementation**

```python
# scripts/upload_slice_r2.py
from __future__ import annotations

import argparse
from pathlib import Path


def build_object_keys(prefix: str, files: list[str]) -> dict[str, str]:
    return {name: f"{prefix}/{name}" for name in files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix", default="mainnet-v3-slice-1m")
    parser.add_argument("--bucket", default="pir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = ["database.bin", "account-mapping.bin", "storage-mapping.bin", "metadata.json"]
    keys = build_object_keys(args.prefix, files)

    if args.dry_run:
        for name, key in keys.items():
            print(f"DRY RUN: {name} -> s3://{args.bucket}/{key}")
        return

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url="https://pir.53627.org",
    )
    for name, key in keys.items():
        s3.upload_file(str(Path(args.dir) / name), args.bucket, key)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_slice.py -v`
Expected: PASS (no boto3 required in dry-run test).

**Step 5: Commit**

```bash
git add scripts/upload_slice_r2.py tests/test_data_slice.py
git commit -m "feat: add R2 upload helper"
```

---

### Task 5: Add README instructions + integration test gate

**Files:**
- Modify: `README.md`
- Modify: `tests/test_data_slice.py` (add optional integration check)

**Step 1: Write the failing test**

```python
# tests/test_data_slice.py (append)
import os


def test_real_slice_guarded():
    data_dir = os.environ.get("RMS24_DATA_DIR")
    if not data_dir:
        return
    base = Path(data_dir)
    db = base / "database.bin"
    assert db.exists()
    assert db.stat().st_size % 40 == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_slice.py -v`
Expected: PASS if `RMS24_DATA_DIR` unset; otherwise fail if files missing.

**Step 3: Update README**

Add a “Real Data Slice” section with:

- Download: `python scripts/download_slice.py --out tests/data/mainnet-v3-slice-1m`
- Use: `RMS24_DATA_DIR=tests/data/mainnet-v3-slice-1m pytest tests/test_data_slice.py -v`
- Public base URL: `https://pir.53627.org/mainnet-v3-slice-1m/`

**Step 4: Run tests**

Run: `pytest tests/test_data_slice.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md tests/test_data_slice.py
git commit -m "docs: add real data slice download instructions"
```

---

## Execution

Plan complete and saved to `docs/plans/2026-01-28-data-slice-r2-distribution.md`.

Two execution options:

1. Subagent-Driven (this session) — I dispatch a fresh subagent per task, review between tasks.
2. Parallel Session (separate) — Open a new session and use superpowers:executing-plans.

Which approach?
