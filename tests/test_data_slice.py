import hashlib
import json
from pathlib import Path

import pytest

from scripts import data_slice


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
    data = record(b"a" * 20, b"s" * 32, 3) + record(b"b" * 20, b"t" * 32, 10) + record(b"c" * 20, b"u" * 32, 99)
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
    data = record(b"a" * 20, 3)
    src = tmp_path / "account.bin"
    out = tmp_path / "account_out.bin"
    src.write_bytes(data)

    with pytest.raises(ValueError):
        data_slice.filter_account_mapping_file(src, max_index=10, out_path=out, chunk_records=0)


def test_filter_storage_mapping_file_invalid_chunk_records_raises(tmp_path: Path):
    record = lambda addr, slot, idx: addr + slot + idx.to_bytes(4, "little")
    data = record(b"a" * 20, b"s" * 32, 3)
    src = tmp_path / "storage.bin"
    out = tmp_path / "storage_out.bin"
    src.write_bytes(data)

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
