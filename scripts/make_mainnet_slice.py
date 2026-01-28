from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import data_slice

ENTRY_SIZE = 40


def copy_db_slice(source_db: Path, out_db: Path, entries: int) -> None:
    with source_db.open("rb") as src, out_db.open("wb") as dst:
        dst.write(src.read(entries * ENTRY_SIZE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source directory with database.bin and mappings")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--entries", type=int, default=1_000_000)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    copy_db_slice(source / "database.bin", out / "database.bin", args.entries)

    data_slice.filter_account_mapping_file(
        source / "account-mapping.bin",
        max_index=args.entries,
        out_path=out / "account-mapping.bin",
    )
    data_slice.filter_storage_mapping_file(
        source / "storage-mapping.bin",
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


if __name__ == "__main__":
    main()
