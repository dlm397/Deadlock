"""Keep one example CSV per research folder; run after collection has finished.

Run from Deadlock2: python other/cleanup_csv.py
Set DELETE_FILES = False to preview, or True to delete permanently.
Review the setting before running. Parquet files, version
definitions, collection_status.csv, and failures.csv are never deleted.

Set export_csv = false in config.toml to stop future research CSV exports.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from app.console_log import log


DATA_DIR = HERE / "data"
KEEP_MATCH_ID = "109521468"  # Keep this match's CSV in each research folder.
DELETE_FILES = False        # Change to True only after reviewing the preview.
TABLES = ("matches", "end_match", "time_slices", "events")


def main():
    root = DATA_DIR.resolve()
    targets = []
    for name in TABLES:
        folder = (root / name).resolve()
        if folder.parent != root:
            raise ValueError(f"Research folder resolves outside the data directory: {folder}")
        keep = folder / f"{KEEP_MATCH_ID}.csv"
        if not keep.is_file():
            log(f"SKIP: {name}: example CSV {keep.name} is missing; nothing will be deleted here.")
            continue
        redundant = [path for path in sorted(folder.glob("*.csv"))
                     if path != keep and path.resolve().parent == folder
                     and path.with_suffix(".parquet").is_file()]
        targets.extend(redundant)
        log(f"{name}: keep {keep.name}; {'delete' if DELETE_FILES else 'would delete'} "
            f"{len(redundant)} CSVs with matching Parquet files.")

    size_mib = sum(path.stat().st_size for path in targets) / 1024**2
    if not DELETE_FILES:
        log(f"PREVIEW ONLY: {len(targets)} CSVs, {size_mib:.1f} MiB. No files deleted.")
        return
    deleted = 0
    for path in targets:
        try:
            path.unlink()
            deleted += 1
        except OSError as error:
            log(f"Could not delete {path.name}: {error}")
    log(f"Deleted {deleted} of {len(targets)} redundant CSVs.")


if __name__ == "__main__":
    main()
