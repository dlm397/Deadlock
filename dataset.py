from pathlib import Path

import pandas as pd

from main import OUTPUT_DIR
from agent.io import GROUPS, checksum, group_for, load_table, read_json


def match_ids(output_dir=OUTPUT_DIR):
    root = Path(output_dir) / "raw"
    if not root.exists():
        return []
    return sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()
                  and read_json(path / "sources/status.json", {}).get("processed_success")
                  and all((path / f"{group}.zip").exists() for group in GROUPS))


def read_match(match_id, table="player_state", columns=None, output_dir=OUTPUT_DIR):
    folder = Path(output_dir) / "raw" / str(int(match_id))
    status = read_json(folder / "sources/status.json", {})
    filename = f"{group_for(table)}.zip"
    if not status.get("processed_success") or checksum(folder / filename) != status.get("files", {}).get(filename):
        raise ValueError(f"Match {match_id} has an incomplete or damaged {filename}; rerun collect.py")
    return load_table(folder, table, columns)


def iter_matches(table="player_state", columns=None, output_dir=OUTPUT_DIR):
    # Consume one frame at a time to keep collection size out of memory.
    for mid in match_ids(output_dir):
        yield mid, read_match(mid, table, columns, output_dir)


def read_reference(table="heroes", version=None, output_dir=OUTPUT_DIR):
    if table not in ("heroes", "items", "abilities"):
        raise ValueError("Reference table must be heroes, items, or abilities")
    result = pd.read_parquet(Path(output_dir) / "tables" / f"{table}.parquet", dtype_backend="numpy_nullable")
    return result if version is None else result.loc[result["definitions_version"] == version].copy()


def lake_match_ids(output_dir=OUTPUT_DIR):
    from agent.lake_storage import complete_lake_match

    root = Path(output_dir) / "lake"
    if not root.exists():
        return []
    return sorted(int(path.name) for path in root.iterdir()
                  if path.is_dir() and path.name.isdigit() and complete_lake_match(path))


def read_lake_match(match_id, table="match_player", columns=None, output_dir=OUTPUT_DIR):
    from agent.lake_storage import TABLES

    if table not in TABLES:
        raise ValueError(f"Lake table must be one of: {', '.join(TABLES)}")
    folder = Path(output_dir) / "lake" / str(int(match_id))
    filename = f"{table}.parquet"
    try:
        status = read_json(folder / "status.json", {})
        valid = (isinstance(status, dict) and isinstance(status.get("files"), dict)
                 and status.get("success") is True and status.get("match_id") == int(match_id)
                 and checksum(folder / filename) == status.get("files", {}).get(filename))
    except (OSError, ValueError, TypeError, EOFError):
        valid = False
    if not valid:
        raise ValueError(f"Lake match {match_id} has an incomplete or damaged {filename}; rerun collect_lake.py")
    # Keep every source version; nested arrays and maps are JSON strings.
    return pd.read_parquet(folder / filename, columns=columns, dtype_backend="numpy_nullable")


def iter_lake_matches(table="match_player", columns=None, output_dir=OUTPUT_DIR):
    for mid in lake_match_ids(output_dir):
        yield mid, read_lake_match(mid, table, columns, output_dir)
