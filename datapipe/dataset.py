import hashlib
import io
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from main import OUTPUT_DIR
from .sources import read_json, save_json

GROUPS = {
    "summaries": ("matches", "players", "game_versions", "metadata_teams", "metadata_damage",
                  "metadata_custom_stats", "metadata_gold_sources"),
    "states": ("player_state", "team_state", "metadata_player_stats", "metadata_player_paths"),
    "purchases": ("purchases", "metadata_item_records"),
    "events": ("events", "metadata_objectives", "metadata_midboss", "legacy_events", "replay_event_counts"),
}


def match_ids(output_dir=OUTPUT_DIR, lake=False):
    root = Path(output_dir) / ("lake" if lake else "raw")
    names = ("match_player.parquet", "match_salts.parquet") if lake else tuple(f"{g}.zip" for g in GROUPS)
    status_name = "status.json" if lake else "sources/status.json"
    success_key = "success" if lake else "processed_success"
    return sorted(int(p.name) for p in root.glob("*") if p.is_dir() and p.name.isdigit()
                  and read_json(p / status_name, {}).get(success_key)
                  and all((p / name).is_file() for name in names))


def read_match(match_id, table="player_state", columns=None, output_dir=OUTPUT_DIR):
    lake = table in ("match_player", "match_salts")
    folder = Path(output_dir) / ("lake" if lake else "raw") / str(int(match_id))
    status = read_json(folder / ("status.json" if lake else "sources/status.json"), {})
    if not status.get("success" if lake else "processed_success"):
        raise ValueError(f"Match {match_id} is not complete")
    if lake:
        return pd.read_parquet(folder / f"{table}.parquet", columns=columns, dtype_backend="numpy_nullable")
    group = next(g for g, tables in GROUPS.items() if table in tables)
    with ZipFile(folder / f"{group}.zip") as archive:
        return pd.read_parquet(io.BytesIO(archive.read(f"{table}.parquet")),
                               columns=columns, dtype_backend="numpy_nullable")


def iter_matches(table="player_state", columns=None, output_dir=OUTPUT_DIR):
    for mid in match_ids(output_dir, lake=table in ("match_player", "match_salts")):
        yield mid, read_match(mid, table, columns, output_dir)


def read_reference(table="heroes", version=None, output_dir=OUTPUT_DIR):
    frame = pd.read_parquet(Path(output_dir) / "tables" / f"{table}.parquet", dtype_backend="numpy_nullable")
    return frame if version is None else frame.loc[frame.definitions_version.eq(version)].copy()


def arrow_table(rows):
    # Build each column from Python values: nullable IDs must never pass through float.
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["match_id"]
    arrays = {}
    for key in columns:
        values = [row.get(key) for row in rows]
        values = [json.dumps(v, separators=(",", ":"), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                  for v in values]
        present = [v for v in values if v is not None]
        dtype = pa.uint64() if present and all(type(v) is int and v >= 0 for v in present) and max(present) >= 2**63 else None
        # Mixed scalar network fields remain exact textual values.
        if len({type(v) for v in present}) > 1 and not all(type(v) in (int, float) for v in present):
            values = [None if v is None else str(v) for v in values]
        arrays[key] = pa.array(values, type=dtype)
    return pa.table(arrays)


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow_table(rows), path, compression="zstd", compression_level=19)


def save_match(folder, tables, status):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    status = {**status, "processed_success": True, "row_counts": {k: len(v) for k, v in tables.items()}}
    # Stage four bundles; a completion marker is written only after every replacement.
    with tempfile.TemporaryDirectory(prefix=".save-", dir=folder) as temporary:
        stage = Path(temporary)
        for group, names in GROUPS.items():
            with ZipFile(stage / f"{group}.zip", "w", compression=ZIP_STORED) as archive:
                for name in names:
                    path = stage / f"{name}.parquet"
                    write_table(path, tables.get(name, []))
                    archive.write(path, path.name)
                    path.unlink()
                if group == "summaries":
                    archive.writestr("manifest.json", json.dumps(status, separators=(",", ":")))
        (folder / "sources/status.json").unlink(missing_ok=True)
        save_json(folder / "sources/status.json.gz", {"processed_success": False})
        status["files"] = {}
        for group in GROUPS:
            path = stage / f"{group}.zip"
            with path.open("rb") as source:
                status["files"][path.name] = hashlib.file_digest(source, "sha256").hexdigest()
            path.replace(folder / path.name)
    save_json(folder / "sources/status.json.gz", status)
    return status


if __name__ == "__main__":
    print(f"Detailed matches: {len(match_ids())}\nEnd-of-match matches: {len(match_ids(lake=True))}")
