"""Archive legacy full-match outputs without reading them into the new database."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, data_paths
from .sources import REPLAY_SUFFIXES, replay_suffix


PROCESSED_DIRS = ("matches", "end_match", "time_slices", "events")
PROCESSED_LOGS = ("collection_status.csv", "failures.csv")


def fingerprint(path, *, digest=False):
    stat = path.stat()
    result = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        result["sha256"] = hasher.hexdigest()
    return result


def archive(config_path=DEFAULT_CONFIG, *, dry_run=False):
    paths = data_paths(config_path)
    root = paths["full_match"].resolve()
    archive_root = (paths["data_dir"] / "archives").resolve()
    if archive_root.is_relative_to(root) or not archive_root.is_relative_to(paths["data_dir"]):
        raise ValueError("Archive must be outside full_match and inside data_dir")
    # Inspect everything before moving anything. Unknown files are never guessed at.
    processed, raw = {}, {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing linked or out-of-root path: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        with path.open("rb") as stream:
            magic = stream.read(4)
        if path.name.endswith(REPLAY_SUFFIXES):
            replay_suffix(magic)
            raw[str(relative)] = fingerprint(path)
        elif ((relative.parts[0] in PROCESSED_DIRS and path.suffix in (".csv", ".parquet"))
              or str(relative) in PROCESSED_LOGS):
            if magic.startswith((b"PBDE", b"BZh", b"\x28\xb5\x2f\xfd")):
                raise ValueError(f"Replay disguised as processed output: {path}")
            if path.suffix == ".parquet" and magic != b"PAR1":
                raise ValueError(f"Not a Parquet file: {path}")
            processed[str(relative)] = fingerprint(path, digest=True)
        elif relative.parts[0] != "replays" or path.suffix != ".part":
            raise ValueError(f"Unrecognized file; inspect manually before archiving: {path}")
        else:
            raw[str(relative)] = fingerprint(path)
    print(f"Inspected {len(processed)} processed files and {len(raw)} retained raw files.")
    if dry_run or not processed:
        return {"processed_files": len(processed), "raw_files": len(raw), "dry_run": dry_run}
    destination = archive_root / f"full_match_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    if destination.exists():
        raise ValueError(f"Archive already exists: {destination}")
    destination.mkdir(parents=True)
    report = {"source": str(root), "archive": str(destination), "processed_files": processed,
              "retained_raw_files": raw, "verified": False}
    manifest = destination / "archive_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Individual same-volume renames preserve structure, bytes, and timestamps;
    # even an unexpected replay inside a processed directory stays in place.
    for relative in processed:
        source, target = (root / relative).resolve(), (destination / relative).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(destination):
            raise ValueError("Archive path escaped its verified root")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"Refusing to overwrite {target}")
        source.rename(target)
    for relative, expected in processed.items():
        if fingerprint(destination / relative, digest=True) != expected:
            raise RuntimeError(f"Archive verification failed: {relative}; nothing was deleted")
    for relative, expected in raw.items():
        if fingerprint(root / relative) != expected:
            raise RuntimeError(f"Raw inventory changed: {relative}; nothing was deleted")
    report["verified"] = True
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified archive: {destination}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    archive(args.config, dry_run=args.dry_run)
