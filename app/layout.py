"""One-time, local-only relocation of the legacy data layout without overwrites."""

from datetime import datetime, timezone
import hashlib
import json

from .config import DEFAULT_CONFIG, data_paths
from .console_log import log


def inventory(path):
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    return {str(file.relative_to(path)) if path.is_dir() else path.name:
            {"bytes": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns,
             **({"sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                if file.suffix == ".csv" else {})}
            for file in files if file.is_file()}


def migration_plan(paths):
    root, full, end, supplemental = (paths[name] for name in ("data_dir", "full_match", "end_match", "supplemental"))
    pairs = []
    # Move the old flat API player table before recreating the lake dataset root.
    old_end = root / "end_match"
    if old_end.exists() and any(p.stem.isdecimal() for p in old_end.glob("*.parquet")):
        pairs.append((old_end, full / "end_match"))
    for name in ("matches", "time_slices", "events", "replays", "collection_status.csv", "failures.csv"):
        pairs.append((root / name, full / name))
    pairs.extend(((root / "version", supplemental / "version"),
                  (root / "splits.csv", supplemental / "splits.csv"),
                  (root / "lake_samples", end / "samples")))
    plan = []
    for source, target in pairs:
        source, target = source.resolve(), target.resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(root) or source == root or target == root:
            raise ValueError("Migration paths must remain within the resolved data directory")
        if source == target:
            continue
        if source.exists():
            if target.exists():
                raise ValueError(f"Refusing to overwrite migration destination: {target}")
            plan.append((source, target))
    for index, (_, target) in enumerate(plan):
        for source, _ in plan[index:]:
            if target.is_relative_to(source):
                raise ValueError(f"Migration destination lies inside a source that has not moved yet: {target}")
    return plan


def migrate_layout(config_path=DEFAULT_CONFIG, dry_run=False):
    paths = data_paths(config_path)
    root = paths["data_dir"]
    plan = migration_plan(paths)
    for source, target in plan:
        log(f"{'PLAN' if dry_run else 'MOVE'}: {source} -> {target}")
    if dry_run:
        return {"moves": len(plan), "dry_run": True}
    # Verify every path before mutation. Same-volume renames avoid copying replays.
    before = {str(source): inventory(source) for source, _ in plan}
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "moves": []}
    for source, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        after = inventory(target)
        if after != before[str(source)]:
            raise RuntimeError(f"Migration inventory verification failed for {target}; no files were deleted")
        report["moves"].append({"source": str(source.relative_to(root)), "destination": str(target.relative_to(root)),
                                "files": len(after), "bytes": sum(row["bytes"] for row in after.values()),
                                "verified": True})
    for name in ("full_match", "end_match", "supplemental"):
        paths[name].mkdir(parents=True, exist_ok=True)
    if plan:
        report_dir = paths["supplemental"] / "migrations"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"layout_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log(f"Verified {len(plan)} moves; migration record: {report_path}")
    else:
        log("Data layout is already current; no files moved.")
    return report
