import shutil
from pathlib import Path

from .io import checksum, save_json


def move_verified(source, target, workspace):
    source, target, workspace = Path(source).resolve(), Path(target).resolve(), Path(workspace).resolve()
    source.relative_to(workspace)
    target.relative_to(workspace)
    if not source.exists():
        return
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in list(source.iterdir()):
            move_verified(child, target / child.name, workspace)
        source.rmdir()
    else:
        digest = checksum(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if checksum(target) != digest:
                raise FileExistsError(f"Different files at {source} and {target}")
        else:
            shutil.copy2(source, target)
        if checksum(target) != digest:
            raise ValueError(f"Migration checksum mismatch: {source}")
        source.unlink()


def migrate(root="data"):
    root = Path(root).resolve()
    workspace = Path(__file__).resolve().parent.parent
    root.relative_to(workspace)
    backup = workspace / "agent/legacy_data"
    moved = []
    for kind, dest in (("api", "api"), ("replay", "replay")):
        for path in list((root / "raw" / kind).glob("*")):
            target = root / "raw" / path.name / "sources" / dest if path.name.isdigit() else root / "tables/sources" / path.name
            move_verified(path, target, workspace)
            moved.append(str(target.relative_to(root)))
        old = root / "raw" / kind
        if old.exists():
            old.rmdir()
    for path in list((root / "raw/demos").glob("*")):
        mid = path.name.split(".")[0]
        if not mid.isdigit():
            raise ValueError(f"Unrecognized demo filename: {path}")
        target = root / "raw" / mid / "sources" / path.name.replace(mid, "replay", 1)
        move_verified(path, target, workspace)
    if (root / "raw/demos").exists():
        (root / "raw/demos").rmdir()
    for path in list((root / "schema").glob("*")):
        target = root / "raw" / path.name / "sources/schema" if path.name.isdigit() else root / "tables/sources/schema" / path.name
        move_verified(path, target, workspace)
    if (root / "schema").exists():
        (root / "schema").rmdir()
    move_verified(root / "static", root / "tables/sources", workspace)
    for path in list((root / "logs").glob("*")):
        mid = path.stem.removeprefix("parser-")
        target = root / "raw" / mid / "sources/parser.log" if mid.isdigit() else backup / "logs" / path.name
        move_verified(path, target, workspace)
    if (root / "logs").exists():
        (root / "logs").rmdir()
    for name in ("processed", "analysis", "manifest.json", "summary.txt", "INITIAL_REPORT.md"):
        move_verified(root / name, backup / name, workspace)
    (root / "processed").mkdir(exist_ok=True)
    save_json(root / "tables/migration.json", {"source_files_sha256_verified": True, "moved": moved,
                                              "old_derived_outputs": str(backup)})


if __name__ == "__main__":
    migrate()
