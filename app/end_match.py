"""Bounded, resumable ingestion of public match_player Parquet lake shards.

No replay downloads or metadata API queries. The DuckDB database retains the
latest locally observed player rows; it is NOT a canonical snapshot of the lake.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import duckdb
import requests

from .console_log import log

DEFAULT_MANIFEST = "https://data.deadlock-api.com/v1/manifest.json"
REQUIRED_COLUMNS = {
    "match_id", "player_slot", "account_id", "hero_id", "team", "start_time",
    "duration_s", "game_mode", "match_mode", "match_outcome", "winning_team",
    "created_at",
}
LIMITATION = (
    "Bounded subset of manifest shards, not a canonical or globally latest lake "
    "snapshot. Versions are deduplicated by (match_id, player_slot), keeping the "
    "greatest created_at observed locally. Only complete, consistent 6v6 match "
    "versions are exported. Collection is biased toward recent, small shards."
)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _settings(config):
    settings = dict(config)
    settings["output_dir"] = Path(settings.get("output_dir", "data/end_match")).resolve()
    settings["match_limit"] = _positive_int(settings.get("match_limit", 300), "match_limit")
    settings["max_files"] = _positive_int(settings.get("max_files", 1), "max_files")
    size = settings.get("max_download_mib", 32)
    if isinstance(size, bool) or not isinstance(size, (float, int)) or not math.isfinite(size) or size <= 0:
        raise ValueError("max_download_mib must be a finite positive number")
    settings["max_download_bytes"] = int(size * 1024 * 1024)
    ids = settings.get("match_ids") or []
    if not isinstance(ids, (list, tuple)):
        raise ValueError("match_ids must be a list of integers")
    settings["match_ids"] = list(dict.fromkeys(_positive_int(i, "match_id") for i in ids))
    if any(i > 2**64 - 1 for i in settings["match_ids"]):
        raise ValueError("match_id exceeds the unsigned 64-bit range")
    return settings


def _load_state(root):
    path = root / "state.json"
    if not path.exists():
        return {"format_version": 1, "files": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("format_version") != 1 or not isinstance(state.get("files"), dict):
        raise ValueError(f"Unsupported or invalid collection state: {path}")
    for key, entry in state["files"].items():
        if not isinstance(entry, dict) or entry.get("key") != key:
            raise ValueError(f"Invalid collection state entry: {key}")
        entry.setdefault("ingested", entry.get("status") == "imported")
        _cache_path(root, entry)
    return state


def _cache_path(root, entry):
    path = (root / entry["local_path"]).resolve()
    if not path.is_relative_to((root / "shards").resolve()):
        raise ValueError("Shard cache path must remain inside output_dir/shards")
    return path


def _load_manifest(location, root):
    location = str(location)
    if urlparse(location).scheme in {"http", "https"}:
        response = requests.get(location, timeout=(15, 90))
        response.raise_for_status()
        manifest = response.json()
    else:
        manifest = json.loads(Path(location).read_text(encoding="utf-8"))
    table = manifest.get("tables", {}).get("match_player", {})
    if manifest.get("format_version") != 1 or table.get("status") != "ready":
        raise ValueError("Expected a version-1 manifest with a ready match_player table")
    if table.get("partition_expr") != "intDiv(match_id, 1000000)":
        raise ValueError("Unsupported match_player partition expression")
    if table.get("watermark") != "created_at" or not isinstance(table.get("files"), list):
        raise ValueError("Expected match_player files with created_at versioning")
    canonical = json.dumps(manifest, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    path = root / "manifests" / f"{digest}.json"
    if not path.exists():
        _write_json(path, manifest)
    return manifest, digest, str(path.relative_to(root))


def _candidate_files(manifest, settings):
    partitions = {str(i // 1_000_000) for i in settings["match_ids"]}
    candidates = []
    for entry in manifest["tables"]["match_player"]["files"]:
        if not isinstance(entry.get("key"), str) or not entry["key"].endswith(".parquet"):
            continue
        size = entry.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= settings["max_download_bytes"]:
            continue
        if partitions:
            if "partition" in entry and str(entry["partition"]) not in partitions:
                continue
            rows = entry.get("rows_by_partition")
            if rows is not None and not any(rows.get(part, 0) > 0 for part in partitions):
                continue
        candidates.append(entry)
    # Among fitting files prefer useful rosters, then latest ingestion watermark.
    enough_rows = 12 * settings["match_limit"]
    return sorted(candidates, key=lambda entry: (
        entry.get("rows", 0) >= enough_rows, entry.get("hi", 0),
        entry.get("built_at", ""), entry["key"],
    ), reverse=True)


def _source_url(public_url, key):
    if urlparse(str(public_url)).scheme in {"http", "https"}:
        url = urljoin(str(public_url).rstrip("/") + "/", key)
        if urlparse(url).netloc != urlparse(str(public_url)).netloc:
            raise ValueError("Manifest shard key changes the public URL host")
        return url
    return str((Path(public_url) / key).resolve())


def _download(entry, path, remaining_bytes):
    """Charge bytes as they arrive, including failed/oversized responses."""
    expected = entry["metadata"]["bytes"]
    if expected > remaining_bytes:
        raise ValueError("Shard exceeds remaining download budget")
    partial = path.with_suffix(".partial")
    entry["last_attempt_bytes"] = 0
    try:
        url = entry["url"]
        with partial.open("wb") as output:
            if urlparse(url).scheme in {"http", "https"}:
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        entry["last_attempt_bytes"] += len(chunk)
                        if entry["last_attempt_bytes"] > min(expected, remaining_bytes):
                            raise ValueError("Download exceeds advertised size or remaining budget")
                        output.write(chunk)
            else:
                with Path(url).open("rb") as source:
                    while chunk := source.read(64 * 1024):
                        entry["last_attempt_bytes"] += len(chunk)
                        if entry["last_attempt_bytes"] > min(expected, remaining_bytes):
                            raise ValueError("Source exceeds advertised size or remaining budget")
                        output.write(chunk)
        if entry["last_attempt_bytes"] != expected:
            raise ValueError(f"Shard size mismatch: expected {expected}, received {entry['last_attempt_bytes']}")
        entry["sha256"] = _sha256(partial)
        partial.replace(path)
        entry["status"] = "downloaded"
    finally:
        partial.unlink(missing_ok=True)


def _has_table(connection, name):
    return connection.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0] > 0


def _import_shard(connection, path, entry):
    if entry.get("sha256") and _sha256(path) != entry["sha256"]:
        raise ValueError(f"Cached shard checksum mismatch: {path}")
    connection.read_parquet(str(path)).create_view("incoming_file", replace=True)
    columns = {row[0] for row in connection.execute("DESCRIBE incoming_file").fetchall()}
    if not REQUIRED_COLUMNS <= columns:
        raise ValueError(f"Missing source columns: {sorted(REQUIRED_COLUMNS - columns)}")
    if {"_lake_source_key", "_lake_manifest_sha256"} & columns:
        raise ValueError("Source contains reserved provenance column names")
    connection.execute("BEGIN TRANSACTION")
    try:
        # Equal-version copies must be identical; arbitrarily choosing one can
        # conceal conflicting hero, team, or outcome data.
        connection.execute("CREATE OR REPLACE TEMP TABLE unique_incoming AS SELECT DISTINCT * FROM incoming_file")
        conflict = connection.execute("""
            SELECT match_id, player_slot FROM unique_incoming
            GROUP BY match_id, player_slot, created_at HAVING count(*) > 1 LIMIT 1
        """).fetchone()
        if conflict:
            raise ValueError(f"Conflicting rows at identical created_at for match/player {conflict}")
        connection.execute("""
            CREATE OR REPLACE TEMP TABLE incoming_rows AS
            SELECT *, ?::VARCHAR AS _lake_source_key, ?::VARCHAR AS _lake_manifest_sha256
            FROM unique_incoming
            WHERE match_id IS NOT NULL AND player_slot IS NOT NULL
            QUALIFY row_number() OVER (
                PARTITION BY match_id, player_slot ORDER BY created_at DESC NULLS LAST
            ) = 1
        """, [entry["key"], entry["manifest_sha256"]])
        if not _has_table(connection, "raw_match_player"):
            connection.execute("CREATE TABLE raw_match_player AS SELECT * FROM incoming_rows")
        else:
            old_schema = connection.execute("DESCRIBE raw_match_player").fetchall()
            new_schema = connection.execute("DESCRIBE incoming_rows").fetchall()
            if [(row[0], row[1]) for row in old_schema] != [(row[0], row[1]) for row in new_schema]:
                raise ValueError("Lake schema changed; use a separate output directory to inspect the new schema")
            connection.execute("""
                CREATE OR REPLACE TEMP VIEW existing_source AS
                SELECT * EXCLUDE (_lake_source_key, _lake_manifest_sha256) FROM raw_match_player
            """)
            conflict = connection.execute("""
                SELECT incoming.match_id, incoming.player_slot
                FROM unique_incoming incoming JOIN existing_source existing
                  ON incoming.match_id = existing.match_id
                 AND incoming.player_slot = existing.player_slot
                 AND incoming.created_at IS NOT DISTINCT FROM existing.created_at
                WHERE incoming IS DISTINCT FROM existing LIMIT 1
            """).fetchone()
            if conflict:
                raise ValueError(f"Conflicting rows at identical created_at for match/player {conflict}")
            connection.execute("""
                DELETE FROM raw_match_player existing USING incoming_rows incoming
                WHERE existing.match_id = incoming.match_id
                  AND existing.player_slot = incoming.player_slot
                  AND (incoming.created_at > existing.created_at OR existing.created_at IS NULL)
            """)
            connection.execute("""
                INSERT INTO raw_match_player
                SELECT incoming.* FROM incoming_rows incoming
                WHERE NOT EXISTS (
                    SELECT 1 FROM raw_match_player existing
                    WHERE existing.match_id = incoming.match_id
                      AND existing.player_slot = incoming.player_slot
                )
            """)
        connection.execute(Path(__file__).with_suffix(".sql").read_text(encoding="utf-8"))
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    entry["status"] = "imported"
    entry["ingested"] = True
    entry.pop("error", None)


def _select_matches(connection, settings, remaining):
    if not _has_table(connection, "raw_match_player") or remaining <= 0:
        return 0
    requested = settings["match_ids"]
    query = """
        SELECT match_id FROM eligible_matches
        WHERE match_id NOT IN (SELECT match_id FROM selected_matches)
    """
    parameters = []
    if requested:
        query += " AND match_id IN (SELECT unnest(?::UBIGINT[]))"
        parameters.append(requested)
    query += " ORDER BY match_id DESC LIMIT ?"
    parameters.append(remaining)
    selected = connection.execute(query, parameters).fetchall()
    if selected:
        connection.executemany("INSERT INTO selected_matches VALUES (?) ON CONFLICT DO NOTHING", selected)
    return len(selected)


def _export(connection, root, export_csv):
    counts = {"matches": 0, "player_rows": 0, "raw_player_rows": 0}
    if not _has_table(connection, "raw_match_player"):
        return counts
    for table in ("match_player", "matches"):
        order = "match_id DESC, player_slot" if table == "match_player" else "match_id DESC"
        for extension in (["parquet", "csv"] if export_csv else ["parquet"]):
            target = root / "tables" / f"{table}.{extension}"
            temporary = target.with_suffix(f".{extension}.tmp")
            options = "FORMAT PARQUET, COMPRESSION ZSTD" if extension == "parquet" else "FORMAT CSV, HEADER TRUE"
            connection.execute(f"COPY (SELECT * FROM {table} ORDER BY {order}) TO ? ({options})", [str(temporary)])
            temporary.replace(target)
        if not export_csv:
            (root / "tables" / f"{table}.csv").unlink(missing_ok=True)
    counts["matches"] = connection.execute("SELECT count(*) FROM matches").fetchone()[0]
    counts["player_rows"] = connection.execute("SELECT count(*) FROM match_player").fetchone()[0]
    counts["raw_player_rows"] = connection.execute("SELECT count(*) FROM raw_match_player").fetchone()[0]
    counts["observed_matches"] = connection.execute("SELECT count(DISTINCT match_id) FROM raw_match_player").fetchone()[0]
    counts["eligible_matches"] = connection.execute("SELECT count(*) FROM eligible_matches").fetchone()[0]
    counts["selected_matches_not_currently_complete"] = connection.execute("""
        SELECT count(*) FROM selected_matches WHERE match_id NOT IN (SELECT match_id FROM eligible_matches)
    """).fetchone()[0]
    counts["latest_observed_created_at"] = connection.execute("SELECT CAST(max(created_at) AS VARCHAR) FROM raw_match_player").fetchone()[0]
    return counts


def run(config, command="collect"):
    """Collect bounded shards, retry failed pinned shards, or reprocess offline.

    match_limit caps NEW selected matches in this invocation, not existing data.
    reprocess never fetches a manifest or downloads a shard. rebuild_outputs
    reimports retained shards before collecting; it never clears stored data.
    """
    if command not in {"collect", "retry", "reprocess"}:
        raise ValueError(f"Unsupported end-match command: {command}")
    settings = _settings(config)
    root = settings["output_dir"]
    for directory in (root, root / "shards", root / "manifests", root / "tables", root / "runs"):
        directory.mkdir(parents=True, exist_ok=True)
    state = _load_state(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary = {"command": command, "run_id": timestamp, "new_matches": 0,
               "downloaded_bytes": 0, "download_attempts": 0, "errors": [],
               "limitation": LIMITATION}
    connection = duckdb.connect(str(root / "end_match.duckdb"))
    connection.execute("SET TimeZone = 'UTC'")
    try:
        recover_database = state.get("recovering_database", False) or (
            not _has_table(connection, "raw_match_player")
            and any(entry.get("ingested") for entry in state["files"].values())
        )
        if recover_database:
            unavailable = [entry["key"] for entry in state["files"].values()
                           if entry.get("ingested") and not _cache_path(root, entry).exists()]
            if unavailable:
                raise ValueError(
                    "DuckDB data is missing and previously imported shards are not retained. "
                    "Restore end_match.duckdb (and its state.json) from backup, restore the "
                    "missing cached shards, or choose a new output directory. Missing: "
                    + ", ".join(unavailable[:3])
                )
            log("End-match: recovering the missing database from retained shards")
            state["recovering_database"] = True
            _write_json(root / "state.json", state)
        # Recover interrupted imports and support an entirely offline reprocess.
        for entry in state["files"].values():
            path = _cache_path(root, entry)
            if path.exists() and (command == "reprocess" or settings.get("rebuild_outputs", False)
                                  or recover_database or entry["status"] == "downloaded"):
                try:
                    _import_shard(connection, path, entry)
                except Exception as error:
                    if recover_database and entry.get("ingested"):
                        raise ValueError(f"Cannot recover the database from {path}: {error}. Restore the cached shard or database from backup.") from error
                    entry.update(status="failed", error=str(error))
                    summary["errors"].append({"key": entry["key"], "error": str(error)})
                _write_json(root / "state.json", state)
        if recover_database and _has_table(connection, "selected_matches"):
            previous_selection = state.get("selected_match_ids", [])
            if previous_selection:
                connection.executemany("INSERT INTO selected_matches VALUES (?) ON CONFLICT DO NOTHING", [(i,) for i in previous_selection])
            state.pop("recovering_database", None)
            _write_json(root / "state.json", state)
        summary["new_matches"] += _select_matches(connection, settings, settings["match_limit"])

        def needs_more():
            if summary["new_matches"] >= settings["match_limit"]:
                return False
            if settings["match_ids"] and _has_table(connection, "raw_match_player"):
                found = connection.execute("SELECT count(*) FROM matches WHERE match_id IN (SELECT unnest(?::UBIGINT[]))", [settings["match_ids"]]).fetchone()[0]
                if found == len(settings["match_ids"]):
                    return False
            return True

        candidates = []
        if command == "collect" and needs_more():
            manifest, digest, manifest_path = _load_manifest(settings.get("manifest_url", DEFAULT_MANIFEST), root)
            summary.update(manifest_sha256=digest, manifest_version=manifest.get("version"))
            for metadata in _candidate_files(manifest, settings):
                key = metadata["key"]
                entry = state["files"].get(key)
                if entry:
                    if entry["status"] != "pending" and not (entry["status"] == "failed" and settings.get("retry_failures", False)):
                        continue
                else:
                    filename = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".parquet"
                    entry = {"key": key, "metadata": metadata, "manifest_sha256": digest,
                             "manifest_path": manifest_path, "url": _source_url(manifest["public_url"], key),
                             "local_path": f"shards/{filename}", "status": "pending"}
                candidates.append(entry)
        elif command == "retry":
            candidates = [entry for entry in state["files"].values() if entry["status"] in {"failed", "pending"}]

        for entry in candidates:
            if (command != "retry" and not needs_more()) or summary["download_attempts"] >= settings["max_files"]:
                break
            remaining = settings["max_download_bytes"] - summary["downloaded_bytes"]
            path = _cache_path(root, entry)
            cached = path.exists() and entry.get("sha256") == _sha256(path)
            if not cached and entry["metadata"]["bytes"] > remaining:
                continue
            state["files"][entry["key"]] = entry
            _write_json(root / "state.json", state)
            try:
                if not cached:
                    log(f"End-match: downloading {entry['key']} ({entry['metadata']['bytes'] / 1024**2:.2f} MiB)")
                    summary["download_attempts"] += 1
                    try:
                        _download(entry, path, remaining)
                    finally:
                        summary["downloaded_bytes"] += entry.get("last_attempt_bytes", 0)
                    _write_json(root / "state.json", state)
                _import_shard(connection, path, entry)
                summary["new_matches"] += _select_matches(connection, settings, settings["match_limit"] - summary["new_matches"])
                if not settings.get("retain_shards", True):
                    path.unlink(missing_ok=True)
            except Exception as error:
                entry.update(status="failed", error=str(error))
                summary["errors"].append({"key": entry["key"], "error": str(error)})
                log(f"End-match: failed {entry['key']}: {error}")
            _write_json(root / "state.json", state)
        summary.update(_export(connection, root, settings.get("export_csv", False)))
        if settings["match_ids"]:
            found = set()
            if _has_table(connection, "raw_match_player"):
                found = {row[0] for row in connection.execute("SELECT match_id FROM matches").fetchall()}
            summary["requested_ids_not_exported"] = [i for i in settings["match_ids"] if i not in found]
        summary["retained_shards"] = sum(_cache_path(root, entry).exists() for entry in state["files"].values())
        summary["imported_shards"] = sum(bool(entry.get("ingested")) for entry in state["files"].values())
        summary["covered_manifest_sha256"] = sorted({entry["manifest_sha256"] for entry in state["files"].values() if entry.get("ingested")})
        if _has_table(connection, "selected_matches"):
            state["selected_match_ids"] = [row[0] for row in connection.execute("SELECT match_id FROM selected_matches ORDER BY match_id").fetchall()]
        summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
        _write_json(root / "runs" / f"{timestamp}.json", summary)
        _write_json(root / "summary.json", summary)
        _write_json(root / "state.json", state)
        log(f"End-match: added {summary['new_matches']} matches; {summary['matches']} total, {summary['downloaded_bytes'] / 1024**2:.2f} MiB transferred")
        if needs_more():
            log("End-match: target not filled within the selected shard/download budget; see summary.json for coverage")
        return summary
    finally:
        connection.close()
