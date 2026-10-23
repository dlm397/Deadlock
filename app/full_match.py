"""Collect, resume, or resample full matches from Source 2 replay files."""
import csv
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests

from . import normalize, sources
from .console_log import log
from .version_tables import write_version_tables

STATUS_COLUMNS = ("match_id", "status", "sampling_interval_s", "final_observed_game_time_s",
                  "replay_reaches_match_end", "reason", "updated_at_utc")
FAILURE_COLUMNS = ("match_id", "stage", "reason", "attempted_at_utc")
TABLES = ("matches", "end_match", "time_slices", "events")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_log(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {int(row["match_id"]): row for row in csv.DictReader(stream)}


def write_log(path, columns, records):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))
    temporary.replace(path)


def csv_export(path, table):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(table.column_names)
        for row in table.to_pylist():
            writer.writerow(value.isoformat() if isinstance(value, datetime) else value
                            for value in (row[name] for name in table.column_names))
    temporary.replace(path.with_suffix(".csv"))


def write_table(root, name, match_id, table, export_csv):
    path = root / name / f"{match_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    if export_csv:
        csv_export(path, table)
    else:
        # A previous CSV must not silently describe an older resampling run.
        path.with_suffix(".csv").unlink(missing_ok=True)


def short_error(error):
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}: {error.response.reason}"
    if isinstance(error, requests.Timeout):
        return "request timed out"
    if isinstance(error, requests.ConnectionError):
        return "connection failed"
    return " ".join(str(error).split())[:700] or type(error).__name__


def outputs_ready(config, match_id, previous):
    """Only a complete replay-backed dataset at the requested interval is ready."""
    if (previous.get("status") != "complete" or not previous.get("sampling_interval_s")
            or float(previous["sampling_interval_s"]) != config["sampling_interval_s"]):
        return False
    return all((Path(config["output_dir"]) / name / f"{match_id}.parquet").exists()
               for name in TABLES)


def discovery_exclusions(config, statuses, failures):
    """Keep candidates skipped by this run from consuming the discovery limit."""
    excluded = set() if config["retry_failures"] else set(failures)
    for mid, previous in statuses.items():
        if previous.get("status") in ("ineligible", "unverified_eligibility") and not (
                config["retry_failures"] or config["rebuild_outputs"]):
            excluded.add(mid)
        if (outputs_ready(config, mid, previous) and not config["rebuild_outputs"]
                and not (config["retry_failures"] and mid in failures)):
            excluded.add(mid)
    return excluded


def migrate_outputs(config):
    """Split existing end-match files and populate version catalogs without replay work."""
    root = Path(config["output_dir"])
    migrated, versions = 0, set()
    selected = set(config.get("match_ids", []))
    for path in sorted((root / "end_match").glob("*.parquet")):
        mid = int(path.stem)
        if selected and mid not in selected:
            continue
        players = pq.read_table(path)
        match_path = root / "matches" / path.name
        export_csv = config["export_csv"] or path.with_suffix(".csv").exists()
        if "duration_s" in players.column_names:
            # Validate the repeated values before taking one representative row.
            shared = players.select(normalize.MATCH_SCHEMA.names)
            rows = shared.to_pylist()
            if not rows or any(row != rows[0] for row in rows) or rows[0]["match_id"] != mid:
                raise ValueError(f"Match {mid}: shared metadata is empty or inconsistent; cannot split it")
            match = shared.slice(0, 1).cast(normalize.MATCH_SCHEMA)
            players = players.select(normalize.END_MATCH_SCHEMA.names).cast(normalize.END_MATCH_SCHEMA)
            # Store shared fields first so an interrupted migration can safely rerun.
            write_table(root, "matches", mid, match, export_csv)
            write_table(root, "end_match", mid, players, export_csv)
            migrated += 1
        else:
            if not match_path.exists():
                raise ValueError(f"Match {mid}: shared match table is missing; collect this ID again")
            match = pq.read_table(match_path)
            if export_csv:
                for table_path, table in ((match_path, match), (path, players)):
                    csv_export(table_path, table)
        versions.update(v for v in match["client_version"].to_pylist() if v is not None)
    for version in sorted(versions):
        try:
            items = sources.definitions(version, config)
        except (requests.RequestException, ValueError) as error:
            items = []
            log(f"NOTE: Version {version} item definitions could not be fetched: {short_error(error)}", flush=True)
        available = write_version_tables(config["supplemental_dir"], version, items)
        log(f"Version {version}: event definitions written; item definitions "
              f"{'available' if available else 'unavailable'}.", flush=True)
    log(f"Migrated {migrated} end-match files; populated definitions for {len(versions)} known versions.", flush=True)
    return {"migrated": migrated, "versions": sorted(versions)}


def run(config, command="collect"):
    """Run full replay collection using a resolved full_match config section."""
    if command not in ("collect", "retry", "reprocess", "migrate"):
        raise ValueError(f"unknown full-match command: {command}")
    if command == "migrate":
        return migrate_outputs(config)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    status_path, failure_path = root / "collection_status.csv", root / "failures.csv"
    statuses, failures = read_log(status_path), read_log(failure_path)
    # Header-only logs also make a fresh run's state explicit.
    write_log(status_path, STATUS_COLUMNS, statuses)
    write_log(failure_path, FAILURE_COLUMNS, failures)
    selected = config["match_ids"]
    automatic = False
    if not selected:
        if command == "retry":
            selected = sorted(failures)
        elif command == "reprocess":
            folder = root / "replays"
            selected = sorted({int(p.name.split(".", 1)[0]) for suffix in sources.REPLAY_SUFFIXES
                               for p in folder.glob(f"*{suffix}") if p.name.split(".", 1)[0].isdigit()})
        else:
            excluded = discovery_exclusions(config, statuses, failures)
            log(f"Discovering up to {config['discovery_limit']} matches using recorded replay salts via MCP...", flush=True)
            selected = sources.discover(config, exclude=excluded)
            log(f"  selected {len(selected)} candidates; previously handled matches do not count toward the limit", flush=True)
            automatic = True
    retry = command == "retry" or config["retry_failures"]
    rebuild = command == "reprocess" or config["rebuild_outputs"]
    completed, skipped, failed = 0, 0, 0
    for mid in selected:
        previous = statuses.get(mid, {})
        paths = {name: root / name / f"{mid}.parquet" for name in TABLES}
        if outputs_ready(config, mid, previous) and not rebuild and mid not in failures:
            if config["export_csv"]:
                for name in TABLES:
                    path = paths[name]
                    if not path.with_suffix(".csv").exists():
                        csv_export(path, pq.read_table(path))
            log(f"SKIP: Match {mid}: complete at {config['sampling_interval_s']:g}s.", flush=True)
            skipped += 1
            continue
        if mid in failures and not retry:
            log(f"SKIP: Match {mid} has an active failure; use retry.", flush=True)
            skipped += 1
            continue
        if previous.get("status") in ("ineligible", "unverified_eligibility") and not (retry or rebuild):
            log(f"SKIP: Match {mid} is {previous['status']}: {previous.get('reason', '')}", flush=True)
            skipped += 1
            continue
        existing_complete = previous.get("status") == "complete" and all(path.exists() for path in paths.values())
        log(f"Match {mid}: retrieving metadata...", flush=True)
        stage, end_written = "metadata", False
        record = dict.fromkeys(STATUS_COLUMNS)
        record.update(match_id=mid, status="failed", reason="", updated_at_utc=now())
        try:
            payload = sources.metadata(mid, config)
            stage = "eligibility"
            eligibility, reason = normalize.eligibility(payload)
            if eligibility != "eligible":
                record.update(status="ineligible" if eligibility == "ineligible" else "unverified_eligibility", reason=reason)
                # Never leave newly disqualified matches in eligible research outputs.
                for path in paths.values():
                    path.unlink(missing_ok=True)
                    path.with_suffix(".csv").unlink(missing_ok=True)
                statuses[mid] = record
                if eligibility == "ineligible":
                    failures.pop(mid, None)
                else:
                    failures[mid] = dict(match_id=mid, stage=stage, reason=reason, attempted_at_utc=now())
                write_log(status_path, STATUS_COLUMNS, statuses)
                write_log(failure_path, FAILURE_COLUMNS, failures)
                label = "SKIP" if eligibility == "ineligible" else "ERROR"
                log(f"{label}: Match {mid}: {reason}.", flush=True)
                skipped += 1
                continue
            stage = "end_match"
            # Preserve known header metadata if a later rebuild fails.
            header_path = paths["matches"] if paths["matches"].exists() else paths["end_match"]
            known_header = pq.read_table(header_path).to_pylist()[0] if header_path.exists() else None
            write_table(root, "matches", mid, normalize.match(payload, known_header), config["export_csv"])
            write_table(root, "end_match", mid, normalize.end_match(payload), config["export_csv"])
            end_written = True
            record["status"] = "replay_pending"
            statuses[mid] = ({**previous, "reason": f"Rebuild in progress at {config['sampling_interval_s']:g}s; existing outputs retained",
                              "updated_at_utc": now()} if existing_complete else record)
            write_log(status_path, STATUS_COLUMNS, statuses)
            stage = "replay_lookup"
            replay = sources.find_replay(mid, root)
            if replay is None:
                if command == "reprocess":
                    raise ValueError("no retained replay; use collect to download it")
                stage = "replay_download"
                log(f"  downloading replay...", flush=True)
                replay = sources.download_replay(mid, config)
            else:
                log("  using retained replay", flush=True)
            stage = "replay_parse"
            log(f"  parsing at {config['sampling_interval_s']:g}s...", flush=True)
            summary, samples, replay_events = sources.parse_replay(replay, payload, config["sampling_interval_s"])
            stage = "definitions"
            warnings = list(summary.get("warnings", []))
            try:
                items = sources.definitions(summary.get("client_version"), config)
            except (requests.RequestException, ValueError) as error:
                items = []
                warnings.append(f"optional item definitions could not be fetched: {short_error(error)}")
            if not items:
                warnings.append("exact-version item definitions unavailable; item/ability classification is limited")
            write_version_tables(config["supplemental_dir"], summary.get("client_version"), items)
            stage = "normalize"
            tables = {"matches": normalize.match(payload, summary),
                      "end_match": normalize.end_match(payload),
                      "time_slices": normalize.time_slices(payload, samples),
                      "events": normalize.events(payload, replay_events, items)}
            if (not tables["time_slices"].num_rows or
                    tables["time_slices"]["observed_time_s"].null_count == tables["time_slices"].num_rows):
                raise ValueError("no verified replay gameplay observations")
            stage = "write_outputs"
            for name, table in tables.items():
                write_table(root, name, mid, table, config["export_csv"])
            record.update(status="complete", sampling_interval_s=config["sampling_interval_s"],
                          final_observed_game_time_s=summary.get("final_observed_game_time_s"),
                          replay_reaches_match_end=summary.get("replay_reaches_match_end"),
                          reason="; ".join(dict.fromkeys(warnings)), updated_at_utc=now())
            statuses[mid] = record
            failures.pop(mid, None)
            write_log(status_path, STATUS_COLUMNS, statuses)
            write_log(failure_path, FAILURE_COLUMNS, failures)
            if not config["retain_replays"]:
                replay.unlink()
            completed += 1
            log(f"SAVED: Match {mid}: {tables['time_slices'].num_rows} states, "
                  f"{tables['events'].num_rows} event participant rows; "
                  f"reaches end={record['replay_reaches_match_end']}.", flush=True)
            if warnings:
                log("  NOTE: " + record["reason"], flush=True)
        except Exception as error:
            # Expected per-match problems continue; coding errors keep their traceback and stop.
            expected = isinstance(error, (requests.RequestException, OSError, ValueError, RuntimeError))
            reason = short_error(error)
            record.update(status="replay_pending" if end_written else "failed", reason=reason, updated_at_utc=now())
            if stage == "metadata" and isinstance(error, ValueError) or isinstance(error, sources.EligibilityConflict):
                record["status"] = "unverified_eligibility"
            if record["status"] == "unverified_eligibility":
                for path in paths.values():
                    path.unlink(missing_ok=True)
                    path.with_suffix(".csv").unlink(missing_ok=True)
            elif existing_complete and stage not in ("write_outputs", "eligibility"):
                record = {**previous, "reason": f"Rebuild at {config['sampling_interval_s']:g}s failed; existing outputs retained. {reason}",
                          "updated_at_utc": now()}
            failures[mid] = dict(match_id=mid, stage=stage, reason=reason, attempted_at_utc=now())
            statuses[mid] = record
            write_log(status_path, STATUS_COLUMNS, statuses)
            write_log(failure_path, FAILURE_COLUMNS, failures)
            log(f"ERROR: Match {mid} {stage.replace('_', ' ')} failed: {reason}.", flush=True)
            failed += 1
            if not expected:
                raise
        if automatic and completed >= config["discovery_count"]:
            break
    log(f"Finished: {completed} complete, {skipped} skipped, {failed} failed.", flush=True)
    return {"complete": completed, "skipped": skipped, "failed": failed}

