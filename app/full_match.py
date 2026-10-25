"""Bounded local replay ingestion into one DuckDB database."""

import hashlib
import math
import os
from pathlib import Path

import duckdb
import requests

from . import full_match_store as store, normalize, sources
from .console_log import log


def short_error(error):
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}: {error.response.reason}"
    return " ".join(str(error).split())[:1500] or type(error).__name__


def parser_revision():
    """Identify the actual extraction/normalization code, including local fixes."""
    digest = hashlib.sha256()
    for name in ("replay.mjs", "normalize.py", "sources.py", "full_match_store.py",
                 "full_match.sql", "package-lock.json"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def local_replay(folder, match_id):
    existing = [folder / f"{match_id}{suffix}" for suffix in sources.REPLAY_SUFFIXES
                if (folder / f"{match_id}{suffix}").is_file()]
    if len(existing) > 1:
        raise ValueError(f"Multiple local replays for {match_id}; select one with --replay")
    return existing[0] if existing else folder / f"{match_id}.dem"


def candidates(config, connection, command):
    """Yield paths lazily; never enumerate the complete collection in RAM."""
    folder = Path(config.get("replay_dir") or Path(config["output_dir"]) / "replays").resolve()
    selected = config.get("match_ids", [])
    if config.get("replay_path"):
        replay = Path(config["replay_path"]).resolve()
        if selected and len(selected) != 1:
            raise ValueError("--replay accepts exactly one --match-ids value")
        match_id = selected[0] if selected else sources._match_id(replay.name.split(".", 1)[0])
        yield match_id, replay
    elif selected:
        for match_id in selected:
            yield match_id, local_replay(folder, match_id)
    else:
        # Retry/resume paths outside replay_dir too, using recorded provenance.
        if command in ("retry", "resume"):
            statuses = ("failed", "unverified_eligibility", "interrupted") if command == "retry" else ("interrupted",)
            cursor = connection.cursor()
            try:
                cursor.execute("""SELECT a.match_id, s.path FROM (
                    SELECT *, row_number() OVER (PARTITION BY match_id
                        ORDER BY started_at_utc DESC, attempt_id DESC) AS rank
                    FROM processing_attempts) a JOIN replay_sources s USING(source_id)
                    WHERE rank = 1 AND status IN ("""
                    + ",".join("?" for _ in statuses) + ") ORDER BY a.match_id", statuses)
                while rows := cursor.fetchmany(100):
                    for match_id, path in rows:
                        recorded = Path(path) if path else None
                        if recorded is not None and recorded.is_file():
                            replay = recorded
                        else:
                            # A missing-ID attempt records a .dem placeholder;
                            # the supplied file may later be .dem.zst/.dem.bz2.
                            local = local_replay(folder, match_id)
                            replay = local if local.is_file() else recorded or local
                        yield match_id, replay
            finally:
                cursor.close()
            if command == "retry":
                return
        if folder.is_dir():
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(sources.REPLAY_SUFFIXES):
                        token = entry.name.split(".", 1)[0]
                        if token.isdecimal():
                            match_id = sources._match_id(token)
                            yield match_id, local_replay(folder, match_id)


def run(config, command="ingest"):
    if command not in ("ingest", "collect", "resume", "retry", "reprocess", "init"):
        raise ValueError(f"unknown full-match command: {command}")
    root = Path(config["output_dir"]).resolve()
    interval, bound = config.get("sampling_interval_s", 20), config.get("batch_size", 1)
    if type(bound) is not int or bound < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("sampling_interval_s must be positive and finite")
    if config.get("export_csv") or config.get("retain_replays") is False:
        raise ValueError("Full-match uses DuckDB only and always preserves raw replays")
    if any((root / name).is_file() for name in ("collection_status.csv", "failures.csv")) or any(
            any((root / name).glob("*.*")) for name in ("matches", "end_match", "time_slices", "events")):
        raise ValueError("Archive legacy full-match outputs first: python -m app.archive_full_match")
    root.mkdir(parents=True, exist_ok=True)
    connection = store.connect(root / "full_match.duckdb")
    completed = skipped = failed = attempted = 0
    try:
        # DuckDB's process write lock is acquired before recovering stale attempts.
        connection.execute("""UPDATE processing_attempts SET status='interrupted',
            finished_at_utc=current_timestamp, stage='interrupted',
            reason='Previous writer stopped before finishing' WHERE status='running'""")
        if command == "init":
            log(f"Initialized {root / 'full_match.duckdb'} (no replay processing).")
            return {"complete": 0, "skipped": 0, "failed": 0}
        rebuild = command == "reprocess" or config.get("rebuild_outputs", False)
        retry = command == "retry" or config.get("retry_failures", False)
        revision = parser_revision()
        seen = set()  # Only bounded attempted IDs, not the entire collection.
        for match_id, replay in candidates(config, connection, command):
            if match_id in seen:
                continue
            previous = store.status(connection, match_id)
            recovering = (command == "resume" and previous.get("latest_status") == "interrupted"
                          or retry and previous.get("latest_status") in ("failed", "interrupted", "unverified_eligibility"))
            if (previous.get("status") == "complete" and not rebuild and not recovering
                    and not config.get("match_ids") and not config.get("replay_path")):
                # Collection scans advance past successes at any interval.
                # Only explicitly selected matches check for interval changes.
                skipped += 1
                continue
            attempt_interval = interval
            if recovering and not rebuild:
                attempt_interval = connection.execute(
                    "SELECT sampling_interval_s FROM processing_attempts WHERE attempt_id=?",
                    [previous["attempt_id"]]).fetchone()[0]
                if config.get("sampling_interval_explicit") and interval != attempt_interval:
                    raise ValueError(f"Match {match_id}: recovery uses its recorded {attempt_interval:g}s interval; "
                                     "use reprocess to choose a different interval")
            if not store.needs_processing(connection, match_id, attempt_interval,
                                          reprocess=rebuild or recovering, retry=retry):
                skipped += 1
                continue
            seen.add(match_id)
            attempt = store.start_attempt(connection, match_id, replay, attempt_interval,
                metadata_source=config["api_base"].rstrip("/") + f"/v1/matches/{match_id}/metadata",
                parser_revision=revision)
            attempted += 1
            stage = "source"
            try:
                if not replay.is_file():
                    raise ValueError(f"No local replay: {replay}; this pipeline does not download replays")
                before = replay.stat()
                stage = "metadata"
                log(f"Match {match_id}: fetching metadata; local replay {replay.name}.", flush=True)
                payload = sources.metadata(match_id, config)
                store.record_metadata(connection, attempt, payload)
                if payload.get("match_info", {}).get("match_id") != match_id:
                    raise ValueError("metadata belongs to another match")
                stage = "eligibility"
                eligible, reason = normalize.eligibility(payload)
                if eligible != "eligible":
                    status = "ineligible" if eligible == "ineligible" else "unverified_eligibility"
                    store.finish_attempt(connection, attempt, status, stage, reason)
                    skipped += eligible == "ineligible"
                    failed += eligible != "ineligible"
                    log(f"{status.upper()}: Match {match_id}: {reason}.", flush=True)
                else:
                    stage = "replay_parse"
                    log(f"  parsing once at {attempt_interval:g}s...", flush=True)
                    summary, samples, events = sources.parse_replay(replay, payload, attempt_interval)
                    after = replay.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("Replay changed during parsing; result discarded")
                    stage = "definitions"
                    try:
                        definitions = sources.definitions(summary.get("client_version"), config)
                    except (requests.RequestException, ValueError) as error:
                        definitions = []
                        summary.setdefault("warnings", []).append(f"Optional definitions: {short_error(error)}")
                    if not definitions:
                        summary.setdefault("warnings", []).append("Exact-version item/ability classification unavailable")
                    stage = "database_write"
                    store.replace_match(connection, attempt, payload, summary, samples, events, definitions)
                    completed += 1
                    log(f"SAVED: Match {match_id}; reaches end={summary.get('replay_reaches_match_end')}.", flush=True)
            except KeyboardInterrupt:
                store.finish_attempt(connection, attempt, "interrupted", stage, "Interrupted by user")
                raise
            except Exception as error:
                status = "unverified_eligibility" if isinstance(error, sources.EligibilityConflict) else "failed"
                store.finish_attempt(connection, attempt, status, stage, short_error(error))
                failed += 1
                log(f"ERROR: Match {match_id}, {stage}: {short_error(error)}", flush=True)
                if not isinstance(error, (requests.RequestException, OSError, ValueError, RuntimeError, duckdb.Error)):
                    raise
            if attempted >= bound:
                break
    finally:
        connection.close()
    log(f"Finished: {completed} complete, {skipped} skipped, {failed} failed; {attempted}/{bound} attempt bound.", flush=True)
    return {"complete": completed, "skipped": skipped, "failed": failed}
