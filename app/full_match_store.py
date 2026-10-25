"""The sole DuckDB writer for replay-derived data; one transaction per match.

Parser intermediates and normalization are bounded to one replay. Shared static
identities are stored once, while sparse events have dedicated typed tables.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

import duckdb

from . import normalize

SCHEMA_VERSION = 1
MATCH_TABLES = (
    "kill_assists", "hero_kills", "item_events", "objective_events", "midboss_events",
    "boss_notifications", "rejuv_events", "global_snapshots", "player_snapshots",
    "match_bans", "match_players", "matches",
)


def now():
    return datetime.now(timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def connect(database):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    try:
        connection.execute("SET TimeZone='UTC'")
        connection.execute(Path(__file__).with_name("full_match.sql").read_text(encoding="utf-8"))
        versions = connection.execute("SELECT version FROM schema_info").fetchall()
        if versions != [(SCHEMA_VERSION,)]:
            raise ValueError(f"unsupported full-match schema version: {versions}")
    except BaseException:
        connection.close()
        raise
    return connection


def status(connection, match_id):
    cursor = connection.execute("SELECT * REPLACE(started_at_utc::VARCHAR AS started_at_utc, "
                                "finished_at_utc::VARCHAR AS finished_at_utc) "
                                "FROM processing_status WHERE match_id=?", [match_id])
    row = cursor.fetchone()
    return dict(zip([d[0] for d in cursor.description], row)) if row else {}


def needs_processing(connection, match_id, interval, *, reprocess=False, retry=False):
    previous = status(connection, match_id)
    if reprocess:
        return True
    if previous.get("status") == "complete":
        if previous["sampling_interval_s"] != interval:
            raise ValueError(f"Match {match_id} already uses {previous['sampling_interval_s']:g}s; "
                             "use reprocess to change its sampling interval")
        return retry and previous.get("latest_status") in ("failed", "interrupted", "unverified_eligibility")
    return retry or previous.get("latest_status") not in ("failed", "ineligible", "unverified_eligibility")


def start_attempt(connection, match_id, replay, interval, *, metadata_source=None, parser_revision="unknown"):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("sampling interval must be positive and finite")
    replay = Path(replay).resolve()
    stat = replay.stat() if replay.is_file() else None
    size, mtime = (stat.st_size, stat.st_mtime_ns) if stat else (None, None)
    compression = "bz2" if replay.name.endswith(".bz2") else "zstd" if replay.name.endswith(".zst") else "none"
    source = connection.execute("SELECT source_id,match_id FROM replay_sources "
                                "WHERE path=? AND size_bytes IS NOT DISTINCT FROM ? AND mtime_ns IS NOT DISTINCT FROM ?",
                                [str(replay), size, mtime]).fetchone()
    if source and source[1] != match_id:
        raise ValueError("the same source replay cannot belong to two matches")
    source_id, attempt_id, started = source[0] if source else uuid.uuid4(), uuid.uuid4(), now()
    connection.execute("BEGIN TRANSACTION")
    try:
        if not source:
            connection.execute("INSERT INTO replay_sources VALUES (?,?,?,?,?,?,?)",
                               [source_id, match_id, str(replay), size, mtime, compression, started])
        connection.execute("""INSERT INTO processing_attempts
            (attempt_id,match_id,source_id,status,stage,started_at_utc,sampling_interval_s,
             parser_name,parser_version,parser_revision,schema_version,metadata_source)
            VALUES (?,?,?,'running','metadata',?,?,'deadem','4.0.1',?,?,?)""",
            [attempt_id, match_id, source_id, started, interval, parser_revision, SCHEMA_VERSION, metadata_source])
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return attempt_id


def record_metadata(connection, attempt_id, payload, fetched_at=None):
    """Record fetch provenance immediately, even if eligibility/parsing later fails."""
    connection.execute("UPDATE processing_attempts SET metadata_fetched_at_utc=?,metadata_sha256=? "
                       "WHERE attempt_id=?", [fetched_at or now(), digest(payload), attempt_id])


def finish_attempt(connection, attempt_id, status, stage, reason=""):
    if status not in ("failed", "ineligible", "unverified_eligibility", "interrupted"):
        raise ValueError("complete attempts must be committed through replace_match")
    connection.execute("UPDATE processing_attempts SET status=?,stage=?,reason=?,finished_at_utc=? "
                       "WHERE attempt_id=? AND status='running'", [status, stage, reason, now(), attempt_id])


def _insert(connection, name, rows):
    """Arrow registration avoids per-row SQL; explicit target types preserve uint64 IDs."""
    if not rows:
        return
    fields = connection.execute(f"SELECT * FROM {name} LIMIT 0").to_arrow_table().schema
    table = normalize._table(rows, fields)
    connection.register("_full_match_batch", table)
    try:
        connection.execute(f"INSERT INTO {name} SELECT * FROM _full_match_batch")
    finally:
        connection.unregister("_full_match_batch")


def _known_slot(value, roster):
    return value if value in roster else None


def _events(payload, replay_events, definitions):
    """Use existing metadata/replay reconciliation, then remove sparse union columns."""
    normalized = normalize.events(payload, replay_events, definitions).to_pylist()
    rows = defaultdict(list)
    grouped = defaultdict(list)
    roster = normalize._roster(payload["match_info"])
    mid = normalize._int(payload["match_info"]["match_id"])
    duration = payload["match_info"]["duration_s"]
    replay_kills = [r for r in replay_events
                    if r.get("event_type", "").removeprefix("k_EUserMsg_") == "HeroKilled"]
    for row in normalized:
        if row["event_type"] == "hero_kill":
            grouped[row["event_id"]].append(row)
            continue
        common = {k: row[k] for k in ("match_id", "event_id", "event_time_s", "event_type", "event_source")}
        if row["event_type"].startswith("objective_"):
            rows["objective_events"].append({**common, "objective_id": row["objective_id"],
                                             "objective_team_id": row["objective_team_id"]})
        elif row["event_type"] in ("midboss_killed", "rejuvenator_claim"):
            rows["midboss_events"].append({**common, "team_id": row["team_id"],
                                           "related_event_id": row["related_event_id"]})
        else:
            rows["item_events"].append({**common, **{k: row[k] for k in (
                "player_slot", "item_id", "ability_id", "upgrade_id", "upgrade_info", "imbued_ability_id")}})
    for event_id, participants in grouped.items():
        victim = next(row for row in participants if row["participant_role"] == "victim")
        killer = next((row for row in participants if row["participant_role"] == "killer"), {})
        rec = {"match_id": mid, "event_id": event_id,
               "event_time_s": victim["event_time_s"], "event_source": victim["event_source"],
               "victim_player_slot": victim["target_player_slot"],
               "killer_player_slot": victim["actor_player_slot"],
               "death_duration_s": victim["duration_s"], "time_to_kill_s": victim["time_to_kill_s"],
               **{f"victim_{k}": victim[k] for k in "xyz"},
               **{f"killer_{k}": killer.get(k) for k in "xyz"}}
        # Precise matching only; do not guess which raw notification supplied provenance.
        evidence = [r for r in replay_kills if r.get("game_time_s") == rec["event_time_s"]
                    and r.get("target_player_slot") == rec["victim_player_slot"]
                    and r.get("actor_player_slot") == rec["killer_player_slot"]]
        if "replay" in rec["event_source"] and len(evidence) == 1:
            raw = evidence[0]
            rec.update(demo_tick=raw.get("tick"), network_tick=raw.get("network_tick"),
                       scorer_player_slot=_known_slot(raw.get("scorer_player_slot"), roster),
                       attacker_player_slot=_known_slot(raw.get("attacker_player_slot"), roster))
        rows["hero_kills"].append(rec)
        rows["kill_assists"].extend({"match_id": mid, "event_id": event_id, "player_slot": r["player_slot"]}
                                    for r in participants if r["participant_role"] == "assister")
    for index, raw in enumerate(replay_events):
        time = normalize._number(raw.get("game_time_s"), "event game time")
        if time is None or not 0 <= time <= duration + 1:
            continue
        kind = raw.get("event_type", "").removeprefix("k_EUserMsg_")
        if kind not in ("BossKilled", "RejuvStatus"):
            continue
        data = raw.get("data") or {}
        common = {"match_id": mid, "event_id": f"{mid}:notification:{index}", "event_time_s": time,
                  "demo_tick": raw.get("tick"), "network_tick": raw.get("network_tick"),
                  "actor_player_slot": _known_slot(raw.get("actor_player_slot"), roster)}
        if kind == "BossKilled":
            pos = data.get("entityPosition") or {}
            rows["boss_notifications"].append({**common,
                "objective_team_code": data.get("objectiveTeam"),
                "objective_mask_change": data.get("objectiveMaskChange"),
                "entity_killed_class": data.get("entityKilledClass"), "bosses_remaining": data.get("bossesRemaining"),
                **{k: pos.get(k) for k in "xyz"}})
        else:
            rows["rejuv_events"].append({**common, "killing_team_code": data.get("killingTeam"),
                "user_team_code": data.get("userTeam"), "event_type_code": data.get("eventType")})
    return rows


def _globals(samples, match_id):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["game_time_s"]].append(sample)
    output = []
    for time, group in grouped.items():
        values = [r.get("global") for r in group]
        if any(value != values[0] for value in values):
            raise ValueError("inconsistent global state within a snapshot")
        value = values[0] or {}
        observed = normalize._number(value.get("observed_time_s"), "global observation time")
        if observed is not None and not 0 <= observed <= time:
            raise ValueError("invalid global observation time or future backfill")
        if observed is None and any(value.get(key) is not None for key in ("game_state", "paused", "server_tick")):
            raise ValueError("unobserved global snapshot contains state values")
        output.append({"match_id": match_id, "game_time_s": time, "observed_time_s": observed,
                       "is_final_observation": any(r.get("is_final_observation") for r in group),
                       **{key: value.get(key) for key in ("game_state", "paused", "server_tick")}})
    return output


def replace_match(connection, attempt_id, payload, summary, samples, replay_events, definitions=None):
    """Validate one replay and publish all its tables atomically.

    An exception rolls back every research table and leaves the earlier successful
    attempt intact. The caller records the new failure outside this transaction.
    """
    attempt = connection.execute("SELECT match_id,status,sampling_interval_s FROM processing_attempts "
                                 "WHERE attempt_id=?", [attempt_id]).fetchone()
    if not attempt or attempt[1] != "running":
        raise ValueError("replacement requires an active processing attempt")
    mid, _, interval = attempt
    if normalize._int(payload["match_info"]["match_id"]) != mid:
        raise ValueError("metadata belongs to another match")
    eligibility, reason = normalize.eligibility(payload)
    if eligibility != "eligible":
        raise ValueError(f"match is {eligibility}: {reason}")
    if summary.get("interval_s", interval) != interval:
        raise ValueError("parser sampling interval differs from the processing attempt")
    samples, replay_events = list(samples), list(replay_events)
    snapshots = normalize.time_slices(payload, samples)
    if not snapshots.num_rows or snapshots["observed_time_s"].null_count == snapshots.num_rows:
        raise ValueError("no verified replay gameplay observations")
    duration = payload["match_info"]["duration_s"]
    if any(t > duration + 1 for t in snapshots["game_time_s"].to_pylist()):
        raise ValueError("snapshot is later than match duration precision permits")
    final_times = {raw["game_time_s"] for raw in samples if raw.get("is_final_observation")}
    for raw in samples:
        time = raw["game_time_s"]
        if time not in final_times and not math.isclose(time / interval, round(time / interval), abs_tol=1e-8):
            raise ValueError("non-final snapshot is not on the configured sampling grid")
    definitions = definitions or []
    if isinstance(definitions, dict):
        definitions = definitions.get("items", [])
    tables = _events(payload, replay_events, definitions)
    tables["matches"] = [{**normalize.match(payload, summary).to_pylist()[0], "successful_attempt_id": str(attempt_id)}]
    tables["match_players"] = normalize.end_match(payload).to_pylist()
    for player in tables["match_players"]:
        account = player["account_id"]
        player["account_id"] = account or None
        for source, dest in (("hero_build_ids", "hero_build_id"), ("pregame_hero_ids", "pregame_hero_id")):
            values = payload.get(source) or {}
            player[dest] = normalize._int(values.get(str(account), values.get(account)), dest)
    tables["match_bans"] = [{"match_id": mid, "hero_id": normalize._int(hero, "banned hero")}
                             for hero in sorted(set(payload.get("banned_hero_ids") or []))]
    tables["player_snapshots"] = snapshots.drop(["account_id", "hero_id", "team_id", "hero_damage_taken"]).to_pylist()
    tables["global_snapshots"] = _globals(samples, mid)
    # No NULL/default confusion: metadata rank 0 is unknown; absent statistics stay NULL.
    for row in tables["player_snapshots"]:
        row["is_final_observation"] = bool(row["is_final_observation"])
        if row["observed_time_s"] is None and any(value is not None for key, value in row.items()
                                                 if key not in ("match_id", "player_slot", "game_time_s",
                                                                "observed_time_s", "is_final_observation")):
            raise ValueError("unobserved player snapshot contains state values")
    connection.execute("BEGIN TRANSACTION")
    try:
        for table in MATCH_TABLES:
            connection.execute(f"DELETE FROM {table} WHERE match_id=?", [mid])
        for table in reversed(MATCH_TABLES):
            _insert(connection, table, tables[table])
        version = summary.get("client_version")
        if version is not None and definitions:
            used_ids = {r.get(k) for r in tables["item_events"] for k in ("item_id", "ability_id")}
            for item in definitions:
                if item.get("id") in used_ids:
                    connection.execute("INSERT INTO item_definitions VALUES (?,?,?,?) ON CONFLICT DO NOTHING",
                                       [version, item["id"], item.get("name"), item.get("type")])
        connection.execute("""UPDATE processing_attempts SET status='complete',stage='complete',reason='',
            finished_at_utc=?, metadata_sha256=?, definitions_sha256=?, definitions_available=?,
            parser_name=?,parser_version=?,clock_method=?,tick_interval_s=?,replay_match_id=?,
            replay_game_mode=?,replay_match_mode=?,final_observed_game_time_s=?,terminal_game_time_s=?,
            terminal_duration_difference_s=?,replay_reaches_match_end=?,warnings=? WHERE attempt_id=?""",
            [now(), digest(payload), digest(definitions) if definitions else None, bool(definitions),
             summary.get("parser", "deadem"), summary.get("parser_version", "4.0.1"), summary.get("clock_method"),
             summary.get("tick_interval_s"), summary.get("replay_match_id"), summary.get("replay_game_mode"),
             summary.get("replay_match_mode"), summary.get("final_observed_game_time_s"),
             summary.get("terminal_game_time_s"), summary.get("terminal_duration_difference_s"),
             summary.get("replay_reaches_match_end"), summary.get("warnings", []), attempt_id])
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return {table: len(rows) for table, rows in tables.items()}
