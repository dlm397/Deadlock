"""Convert the old Deadlock archive into this project's current research tables.

Run from Deadlock2 with its Python environment and installed Node dependencies:
    python other/import_legacy.py --dry-run
    python other/import_legacy.py 
    python other/import_legacy.py --match-ids 109521468 --output data_import_check

This is one new entry-point script using Deadlock2's existing normalization and
replay parser. It is offline: no downloads, and no changes to the source archive.
Retained demos are read in place, not copied. Old sampled tables and analysis/CSV
exports are not used because they duplicate sources or have different semantics.

Defaults come from config.toml (interval, output directory, CSV setting). Existing
research files are skipped unless --overwrite is given. To retry a partial import
or change its interval, use --overwrite --match-ids ID [...]. Metadata-only matches
produce matches/end_match/events; time slices require a retained replay. Unknown
versions and unavailable values remain missing. Use this importer again to resample
the old archive, since it does not copy demos into Deadlock2/data/replays.
"""

import argparse
import csv
import gzip
import json
import math
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from app import normalize, sources
from app.console_log import log
from app.pipeline import (FAILURE_COLUMNS, STATUS_COLUMNS, TABLES, now, read_log,
                          short_error, write_log, write_table)
from app.version_tables import write_version_tables


DEFAULT_SOURCE = HERE.parent / "Deadlock" / "data"


def read_json(path):
    """Read one known JSON archive, preferring the plain file if both exist."""
    path = Path(path)
    if not path.is_file():
        path = Path(str(path) + ".gz")
    if not path.is_file():
        return None
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def match_ids(root):
    """Inspect only immediate match directories, not the entire 100+ GiB tree."""
    ids = set()
    for name in ("raw", "lake"):
        folder = root / name
        if folder.is_dir():
            ids.update(int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isascii() and p.name.isdecimal())
    return sorted(ids, reverse=True)


# The lake adapter below reconstructs the API structure expected by normalize.
# Requires imports: json, gzip, datetime (class), timezone, Path, pyarrow.parquet as pq.
# Interface: load_lake_metadata(data_root, match_id) -> (API-shaped metadata, notes).
# data_root is the original project's data/ directory. This helper only reads it.

def load_lake_metadata(data_root, match_id):
    folder = Path(data_root) / "lake" / str(match_id)
    status_path = folder / "status.json.gz"
    if status_path.is_file():
        with gzip.open(status_path, "rt", encoding="utf-8") as file:
            status = json.load(file)
    else:
        with (folder / "status.json").open(encoding="utf-8") as file:
            status = json.load(file)
    if status.get("success") is not True or status.get("match_id") != match_id:
        raise ValueError("legacy lake snapshot is not marked successful for this match")
    rows = pq.read_table(folder / "match_player.parquet").to_pylist()
    if not rows or status.get("row_counts", {}).get("match_player") != len(rows):
        raise ValueError("legacy lake row count disagrees with its completed snapshot")
    if any(row.get("match_id") != match_id for row in rows):
        raise ValueError("legacy lake contains a different match ID")

    def stamp(value):
        if value is None:
            return None
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("legacy lake timestamp has no UTC offset")
        return value

    cohorts = {}
    for row in rows:
        cohorts.setdefault(row.get("created_at"), {})[
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ] = row
    if len(cohorts) > 1 and None in cohorts:
        raise ValueError("cannot order legacy lake cohorts with missing created_at")
    selected_stamp = max(cohorts, key=stamp) if len(cohorts) > 1 else next(iter(cohorts))
    selected = list(cohorts[selected_stamp].values())
    slots = [row.get("player_slot") for row in selected]
    if any(type(slot) is not int or slot < 0 for slot in slots) or len(slots) != len(set(slots)):
        raise ValueError("latest legacy lake cohort has missing or conflicting player slots")
    # Retrieval was an uncapped, digest-checked match snapshot. If cohorts have
    # different rosters, there is no evidence that the latest cohort is complete.
    if any({row.get("player_slot") for row in cohort.values()} != set(slots)
           for cohort in cohorts.values()):
        raise ValueError("legacy lake cohorts disagree on roster completeness")
    notes = []
    duplicate_count = len(rows) - sum(len(cohort) for cohort in cohorts.values())
    if duplicate_count:
        notes.append(f"Removed {duplicate_count} exact duplicate lake rows")
    if len(cohorts) > 1:
        notes.append(f"Selected latest of {len(cohorts)} lake cohorts: {selected_stamp}")

    match_fields = ("match_id", "start_time", "duration_s", "game_mode", "match_mode",
                    "match_outcome", "winning_team", "game_mode_version")
    first = selected[0]
    shared_fields = set(match_fields) | {key for row in selected for key in row
                                       if key.startswith(("objectives.", "mid_boss."))}
    if any(any(row.get(key) != first.get(key) for key in shared_fields) for row in selected):
        raise ValueError("latest legacy lake cohort has conflicting match metadata")

    def nested(row, prefix, fields=None):
        arrays = {}
        for key, value in row.items():
            if not key.startswith(prefix + "."):
                continue
            name = key[len(prefix) + 1:]
            if fields is not None and name not in fields:
                continue
            value = json.loads(value) if isinstance(value, str) else value
            if value is not None and not isinstance(value, list):
                raise ValueError(f"legacy lake {key} is not an array")
            arrays[name] = value
        lengths = {len(value) for value in arrays.values() if value is not None}
        if len(lengths) > 1:
            raise ValueError(f"legacy lake {prefix} arrays have inconsistent lengths")
        count = next(iter(lengths), 0)
        return [{key: values[index] if values is not None else None
                 for key, values in arrays.items()} for index in range(count)]

    info = {key: first.get(key) for key in match_fields}
    if info["start_time"] is not None:
        info["start_time"] = int(stamp(info["start_time"]).timestamp())
    scalar_fields = ("account_id", "player_slot", "hero_id", "team", "kills", "deaths",
                     "assists", "net_worth", "last_hits", "denies", "assigned_lane")
    stat_fields = {"time_stamp_s", "kills", "deaths", "assists", "net_worth", "last_hits",
                   "denies", "level", "player_damage", "boss_damage", "player_healing",
                   "self_healing", "player_damage_taken"}
    info["players"] = []
    for row in sorted(selected, key=lambda record: record["player_slot"]):
        player = {key: row.get(key) for key in scalar_fields}
        player["level"] = row.get("player_level")
        player["player_rank_data"] = {
            key.removeprefix("player_rank_"): value for key, value in row.items()
            if key.startswith("player_rank_")}
        player["stats"] = nested(row, "stats", stat_fields)
        player["items"] = nested(row, "items")
        player["death_details"] = nested(row, "death_details")
        for death in player["death_details"]:
            for key in ("death_pos", "killer_pos"):
                pos = death.get(key)
                if pos is not None:
                    if not isinstance(pos, dict):
                        raise ValueError(f"legacy lake {key} is not a vector")
                    death[key] = {axis: pos.get(axis, pos.get(str(index)))
                                  for index, axis in enumerate("xyz", 1)}
        info["players"].append(player)

    # Valve ECitadelTeamObjective: SteamDatabase/Protobufs
    # deadlock/citadel_gcmessages_common.proto, enum values 0 through 15.
    objective_names = ("Core", "Tier1Lane1", "Tier1Lane2", "Tier1Lane3", "Tier1Lane4",
                       "Tier2Lane1", "Tier2Lane2", "Tier2Lane3", "Tier2Lane4", "Titan",
                       "TitanShieldGenerator1", "TitanShieldGenerator2",
                       "BarrackBossLane1", "BarrackBossLane2", "BarrackBossLane3", "BarrackBossLane4")
    objective_ids = {name.lower(): index for index, name in enumerate(objective_names)}
    info["objectives"] = nested(first, "objectives")
    for objective in info["objectives"]:
        value = objective.pop("team_objective", None)
        if isinstance(value, str):
            value = objective_ids.get(value.removeprefix("k_eCitadelTeamObjective_").replace("_", "").lower())
        elif type(value) is not int or not 0 <= value <= 15:
            value = None
        objective["team_objective_id"] = value
    info["mid_boss"] = nested(first, "mid_boss")
    # No client/build version is present in this lake schema. A replay header can
    # supply it later; game_mode_version must not be substituted for client_version.
    return {"match_info": info}, notes



def archived_metadata(root, mid):
    payload = read_json(root / "raw" / str(mid) / "sources" / "api" / "metadata.json")
    if payload is not None:
        info = payload.get("match_info") if isinstance(payload, dict) else None
        if not isinstance(info, dict) or info.get("match_id") != mid:
            raise ValueError("archived metadata is malformed or belongs to another match")
        return payload, "raw metadata", []
    if (root / "lake" / str(mid) / "match_player.parquet").is_file():
        payload, notes = load_lake_metadata(root, mid)
        return payload, "lake player records", notes
    raise FileNotFoundError("no archived full metadata or complete lake player table; salts/URLs alone are insufficient")


def archived_replay(root, mid):
    folder = root / "raw" / str(mid) / "sources"
    found = [folder / f"replay{suffix}" for suffix in sources.REPLAY_SUFFIXES
             if (folder / f"replay{suffix}").is_file()]
    if len(found) > 1:
        raise ValueError("multiple retained replays; keep one intended replay in the archive")
    return found[0] if found else None


def archived_header(root, mid):
    summary = read_json(root / "raw" / str(mid) / "sources" / "replay" / "summary.json") or {}
    if not summary.get("success"):
        return {}
    header = summary.get("header") or {}
    return {"client_version": summary.get("game_version"), "map_name": header.get("mapName"),
            "replay_build_num": header.get("buildNum"), "replay_patch_version": header.get("patchVersion")}


def archived_items(root, output, version, cache):
    if version is None:
        return []
    if version not in cache:
        rows = read_json(root / "tables" / "sources" / str(version) / "items.json")
        if rows is None:
            # A previously exported exact-version catalog is enough to classify events.
            path = output / "version" / f"{version}_item.csv"
            if path.is_file():
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = [{"id": int(row["item_id"]), **{key: value or None for key, value in row.items()
                             if key not in ("item_id", "client_version")}} for row in csv.DictReader(stream)]
        if rows is not None and not isinstance(rows, list):
            raise ValueError(f"version {version} item definitions are not a list")
        cache[version] = rows or []
    return cache[version]


def run(args, config):
    root, output = args.source, config["output_dir"]
    selected = list(dict.fromkeys(args.match_ids)) if args.match_ids else match_ids(root)
    if args.limit is not None:
        selected = selected[:args.limit]
    counts, item_cache, written_versions = Counter(), {}, set()
    failure_examples = 0
    statuses = read_log(output / "collection_status.csv") if not args.dry_run else {}
    failures = read_log(output / "failures.csv") if not args.dry_run else {}
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    log(f"{'Inspecting' if args.dry_run else 'Importing'} {len(selected)} archived match IDs; "
        f"interval={config['sampling_interval_s']:g}s; source files remain read-only.")

    def save_logs():
        write_log(output / "collection_status.csv", STATUS_COLUMNS, statuses)
        write_log(output / "failures.csv", FAILURE_COLUMNS, failures)

    def definitions(version):
        items = archived_items(root, output, version, item_cache)
        if version is not None and version not in written_versions:
            write_version_tables(output, version, items)
            written_versions.add(version)
        return items

    for index, mid in enumerate(selected, 1):
        paths = {name: output / name / f"{mid}.parquet" for name in TABLES}
        existing = any(path.exists() or path.with_suffix(".csv").exists() for path in paths.values())
        if existing and not args.overwrite:
            counts["existing_skipped"] += 1
            if not args.dry_run:
                log(f"SKIP: Match {mid}: existing research output (use --overwrite to replace).")
            continue
        previous = statuses.get(mid, {})
        stage, base_written = "legacy_metadata", False
        record = dict.fromkeys(STATUS_COLUMNS)
        record.update(match_id=mid, status="failed", reason="", updated_at_utc=now())
        try:
            payload, source_name, notes = archived_metadata(root, mid)
            eligibility, reason = normalize.eligibility(payload)
            if eligibility != "eligible":
                state = "ineligible" if eligibility == "ineligible" else "unverified_eligibility"
                counts[state] += 1
                if not args.dry_run:
                    record.update(status=state, reason=reason)
                    if existing:
                        # Failed replacement must not destroy earlier research output.
                        statuses[mid] = {**record, **previous, "match_id": mid,
                                         "reason": f"Legacy import rejected; existing files retained: {reason}"}
                    else:
                        statuses[mid] = record
                    if state == "unverified_eligibility" or existing:
                        failures[mid] = dict(match_id=mid, stage="eligibility", reason=reason, attempted_at_utc=now())
                    else:
                        failures.pop(mid, None)
                    save_logs()
                    log(f"SKIP: Match {mid}: {reason}.")
                continue
            replay = archived_replay(root, mid)
            if args.dry_run:
                counts["eligible_with_replay" if replay else "eligible_metadata_only"] += 1
                counts["raw_metadata" if source_name == "raw metadata" else "lake_metadata"] += 1
                if index % 250 == 0:
                    log(f"Inspected {index}/{len(selected)} match IDs.")
                continue
            log(f"Match {mid}: converting {source_name}.")
            header = archived_header(root, mid)
            stage = "legacy_definitions"
            items = definitions(header.get("client_version"))
            tables = {"matches": normalize.match(payload, header), "end_match": normalize.end_match(payload),
                      "events": normalize.events(payload, [], items)}
            # Preserve any prior output through a failed --overwrite attempt.
            if not existing:
                for name, table in tables.items():
                    write_table(output, name, mid, table, config["export_csv"])
                base_written = True
                record.update(status="end_match_only", reason="Legacy import in progress; metadata tables written.")
                statuses[mid] = dict(record)
                save_logs()
            if replay is None or args.metadata_only:
                if paths["time_slices"].exists() or paths["time_slices"].with_suffix(".csv").exists():
                    raise ValueError("cannot replace time-slice output with metadata-only data; use a separate --output directory")
                if existing:
                    for name, table in tables.items():
                        write_table(output, name, mid, table, config["export_csv"])
                    base_written = True
                reason = "no retained replay" if replay is None else "replay parsing disabled by --metadata-only"
                notes.append(f"Legacy import: {reason}; metadata events only; time slices unavailable")
                record.update(status="end_match_only", reason="; ".join(notes), updated_at_utc=now())
                statuses[mid] = record
                failures.pop(mid, None)
                counts["end_match_only"] += 1
                save_logs()
                log(f"SAVED: Match {mid}: 1 match row, 12 player rows, metadata events; {reason}.")
                continue
            stage = "legacy_replay_parse"
            log(f"Match {mid}: parsing the retained replay at {config['sampling_interval_s']:g}s...")
            # parse_replay sniffs compression: many .bz2 files are actually Zstandard.
            summary, samples, replay_events = sources.parse_replay(replay, payload, config["sampling_interval_s"])
            notes.extend(summary.get("warnings", []))
            stage = "legacy_definitions"
            items = definitions(summary.get("client_version"))
            if not items:
                notes.append("exact-version item definitions unavailable locally; classification is limited")
            stage = "normalize"
            tables = {"matches": normalize.match(payload, summary), "end_match": normalize.end_match(payload),
                      "time_slices": normalize.time_slices(payload, samples),
                      "events": normalize.events(payload, replay_events, items)}
            states = tables["time_slices"]
            if not states.num_rows or states["observed_time_s"].null_count == states.num_rows:
                raise ValueError("no verified replay gameplay observations")
            stage = "write_outputs"
            for name, table in tables.items():
                write_table(output, name, mid, table, config["export_csv"])
            record.update(status="complete", sampling_interval_s=config["sampling_interval_s"],
                          final_observed_game_time_s=summary.get("final_observed_game_time_s"),
                          replay_reaches_match_end=summary.get("replay_reaches_match_end"),
                          reason="; ".join(dict.fromkeys(notes)), updated_at_utc=now())
            statuses[mid] = record
            failures.pop(mid, None)
            counts["complete"] += 1
            save_logs()
            log(f"SAVED: Match {mid}: {states.num_rows} time-slice rows, "
                f"{tables['events'].num_rows} event participant rows.")
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            reason = short_error(error)
            counts["unavailable_sources" if isinstance(error, FileNotFoundError) else "failed"] += 1
            if args.dry_run and (failure_examples < 5 or len(selected) <= 10):
                log(f"UNAVAILABLE: Match {mid}: {reason}.")
                failure_examples += 1
            if not args.dry_run:
                record.update(status="end_match_only" if base_written else "failed", reason=reason, updated_at_utc=now())
                if isinstance(error, sources.EligibilityConflict):
                    record["status"] = "unverified_eligibility"
                    if not existing:
                        for path in paths.values():
                            path.unlink(missing_ok=True)
                            path.with_suffix(".csv").unlink(missing_ok=True)
                statuses[mid] = ({**record, **previous, "reason": f"Legacy import failed; earlier outputs retained: {reason}",
                                  "updated_at_utc": now()}
                                 if existing and stage != "write_outputs" else record)
                failures[mid] = dict(match_id=mid, stage=stage, reason=reason, attempted_at_utc=now())
                save_logs()
                log(f"ERROR: Match {mid}: {reason}.")
    log("Finished: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="old Deadlock data directory")
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--output", type=Path, help="destination; defaults to config output_dir")
    parser.add_argument("--interval", type=float, help="sampling seconds; defaults to config sampling_interval_s")
    parser.add_argument("--match-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int, help="maximum candidate IDs to inspect/process")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs after successful conversion")
    parser.add_argument("--metadata-only", action="store_true", help="convert metadata/events without parsing replays")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and eligibility without writing or parsing replays")
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction, default=None, help="override config CSV export")
    args = parser.parse_args()
    try:
        args.source = args.source.resolve()
        if not any((args.source / name).is_dir() for name in ("raw", "lake")):
            raise ValueError("source must contain the legacy raw and/or lake directories")
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)
        output = (args.output if args.output is not None else args.config.resolve().parent / config["output_dir"]).resolve()
        if output.is_relative_to(args.source) or args.source.is_relative_to(output):
            raise ValueError("source and output directories must be separate, non-nested trees")
        config["output_dir"] = output
        config["sampling_interval_s"] = args.interval if args.interval is not None else config["sampling_interval_s"]
        interval = config["sampling_interval_s"]
        if isinstance(interval, bool) or not isinstance(interval, (float, int)) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("sampling interval must be a positive finite number")
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive")
        if args.match_ids and any(not 0 < mid < 2**64 for mid in args.match_ids):
            raise ValueError("match IDs must be positive uint64 integers")
        if args.csv is not None:
            config["export_csv"] = args.csv
        run(args, config)
    except (OSError, ValueError) as error:
        log(f"ERROR: Import stopped: {short_error(error)}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
