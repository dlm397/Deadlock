import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import API, now, save_json
from .lake_source import LakeSource
from .lake_storage import complete_lake_match, save_lake_match

LOG = logging.getLogger(__name__)
MANIFEST_URL = "https://data.deadlock-api.com/v1/manifest.json"
TABLES = ("match_player", "match_salts")


def match_id(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Match IDs must be positive integers")
    if isinstance(value, str) and not value.isdecimal():
        raise ValueError("Match IDs must be positive integers")
    value = int(value)
    if not 0 < value < 2**64:
        raise ValueError("Match IDs must fit unsigned 64-bit integers")
    return value


def utc_date(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Dates must be ISO date/time strings, or None")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def discovery_windows(manifest, low, high):
    table = (manifest or {}).get("tables", {}).get("match_player", {})
    partition = re.fullmatch(r"intDiv\(\s*match_id\s*,\s*([1-9][0-9]*)\s*\)", table.get("partition_expr", ""))
    parts = {int(file["partition"]) for file in table.get("files", []) if file.get("partition") is not None}
    for file in table.get("files", []):
        parts.update(int(key) for key in file.get("rows_by_partition", {}))
    if not partition or not parts:
        yield low, high
        return
    stride = int(partition[1])
    floor = low if low is not None else 1
    newest = max(parts) * stride
    # The open upper end also includes rows newer than the observed manifest.
    if high is None or high >= max(floor, newest):
        yield max(floor, newest), high
    end = min(high, newest - 1) if high is not None else newest - 1
    while end >= floor:
        begin = max(floor, end // stride * stride)
        yield begin, end
        end = begin - 1


def candidate_matches(source, filters, page_size, manifest=None):
    clauses = [f"duration_s >= {filters['min_duration_s']}"]
    for field in ("game_mode", "match_mode"):
        if filters[field] is not None:
            clauses.append(f"CAST({field} AS VARCHAR) = {literal(filters[field])}")
    for key, operator in (("start_date", ">="), ("end_date", "<")):
        if filters[key] is not None:
            clauses.append(f"start_time {operator} TIMESTAMPTZ {literal(filters[key])}")
    roster = (f" HAVING COUNT(DISTINCT player_slot) = {filters['expected_players']}"
              if filters["expected_players"] is not None else "")
    for low, high in discovery_windows(manifest, filters["min_match_id"], filters["max_match_id"]):
        bounds = ([f"match_id >= {low}"] if low is not None else [])
        bounds += [f"match_id <= {high}"] if high is not None else []
        cursor = None
        while True:
            before = [f"match_id < {cursor}"] if cursor is not None else []
            sql = ("SELECT match_id FROM match_player WHERE " + " AND ".join(clauses + bounds + before)
                   + f" GROUP BY match_id{roster} ORDER BY match_id DESC LIMIT {page_size}")
            LOG.info("Discovering lake matches%s", f" below {cursor}" if cursor else f" in ID range {low} to {high or 'latest'}")
            rows = source.query(sql)
            ids = sorted({match_id(row["match_id"]) for row in rows}, reverse=True)
            if not ids:
                break
            if cursor is not None and ids[0] >= cursor:
                raise ValueError("Lake discovery returned a non-advancing page")
            if low is not None and ids[-1] < low or high is not None and ids[0] > high:
                raise ValueError("Lake discovery returned IDs outside the requested range")
            yield from ids
            cursor = ids[-1]
            if len(rows) < page_size:
                break


def eligible(rows, filters):
    slots = {row.get("player_slot") for row in rows}
    if None in slots or (filters["expected_players"] is not None and len(slots) != filters["expected_players"]):
        return False
    for row in rows:
        if row.get("duration_s", 0) < filters["min_duration_s"]:
            return False
        for field in ("game_mode", "match_mode"):
            if filters[field] is not None and row.get(field) != filters[field]:
                return False
        if filters["start_date"] or filters["end_date"]:
            started = utc_date(row["start_time"])
            if filters["start_date"] and started < utc_date(filters["start_date"]):
                return False
            if filters["end_date"] and started >= utc_date(filters["end_date"]):
                return False
    return True


def run(n_matches=10, output_dir="./data/lake", match_ids=None, game_mode="Normal", match_mode=None,
        min_duration_s=900, expected_players=12, start_date=None, end_date=None,
        min_match_id=None, max_match_id=None, reprocess=False, page_size=100,
        max_candidates=None, request_delay=0.0, compression_level=19):
    for name, value, minimum, maximum in (
        ("N_MATCHES", n_matches, 1, None), ("MIN_DURATION_S", min_duration_s, 0, None),
        ("PAGE_SIZE", page_size, 1, 256), ("COMPRESSION_LEVEL", compression_level, 1, 22),
        ("EXPECTED_PLAYERS", expected_players, 1, None), ("MAX_CANDIDATES", max_candidates, 1, None),
    ):
        if value is None and name in ("EXPECTED_PLAYERS", "MAX_CANDIDATES"):
            continue
        if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
                or maximum is not None and value > maximum):
            raise ValueError(f"Invalid {name}: {value}")
    if isinstance(request_delay, bool) or not isinstance(request_delay, (int, float)) or not math.isfinite(request_delay) or request_delay < 0:
        raise ValueError("REQUEST_DELAY must be a finite nonnegative number")
    for mode in (game_mode, match_mode):
        if mode is not None and (not isinstance(mode, str) or not mode.strip()):
            raise ValueError("Game and match modes must be names, or None")
    start, end = utc_date(start_date), utc_date(end_date)
    if start and end and start >= end:
        raise ValueError("START_DATE must precede exclusive END_DATE")
    low = match_id(min_match_id) if min_match_id is not None else None
    high = match_id(max_match_id) if max_match_id is not None else None
    if low is not None and high is not None and low > high:
        raise ValueError("MIN_MATCH_ID must not exceed MAX_MATCH_ID")
    explicit = list(dict.fromkeys(match_id(mid) for mid in match_ids)) if match_ids is not None else None
    filters = dict(game_mode=game_mode, match_mode=match_mode, min_duration_s=min_duration_s,
                   expected_players=expected_players, start_date=start.isoformat() if start else None,
                   end_date=end.isoformat() if end else None, min_match_id=low, max_match_id=high)
    settings = dict(n_matches=n_matches, match_ids=explicit, **filters, reprocess=reprocess,
                    page_size=page_size, max_candidates=max_candidates, request_delay=request_delay,
                    compression_level=compression_level)
    root = Path(output_dir)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / "run.log", encoding="utf-8")], force=True)
    api = API(root)
    totals = {"saved": 0, "skipped": 0, "failed": 0}
    started = time.perf_counter()
    run_error = None
    attempted = 0
    try:
        source = LakeSource(api, request_delay=request_delay)
        manifest, manifest_prov = api.json(MANIFEST_URL)
        manifest_path = f"tables/manifest_{manifest['version']}.json.gz"
        save_json(root / manifest_path, manifest)
        schemas = {table: source.describe(table) for table in TABLES}
        save_json(root / "tables/schema.json.gz", schemas)
        save_json(root / "tables/setup_requests.json.gz", api.requests)
        api.requests.clear()
        provenance = {"source": "https://api.deadlock-api.com/v1/mcp", "schemas": schemas,
                      "manifest_observed": {"version": manifest["version"], "path": manifest_path,
                                            "generated_at": manifest.get("generated_at"), **manifest_prov},
                      "snapshot_note": "MCP follows the current lake; each table payload is checked for changes during retrieval."}
        selected = explicit if explicit is not None else candidate_matches(source, filters, page_size, manifest)
        for mid in selected:
            folder = root / str(mid)
            if not reprocess and complete_lake_match(folder):
                totals["skipped"] += 1
                LOG.info("Lake match %s already complete; skipping", mid)
                continue
            attempted += 1
            try:
                LOG.info("Lake match %s: fetching every source column", mid)
                players = source.fetch_match("match_player", mid)
                if not players or explicit is None and not eligible(players, filters):
                    totals["skipped"] += 1
                    LOG.info("Lake match %s skipped: missing players or selection filters", mid)
                    save_json(folder / "last_error.json.gz", {"match_id": mid, "skipped": True,
                              "reason": "No player rows or roster/selection filters not satisfied", "completed_at": now()})
                else:
                    salts = source.fetch_match("match_salts", mid)
                    tables = {"match_player": players, "match_salts": salts}
                    save_lake_match(root, mid, tables, {**provenance, "selection": settings,
                                    "retrieved_at": now()}, compression_level=compression_level)
                    totals["saved"] += 1
                    LOG.info("Lake match %s saved (%s player records, %s source columns)", mid,
                             len(players), len(players[0]))
            except Exception as exc:
                totals["failed"] += 1
                LOG.warning("Lake match %s failed; continuing: %s", mid, exc)
                save_json(folder / "last_error.json.gz", {"match_id": mid, "processed_success": False,
                          "error": str(exc), "completed_at": now()})
            finally:
                save_json(folder / "requests.json.gz", api.requests)
                api.requests.clear()
            if explicit is None and (totals["saved"] >= n_matches or attempted >= (max_candidates or max(100, n_matches * 10))):
                break
    except Exception as exc:
        run_error = str(exc)
        raise
    finally:
        if api.requests:
            save_json(root / "tables/requests.json.gz", api.requests)
        api.session.close()
        save_json(root / "last_run.json", {**totals, "attempted": attempted, "completed_at": now(),
                  "elapsed_seconds": time.perf_counter() - started, "settings": settings, "error": run_error})
    LOG.info("Lake collection finished: %s", totals)
    if explicit is None and totals["saved"] < n_matches:
        LOG.warning("Saved %s of %s requested lake matches; candidate limit or available matches exhausted", totals["saved"], n_matches)
    return totals
