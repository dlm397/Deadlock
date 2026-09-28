import gc
import hashlib
import logging
from pathlib import Path

import pandas as pd
import requests

from .acquire import ReplayUnavailable, acquire_replay, select_matches, static_data
from .quality import audit_metadata, validate_tables
from .io import API, GROUPS, checksum, compact_sources, now, read_json, save_json, save_match, write_table
from .normalize import normalize_metadata
from .replay_data import aggregate_teams, merge_replay, replay_version

LOG = logging.getLogger(__name__)


def signature(interval, parse, query, save_demo, save_raw):
    digest = hashlib.sha256(f"{interval}:{parse}:{query}:{save_demo}:{save_raw}".encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    digest.update((Path(__file__).parent / "replay/export.mjs").read_bytes())
    return digest.hexdigest()


def complete(folder, fingerprint=None):
    try:
        status = read_json(folder / "sources/status.json", {})
        if not status.get("processed_success") or (fingerprint and status.get("signature") != fingerprint):
            return False
        files = status.get("files", {})
        return all((folder / f"{group}.zip").exists()
                   and checksum(folder / f"{group}.zip") == files.get(f"{group}.zip") for group in GROUPS)
    except (OSError, ValueError, EOFError):
        return False


def local_replay(folder, interval):
    sources = folder / "sources"
    summary = read_json(sources / "replay/summary.json", {})
    if summary.get("success"):
        required = ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz")
        if not all((sources / "replay" / name).exists() for name in required) or not read_json(sources / "replay/schema.json"):
            raise FileNotFoundError("Replay extracts are incomplete; use collect.py to reparse")
    if summary.get("success") and summary.get("sample_interval") != interval:
        raise ValueError("Archived replay interval differs; use collect.py to reparse the demo")
    demo = sources / "replay.dem.bz2"
    result = dict(replay_available=demo.exists(), replay_parsed=summary.get("success", False),
                  parser_summary=summary, parser_output=str(sources / "replay"),
                  demo_bytes=demo.stat().st_size if demo.exists() else 0)
    query = sources / "api/demo_query/hero_killed.parquet"
    if query.exists():
        result["demo_query"] = {"status": "done", "path": str(query)}
    return result


def references(root, definitions):
    for name in ("items", "heroes", "abilities"):
        rows = definitions.get("items" if name == "abilities" else name, [])
        if name == "abilities":
            rows = [row for row in rows if row.get("type") == "ability"]
        if not rows:
            continue
        provenance = definitions.get(("items" if name == "abilities" else name) + "_provenance", {})
        records = [{**row, **provenance} for row in rows]
        path = root / "tables" / f"{name}.parquet"
        if path.exists():
            existing = pd.read_parquet(path, dtype_backend="numpy_nullable")
            version = provenance.get("definitions_version")
            keep = existing["definitions_version"].notna() if version is None else existing["definitions_version"].ne(version).fillna(True)
            records = existing.loc[keep].astype(object).where(existing.loc[keep].notna(), None).to_dict("records") + records
        write_table(path, records)


def build_match(api, mid, interval, parse, query, save_demo, offline, fingerprint):
    folder = api.root / "raw" / str(mid)
    payload, prov = api.json(f"/v1/matches/{mid}/metadata", f"raw/{mid}/sources/api/metadata.json",
                            {"disable_steam": "true"}, cached=True) if not offline else (
        read_json(folder / "sources/api/metadata.json"),
        read_json(folder / "sources/api/metadata.json.provenance.json", {}))
    if not payload:
        raise FileNotFoundError(f"No metadata archived for {mid}")
    prov = {"source": prov.get("source", prov.get("url")), "retrieved_at": prov.get("retrieved_at"),
            "metadata_retrieved_at": prov.get("retrieved_at")}
    audit_metadata(payload, folder / "sources/schema")
    replay = local_replay(folder, interval) if offline else acquire_replay(api, mid, interval, save_demo, parse, query)
    if parse and not replay.get("replay_parsed"):
        if replay.get("unavailable_reason") and not replay.get("errors"):
            raise ReplayUnavailable(replay["unavailable_reason"])
        raise RuntimeError("; ".join(replay.get("errors", [])) or "Replay parsing did not succeed")
    version = replay_version(replay.get("parser_summary", {}))
    if offline:
        static = {}
        for name in ("items", "heroes", "ranks", "generic-data"):
            path = api.root / "tables/sources" / str(version) / f"{name}.json"
            value = read_json(path)
            if value is not None:
                static[name] = value
                record = read_json(path.with_suffix(".json.provenance.json"), {})
                static[name + "_provenance"] = {"source": record.get("url"), "retrieved_at": record.get("retrieved_at"),
                                               "definitions_version": version, "version_match": True}
    else:
        versions, _ = api.json("/v1/assets/client-versions", "tables/sources/client_versions.json",
                              required_archive=True, cached=True)
        static = static_data(api, version if version in versions else None)
    exact = version is not None and static.get("items_provenance", {}).get("definitions_version") == version
    prov.update(match_id=mid, game_version=version, definitions_version=version if exact else None)
    tables = normalize_metadata(payload, prov, static.get("items", []) if exact else [], interval)
    merge_replay(tables, replay, prov)
    aggregate_teams(tables, prov)
    tables["game_versions"] = [{**prov, "exact_assets_available": exact,
                                "game_mode_version": payload["match_info"].get("game_mode_version"),
                                "replay_header": replay.get("parser_summary", {}).get("header")}]
    warnings = validate_tables({**tables, "items": static.get("items", []) if exact else []})
    states = tables.get("player_state", [])
    fields = sorted({key for row in states for key in row})
    coverage = {key: {"non_null": sum(row.get(key) is not None for row in states), "rows": len(states)} for key in fields}
    status = dict(match_id=mid, processed_success=True, signature=fingerprint, completed_at=now(),
                  sample_interval=interval, game_version=version, warnings=warnings,
                  replay_parsed=replay.get("replay_parsed"), replay_errors=replay.get("errors", []),
                  variable_coverage=coverage)
    references(api.root, static)
    status = save_match(folder, tables, status)
    status["files"] = {f"{group}.zip": checksum(folder / f"{group}.zip") for group in GROUPS}
    save_json(folder / "sources/status.json", status)
    return status


def update_catalog(root, status):
    path = root / "tables/catalog.json"
    try:
        catalog = read_json(path, {})
    except (OSError, ValueError, EOFError):
        catalog = {}
        LOG.warning("Rebuilding unreadable catalog from per-match status files")
        for folder in (root / "raw").iterdir():
            try:
                previous = read_json(folder / "sources/status.json", {})
                if previous.get("processed_success"):
                    catalog[folder.name] = {key: previous.get(key) for key in
                                          ("match_id", "processed_success", "completed_at", "game_version",
                                           "sample_interval", "row_counts", "error")}
            except (OSError, ValueError, EOFError):
                continue
    catalog[str(status["match_id"])] = {key: status.get(key) for key in
                                      ("match_id", "processed_success", "completed_at", "game_version",
                                       "sample_interval", "row_counts", "error")}
    save_json(path, catalog)


def run(n_matches=5, sample_interval=10, output_dir="./data", match_ids=None,
        save_raw_demos=True, save_raw_json=True, parse_replays=True, query_demo_api=True,
        reprocess=False, offline=False):
    if n_matches < 1 or sample_interval <= 0:
        raise ValueError("N_MATCHES and SAMPLE_INTERVAL must be positive")
    root = Path(output_dir)
    for name in ("raw", "processed", "tables"):
        (root / name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / "tables/run.log", encoding="utf-8")], force=True)
    api = API(root, save_raw_json)
    if not offline:
        try:
            api.json("/v1/assets/client-versions", "tables/sources/client_versions.json", required_archive=True)
        except Exception as exc:
            LOG.warning("Version index refresh failed; using archived index if available: %s", exc)
    fingerprint = signature(sample_interval, parse_replays, query_demo_api, save_raw_demos, save_raw_json)
    done = lambda mid: complete(root / "raw" / str(mid), fingerprint) and not reprocess
    if offline and match_ids is None:
        selected = sorted(int(path.name) for path in (root / "raw").iterdir() if path.is_dir() and path.name.isdigit())
    else:
        selected = select_matches(api, n_matches, match_ids, done, require_replay=parse_replays)
    totals = {"saved": 0, "skipped": 0, "failed": 0}
    try:
        for mid in selected:
            if done(mid):
                totals["skipped"] += 1
                LOG.info("Match %s already complete; skipping", mid)
                continue
            folder = root / "raw" / str(mid)
            try:
                LOG.info("Match %s: download, normalize, save", mid)
                if not offline and match_ids is None:
                    payload, _ = api.json(f"/v1/matches/{mid}/metadata", f"raw/{mid}/sources/api/metadata.json",
                                          {"disable_steam": "true"}, cached=True)
                    info = payload["match_info"]
                    eligible = (info.get("game_mode") == 1 and info.get("duration_s", 0) >= 900
                                and len(info.get("players", [])) == 12 and info.get("winning_team") in (0, 1))
                    if not eligible:
                        LOG.info("Match %s skipped: eligibility filter (mode=%s, duration=%ss, players=%s, winner=%s)",
                                 mid, info.get("game_mode"), info.get("duration_s"),
                                 len(info.get("players", [])), info.get("winning_team"))
                    del payload, info
                    if not eligible:
                        totals["skipped"] += 1
                        continue
                status = build_match(api, mid, sample_interval, parse_replays, query_demo_api,
                                     save_raw_demos, offline, fingerprint)
                update_catalog(root, status)
                totals["saved"] += 1
                LOG.info("Match %s saved (%s state rows)", mid, status["row_counts"]["player_state"])
            except ReplayUnavailable as exc:
                totals["skipped"] += 1
                LOG.info("Match %s skipped: %s", mid, exc)
                save_json(folder / "sources/last_error.json",
                          {"match_id": mid, "processed_success": False, "skipped": True, "reason": str(exc), "completed_at": now()})
            except Exception as exc:
                totals["failed"] += 1
                status = {"match_id": mid, "processed_success": False, "error": str(exc), "completed_at": now()}
                if isinstance(exc, requests.RequestException):
                    LOG.warning("Match %s could not fetch data; continuing: %s", mid, exc)
                else:
                    LOG.exception("Match %s failed; continuing", mid)
                try:
                    save_json(folder / "sources/last_error.json", status)
                    if not complete(folder):
                        update_catalog(root, status)
                except Exception:
                    LOG.exception("Could not write failure status for match %s", mid)
            finally:
                try:
                    if api.requests:
                        save_json(folder / "sources/requests.json", api.requests)
                    compact_sources(folder / "sources")
                except Exception:
                    LOG.exception("Match %s source compaction failed; sources retained", mid)
                api.requests.clear()
                gc.collect()
            if not offline and match_ids is None and totals["saved"] >= n_matches:
                break
    finally:
        api.session.close()
        compact_sources(root / "tables/sources")
        save_json(root / "tables/last_run.json", {**totals, "completed_at": now()})
    LOG.info("Finished: %s", totals)
    if not offline and match_ids is None and totals["saved"] < n_matches:
        LOG.warning("Saved %s of %s requested matches; available replay candidates were exhausted", totals["saved"], n_matches)
    return totals
