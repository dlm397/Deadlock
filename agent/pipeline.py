import gc
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from .acquire import (ReplayUnavailable, acquire_replay, compatible_local_replay,
                      get_salts, select_matches, static_data)
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


def references(root, definitions, written=None):
    for name in ("items", "heroes", "abilities"):
        rows = definitions.get("items" if name == "abilities" else name, [])
        if name == "abilities":
            rows = [row for row in rows if row.get("type") == "ability"]
        if not rows:
            continue
        provenance = definitions.get(("items" if name == "abilities" else name) + "_provenance", {})
        key = (name, provenance.get("definitions_version"), provenance.get("source"), provenance.get("retrieved_at"))
        if written is not None and key in written:
            continue
        records = [{**row, **provenance} for row in rows]
        path = root / "tables" / f"{name}.parquet"
        if path.exists():
            existing = pd.read_parquet(path, dtype_backend="numpy_nullable")
            version = provenance.get("definitions_version")
            keep = existing["definitions_version"].notna() if version is None else existing["definitions_version"].ne(version).fillna(True)
            records = existing.loc[keep].astype(object).where(existing.loc[keep].notna(), None).to_dict("records") + records
        write_table(path, records)
        if written is not None:
            written.add(key)


def build_match(api, mid, interval, parse, query, save_demo, offline, fingerprint, prepared=None):
    started = time.perf_counter()
    folder = api.root / "raw" / str(mid)
    if prepared:
        payload, prov = prepared["metadata"]
    elif offline:
        payload = read_json(folder / "sources/api/metadata.json")
        prov = read_json(folder / "sources/api/metadata.json.provenance.json", {})
    else:
        payload, prov = api.json(f"/v1/matches/{mid}/metadata", f"raw/{mid}/sources/api/metadata.json",
                                {"disable_steam": "true"}, cached=True)
    if not payload:
        raise FileNotFoundError(f"No metadata archived for {mid}")
    prov = {"source": prov.get("source", prov.get("url")), "retrieved_at": prov.get("retrieved_at"),
            "metadata_retrieved_at": prov.get("retrieved_at")}
    audit_metadata(payload, folder / "sources/schema")
    replay = local_replay(folder, interval) if offline else acquire_replay(
        api, mid, interval, save_demo, parse, query, prefetched_salts=prepared.get("salts") if prepared else None,
        prefetched_errors=prepared.get("prefetch_errors") if prepared else None)
    if parse and not replay.get("replay_parsed"):
        if replay.get("unavailable_reason") and not replay.get("errors"):
            raise ReplayUnavailable(replay["unavailable_reason"])
        raise RuntimeError("; ".join(replay.get("errors", [])) or "Replay parsing did not succeed")
    version = replay_version(replay.get("parser_summary", {}))
    cache = api.__dict__.setdefault("_definitions_cache", {})
    cache_key = (version, offline)
    if cache_key in cache:
        static = cache[cache_key]
    elif offline:
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
    if all(name in static for name in ("items", "heroes", "ranks", "generic-data")):
        if cache_key not in cache and len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[cache_key] = static
    exact = version is not None and static.get("items_provenance", {}).get("definitions_version") == version
    prov.update(match_id=mid, game_version=version, definitions_version=version if exact else None)
    normalized_at = time.perf_counter()
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
    status["timings_seconds"] = {**replay.get("timings_seconds", {}),
                                 "normalize_validate": time.perf_counter() - normalized_at}
    if prepared:
        status["timings_seconds"]["prefetch"] = prepared["seconds"]
        if "download_seconds" in prepared:
            status["timings_seconds"]["prefetch_download"] = prepared["download_seconds"]
    saving_at = time.perf_counter()
    references(api.root, static, api.__dict__.setdefault("_written_references", set()))
    status = save_match(folder, tables, status)
    status["timings_seconds"]["save"] = time.perf_counter() - saving_at
    status["timings_seconds"]["processing"] = time.perf_counter() - started
    status["files"] = {f"{group}.zip": checksum(folder / f"{group}.zip") for group in GROUPS}
    save_json(folder / "sources/status.json", status)
    return status


def eligible_match(info):
    return (info.get("game_mode") == 1 and info.get("duration_s", 0) >= 900
            and len(info.get("players", [])) == 12 and info.get("winning_team") in (0, 1))


def prepare_candidate(api, mid, interval, parse, save_demo, automatic):
    # Only small metadata and compressed downloads run ahead of the one active parser.
    child = api.fork()
    started = time.perf_counter()
    result = {"match_id": mid}
    try:
        result["metadata"] = child.json(f"/v1/matches/{mid}/metadata", f"raw/{mid}/sources/api/metadata.json",
                                        {"disable_steam": "true"}, cached=True)
        if automatic and not eligible_match(result["metadata"][0]["match_info"]):
            return result
        if parse or save_demo:
            stage = "salts"
            try:
                salts = get_salts(child, mid)
                result["salts"] = salts
                folder = api.root / "raw" / str(mid) / "sources"
                cached_extract = parse and not save_demo and compatible_local_replay(folder / "replay", interval)
                if salts.get("demo_url") and not cached_extract:
                    stage = "download"
                    LOG.info("Match %s: prefetching replay", mid)
                    downloading_at = time.perf_counter()
                    try:
                        child.download(salts["demo_url"], folder / "replay.dem.bz2")
                    finally:
                        result["download_seconds"] = time.perf_counter() - downloading_at
                    LOG.info("Match %s: replay prefetch ready (%.1fs)", mid, result["download_seconds"])
            except requests.RequestException as exc:
                # Acquisition can still use a local demo or parsed extract.
                result["prefetch_errors"] = {stage: exc}
                LOG.info("Match %s prefetch unavailable; acquisition will check local sources: %s", mid, exc)
    except Exception as exc:
        result["error"] = exc
    finally:
        result["seconds"] = time.perf_counter() - started
        result["requests"] = child.requests
        child.session.close()
    return result


def prepared_candidates(api, selected, prepare, ahead, done):
    selected = iter(selected)
    future = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-prefetch") as pool:
        def submit():
            mid = next(selected, None)
            if mid is None:
                return None
            return pool.submit(lambda: {"match_id": mid} if done(mid) else prepare(mid))

        try:
            future = submit()
            while future is not None:
                result = future.result()
                future = submit() if ahead() else None
                yield result["match_id"], result
                if future is None:
                    future = submit()
        finally:
            if future is not None and not future.cancel():
                pending = future.result()
                folder = api.root / "raw" / str(pending["match_id"]) / "sources"
                try:
                    if pending.get("requests"):
                        save_json(folder / "prefetch_requests.json", pending["requests"])
                    compact_sources(folder)
                except Exception:
                    LOG.exception("Match %s prefetch source compaction failed; sources retained", pending["match_id"])


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
        reprocess=False, offline=False, prefetch=True, download_workers=4):
    if n_matches < 1 or sample_interval <= 0:
        raise ValueError("N_MATCHES and SAMPLE_INTERVAL must be positive")
    root = Path(output_dir)
    for name in ("raw", "processed", "tables"):
        (root / name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / "tables/run.log", encoding="utf-8")], force=True)
    api = API(root, save_raw_json, download_workers=download_workers)
    selection_api = api.fork()
    if not offline:
        try:
            api.json("/v1/assets/client-versions", "tables/sources/client_versions.json", required_archive=True)
        except Exception as exc:
            LOG.warning("Version index refresh failed; using archived index if available: %s", exc)
    fingerprint = signature(sample_interval, parse_replays, query_demo_api, save_raw_demos, save_raw_json)
    checked = {}
    automatic = not offline and match_ids is None

    def done(mid):
        if reprocess:
            return False
        if mid not in checked:
            checked[mid] = complete(root / "raw" / str(mid), fingerprint)
        return checked[mid]
    if offline and match_ids is None:
        selected = sorted(int(path.name) for path in (root / "raw").iterdir() if path.is_dir() and path.name.isdigit())
    else:
        selected = select_matches(selection_api, n_matches, match_ids, done, require_replay=parse_replays)
    totals = {"saved": 0, "skipped": 0, "failed": 0}
    if prefetch and not offline:
        selected = prepared_candidates(api, selected,
            lambda mid: prepare_candidate(api, mid, sample_interval, parse_replays, save_raw_demos, automatic),
            lambda: not automatic or totals["saved"] + 1 < n_matches, done)
    else:
        selected = ((mid, None) for mid in selected)
    run_started = time.perf_counter()
    try:
        for mid, prepared in selected:
            if done(mid):
                totals["skipped"] += 1
                LOG.info("Match %s already complete; skipping", mid)
                continue
            folder = root / "raw" / str(mid)
            try:
                if prepared:
                    api.requests.extend(prepared.get("requests", []))
                    if "error" in prepared:
                        raise prepared["error"]
                LOG.info("Match %s: download, normalize, save", mid)
                if not offline and match_ids is None:
                    payload, _ = prepared["metadata"] if prepared else api.json(f"/v1/matches/{mid}/metadata", f"raw/{mid}/sources/api/metadata.json",
                                          {"disable_steam": "true"}, cached=True)
                    info = payload["match_info"]
                    eligible = eligible_match(info)
                    if not eligible:
                        LOG.info("Match %s skipped: eligibility filter (mode=%s, duration=%ss, players=%s, winner=%s)",
                                 mid, info.get("game_mode"), info.get("duration_s"),
                                 len(info.get("players", [])), info.get("winning_team"))
                    del payload, info
                    if not eligible:
                        totals["skipped"] += 1
                        continue
                arguments = {"prepared": prepared} if prepared else {}
                status = build_match(api, mid, sample_interval, parse_replays, query_demo_api,
                                     save_raw_demos, offline, fingerprint, **arguments)
                update_catalog(root, status)
                totals["saved"] += 1
                checked[mid] = True
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
        selected.close()
        if selection_api.requests:
            save_json(root / "tables/sources/selection/requests.json", selection_api.requests)
        selection_api.session.close()
        api.session.close()
        compact_sources(root / "tables/sources")
        save_json(root / "tables/last_run.json", {**totals, "completed_at": now(),
                  "elapsed_seconds": time.perf_counter() - run_started,
                  "prefetch": prefetch and not offline, "download_workers": download_workers})
    LOG.info("Finished: %s", totals)
    if not offline and match_ids is None and totals["saved"] < n_matches:
        LOG.warning("Saved %s of %s requested matches; available replay candidates were exhausted", totals["saved"], n_matches)
    return totals
