from pathlib import Path

import pandas as pd

import main as settings
from .dataset import match_ids, save_match, write_table
from .normalize import normalize, replay_version
from .replay import parse_replay
from .sources import download, fetch_json, find_matches, get_assets, now, read_json, save_json, report_error


def save_references(root, definitions):
    for name in ("items", "heroes", "abilities"):
        source = "items" if name == "abilities" else name
        provenance = definitions.get(source + "_provenance", {})
        rows = definitions.get(source, [])
        if name == "abilities":
            rows = [row for row in rows if row.get("type") == "ability"]
        if not rows:
            continue
        path = Path(root) / "tables" / f"{name}.parquet"
        records = [{**row, **provenance} for row in rows]
        if path.exists():
            previous = pd.read_parquet(path, dtype_backend="numpy_nullable")
            if previous.definitions_version.eq(provenance["definitions_version"]).any():
                continue
            records += previous.astype(object).where(previous.notna(), None).to_dict("records")
        temporary = path.with_suffix(".tmp")
        write_table(temporary, records)
        temporary.replace(path)


def collect_match(mid, config=settings, cache_dir=None, replay_folder=None, automatic=False):
    root, cache = Path(config.OUTPUT_DIR), Path(cache_dir or config.OUTPUT_DIR)
    folder, archived = root / "raw" / str(mid), cache / "raw" / str(mid)
    sources, cached_sources = folder / "sources", archived / "sources"
    payload, provenance = fetch_json(f"/v1/matches/{mid}/metadata",
        sources / "api/metadata.json" if config.SAVE_RAW_JSON else None,
        {"disable_steam": "true"}, cached_sources / "api/metadata.json", config.OFFLINE)
    info = payload["match_info"]
    assert info["match_id"] == mid, "Metadata belongs to another match"
    print("    metadata: OK")
    if automatic:
        reasons = []
        if config.GAME_MODE == "Normal" and info.get("game_mode") != 1:
            reasons.append(f"game mode {info.get('game_mode')} (Normal required)")
        if info["duration_s"] < config.MIN_DURATION_S:
            reasons.append(f"duration {info['duration_s']}s (minimum {config.MIN_DURATION_S}s)")
        if config.EXPECTED_PLAYERS and len(info["players"]) != config.EXPECTED_PLAYERS:
            reasons.append(f"{len(info['players'])} players ({config.EXPECTED_PLAYERS} required)")
        if reasons:
            print(f"    SKIPPED: {'; '.join(reasons)}")
            return None

    replay_folder = Path(replay_folder) if replay_folder else None
    summary = read_json(cached_sources / "replay/summary.json", {})
    extracts = cached_sources / "replay"
    complete_extracts = (summary.get("success") and summary.get("sample_interval") == config.SAMPLE_INTERVAL
        and all((extracts / name).is_file() for name in
                ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz"))
        and read_json(extracts / "schema.json") is not None)
    reuse = complete_extracts and (config.OFFLINE or not config.REPROCESS)
    compressed = cached_sources / "replay.dem.bz2"
    needs_parse = config.PARSE_REPLAYS and replay_folder is None and not reuse
    if config.SAVE_RAW_DEMOS or (needs_parse and not config.OFFLINE):
        # Retained demos work even after the remote URL expires.
        if not compressed.is_file():
            salts, _ = fetch_json(f"/v1/matches/{mid}/salts",
                sources / "api/salts.json" if config.SAVE_RAW_JSON else None,
                {"disable_steam": "true"}, cached_sources / "api/salts.json", config.OFFLINE)
            if not salts.get("demo_url"):
                raise ValueError("Replay unavailable; disable SAVE_RAW_DEMOS for metadata-only collection")
            compressed = download(salts["demo_url"], sources / "replay.dem.bz2", offline=config.OFFLINE)
    if config.PARSE_REPLAYS and replay_folder is None:
        if config.OFFLINE or reuse:
            assert complete_extracts, "Archived replay is incomplete or uses another sampling interval; reparse with OFFLINE=False"
            replay_folder = cached_sources / "replay"
        else:
            print("    replay: OK; parsing...")
            replay_folder = parse_replay(compressed, sources / "replay", config.SAMPLE_INTERVAL)
            receipt = read_json(str(compressed) + ".provenance.json")
            if receipt and compressed.parent.resolve() != sources.resolve():
                save_json(sources / "replay.dem.bz2.provenance.json.gz", receipt)
        summary = read_json(replay_folder / "summary.json")
        print("    parsed: OK")
    if replay_folder is not None:
        summary = read_json(replay_folder / "summary.json")
        assert summary["success"] and summary["sample_interval"] == config.SAMPLE_INTERVAL
    version = replay_version(summary)
    definitions = get_assets(version, root, cache, config.OFFLINE)
    provenance.update(match_id=mid, game_version=version,
                      definitions_version=version if definitions.get("items") else None)
    tables = normalize(payload, replay_folder, definitions, provenance, config.SAMPLE_INTERVAL)
    save_references(root, definitions)
    status = save_match(folder, tables, {"match_id": mid, "game_version": version,
        "sample_interval": config.SAMPLE_INTERVAL, "replay_parsed": replay_folder is not None,
        "completed_at": now(), "source_directory": str(cached_sources.resolve()),
        "replay_directory": str(replay_folder.resolve()) if replay_folder else None})
    if not config.SAVE_RAW_DEMOS and compressed.parent.resolve() == sources.resolve():
        compressed.unlink(missing_ok=True)
    print(f"    saved: OK ({status['row_counts']['player_state']} state rows)")
    return status


def run(config=settings):
    assert config.N_MATCHES > 0 and config.SAMPLE_INTERVAL > 0
    if config.MATCH_IDS is None and config.GAME_MODE not in (None, "Normal"):
        raise ValueError("Detailed automatic selection supports Normal or None; use MATCH_IDS for other modes")
    completed = set() if config.REPROCESS else set(match_ids(config.OUTPUT_DIR))
    if config.OFFLINE and config.MATCH_IDS is None:
        selected = sorted(int(p.name) for p in (Path(config.OUTPUT_DIR) / "raw").glob("*") if p.name.isdigit())
    else:
        selected = find_matches(config.N_MATCHES, config.MATCH_IDS, completed)
    saved = failed = skipped = already_saved = 0
    for mid in selected:
        if mid in completed:
            already_saved += 1
            continue
        print(f"Processing match {mid}...")
        try:
            status = collect_match(mid, config, automatic=config.MATCH_IDS is None and not config.OFFLINE)
            if status is None:
                skipped += 1
            else:
                saved += 1
        except Exception as error:
            failed += 1
            report_error(error, Path(config.OUTPUT_DIR) / "raw" / str(mid) / "sources/error.log")
        if config.MATCH_IDS is None and not config.OFFLINE and saved >= config.N_MATCHES:
            break
    target = len(config.MATCH_IDS) if config.MATCH_IDS is not None else len(selected) if config.OFFLINE else config.N_MATCHES
    print(f"Completed: {saved}/{target} | Skipped: {skipped} | Failed: {failed} | Already saved: {already_saved}")
    return {"saved": saved, "failed": failed, "skipped": skipped, "already_saved": already_saved}


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        report_error(error, Path(settings.OUTPUT_DIR) / "tables/collection_error.log", "Collection stopped")
        raise SystemExit(1) from None
