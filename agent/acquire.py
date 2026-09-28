import bz2
import json
import tempfile
import logging
import shutil
import subprocess
import time
from pathlib import Path
import zstandard

from .quality import audit_demo_schema
from .io import save_json, read_json

LOG = logging.getLogger(__name__)


def decompress_demo(compressed, expanded):
    # Current Valve URLs still end .bz2 but sampled files are Zstandard frames.
    # Detect bytes, never infer the codec from the misleading file extension.
    with open(compressed, "rb") as source, open(expanded, "wb") as destination:
        magic = source.read(4)
        source.seek(0)
        if magic == b"\x28\xb5\x2f\xfd":
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                shutil.copyfileobj(reader, destination, 1024 * 1024)
        elif magic[:3] == b"BZh":
            with bz2.open(source, "rb") as reader:
                shutil.copyfileobj(reader, destination, 1024 * 1024)
        elif magic == b"PBDE":
            shutil.copyfileobj(source, destination, 1024 * 1024)
        else:
            raise ValueError(f"Unknown replay compression magic: {magic.hex()}")


class ReplayUnavailable(RuntimeError):
    pass


def get_salts(api, mid):
    filename = f"raw/{mid}/sources/api/salts.json"
    cached = read_json(api.root / filename)
    if cached and cached.get("demo_url"):
        return cached
    salts, _ = api.json(f"/v1/matches/{mid}/salts", filename, {"disable_steam": "true"})
    return salts


def select_matches(api, n, match_ids, completed=None, require_replay=True):
    completed = completed or (lambda mid: False)
    if match_ids is not None:
        for mid in dict.fromkeys(int(x) for x in match_ids):
            yield mid
        return
    seen = set()
    limit = min(1024, max(30, n * 4))
    candidate_budget = max(100, n * 10)
    considered = 0
    cursor = None
    try:
        while True:
            before = f" AND match_id < {cursor}" if cursor is not None else ""
            query = ("SELECT DISTINCT match_id FROM match_salts WHERE replay_salt IS NOT NULL "
                     f"AND cluster_id IS NOT NULL{before} ORDER BY match_id DESC LIMIT {limit}")
            rows = api.lake_query(query)
            save_json(api.root / "tables/sources/selection/replay_candidates.json",
                      {"source": "https://api.deadlock-api.com/v1/mcp", "query": query, "rows": rows})
            if not rows:
                break
            ids = sorted({int(row["match_id"]) for row in rows}, reverse=True)
            if cursor is not None and max(ids) >= cursor:
                raise ValueError("Data lake returned a non-advancing match page")
            for mid in ids:
                if mid not in seen and not completed(mid):
                    considered += 1
                    yield mid
                seen.add(mid)
                if considered >= candidate_budget:
                    break
            cursor = min(ids)
            if len(rows) < limit or considered >= candidate_budget:
                break
    except Exception as exc:
        LOG.warning("Data lake selection unavailable: %s; checking recent matches", exc)
    try:
        recent, _ = api.json("/v1/matches/recently-fetched", "tables/sources/selection/recently_fetched.json")
        for row in sorted(recent, key=lambda r: r.get("start_time", 0), reverse=True):
            mid = int(row["match_id"])
            if mid in seen or completed(mid) or row.get("game_mode") != 1 or row.get("duration_s", 0) < 900:
                continue
            seen.add(mid)
            if require_replay:
                try:
                    if not get_salts(api, mid).get("demo_url"):
                        continue
                except Exception as exc:
                    LOG.info("Match %s: replay availability check failed: %s", mid, exc)
                    continue
            yield mid
    except Exception as exc:
        LOG.warning("Recent-match selection unavailable: %s", exc)


def query_demo(api, mid, schema):
    # Exercise the actual demo SQL service and archive an independent kill audit.
    tables = {t["name"]: t for t in schema["tables"]}
    if "HeroKilledEvent" not in tables:
        raise ValueError("This demo schema does not advertise HeroKilledEvent")
    # DataFusion registers identifiers case-folded even though schema displays
    # Rust type names. An unquoted identifier resolves correctly.
    query = 'SELECT * FROM HeroKilledEvent'
    folder = f"raw/{mid}/sources/api/demo_query"
    status_path = api.root / folder / "status.json"
    submission_path = api.root / folder / "submission.json.provenance.json"
    artifact_path = api.root / folder / "hero_killed.parquet"
    if read_json(status_path) and read_json(submission_path) and artifact_path.exists():
        status = read_json(status_path)
        request = read_json(submission_path)
        if status.get("status") == "done" and request.get("request_body", {}).get("query") == query:
            return {"status": "done", "job_id": status["job_id"], "path": str(artifact_path), "query": query}
    job, _ = api.json("/v1/matches/demo/query", folder + "/submission.json",
                      body={"match_id": mid, "query": query, "format": "parquet"}, required_archive=True)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        status, _ = api.json(f"/v1/matches/demo/query/{job['job_id']}", folder + "/status.json",
                             required_archive=True)
        if status["status"] == "done":
            path = api.download(status["result_url"], api.root / folder / "hero_killed.parquet")
            return {"status": "done", "job_id": job["job_id"], "path": str(path), "query": query}
        if status["status"] == "failed":
            raise RuntimeError(status.get("error", "demo query failed"))
        LOG.info("Match %s API demo query %s (estimated wait %ss)", mid, status["status"], status.get("estimated_wait_seconds"))
        time.sleep(10)
    raise TimeoutError(f"Demo query job {job['job_id']} did not finish in 15 minutes")


def acquire_replay(api, mid, interval, save_demo=True, parse=True, query=True):
    result = {"replay_available": False, "replay_parsed": False, "errors": []}
    folder = api.root / "raw" / str(mid) / "sources/schema"
    schema = None
    salts = {}
    compressed = folder.parent / "replay.dem.bz2"
    out = folder.parent / "replay"
    try:
        salts = get_salts(api, mid)
        result["demo_url"] = salts.get("demo_url")
    except Exception as exc:
        result["errors"].append(f"salts: {exc}")
    try:
        # Only ask for a schema when a replay salt is known: avoids Steam fallback.
        if salts.get("demo_url"):
            schema, _ = api.json("/v1/matches/demo/schema", f"raw/{mid}/sources/schema/api_demo_schema.json", params={"match_id": mid}, cached=True)
            audit_demo_schema(schema, folder)
            result["replay_available"] = True
    except Exception as exc:
        result["errors"].append(f"demo schema: {exc}")
        LOG.warning("Match %s demo schema: %s", mid, exc)
    url = salts.get("demo_url") or (schema or {}).get("demo_url")
    if not url and not compressed.exists():
        result["unavailable_reason"] = "No replay URL is available for this match; metadata availability does not imply replay availability"
    if (url or compressed.exists()) and (save_demo or parse):
        expanded = compressed.with_suffix("")
        try:
            if url:
                LOG.info("Match %s: downloading replay", mid)
                api.download(url, compressed)
            result["replay_available"] = True
            result["demo_bytes"] = compressed.stat().st_size
            previous_summary = out / "summary.json"
            if parse and read_json(previous_summary):
                previous = read_json(previous_summary)
                # Avoid re-parsing an already successful local artifact with the
                # same interval and exporter. No partial/resumable processing.
                import hashlib
                script = Path(__file__).resolve().parent / "replay/export.mjs"
                if (previous.get("success") and previous.get("sample_interval") == interval
                        and previous.get("exporter_sha256") == hashlib.sha256(script.read_bytes()).hexdigest()
                        and all((out / name).exists() for name in ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz"))
                        and read_json(out / "schema.json")):
                    result.update(parser_summary=previous, replay_parsed=True, parser_output=str(out))
            if parse and not result["replay_parsed"]:
                decompress_demo(compressed, expanded)
                out.mkdir(parents=True, exist_ok=True)
                script = Path(__file__).resolve().parent / "replay/export.mjs"
                LOG.info("Match %s: parsing replay locally", mid)
                logpath = folder.parent / "parser.log"
                logpath.parent.mkdir(parents=True, exist_ok=True)
                if not (script.parent / "node_modules/deadem").exists():
                    subprocess.run(["npm.cmd" if __import__("os").name == "nt" else "npm", "ci",
                                    "--ignore-scripts", "--no-audit", "--no-fund"],
                                   cwd=script.parent, check=True, timeout=300)
                with tempfile.TemporaryDirectory(prefix=".parse-", dir=folder.parent) as temporary:
                    staging = Path(temporary) / "replay"
                    staging.mkdir()
                    with logpath.open("w", encoding="utf-8") as log:
                        proc = subprocess.run(["node", "--max-old-space-size=6144", str(script),
                                               str(expanded.resolve()), str(staging.resolve()), str(interval)],
                                              stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                    summary = read_json(staging / "summary.json", {})
                    required = ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz", "schema.json")
                    if proc.returncode or not summary.get("success") or not all((staging / name).exists() for name in required):
                        raise RuntimeError(f"Parser failed; see {logpath}")
                    out.resolve().relative_to(folder.parent.resolve())
                    staging.resolve().relative_to(folder.parent.resolve())
                    previous = Path(temporary) / "previous"
                    if out.exists():
                        out.replace(previous)
                    try:
                        staging.replace(out)
                    except BaseException:
                        if previous.exists():
                            previous.replace(out)
                        raise
                result.update(parser_summary=summary, replay_parsed=True, parser_output=str(out))
        except Exception as exc:
            result["errors"].append(f"local replay: {exc}")
            LOG.exception("Match %s replay failed; retaining metadata", mid)
        finally:
            expanded.unlink(missing_ok=True)
            if not save_demo:
                compressed.unlink(missing_ok=True)
    if query and schema:
        try:
            result["demo_query"] = query_demo(api, mid, schema)
        except Exception as exc:
            result["errors"].append(f"API demo query: {exc}")
            LOG.warning("Match %s API demo query failed: %s", mid, exc)
    save_json(folder / "replay_status.json", result)
    return result


def static_data(api, version):
    label = str(version) if version is not None else "unmatched_latest"
    params = {"client_version": version} if version is not None else None
    result = {}
    for endpoint in ("items", "heroes", "ranks", "generic-data"):
        try:
            payload, prov = api.json(f"/v1/assets/{endpoint}", f"tables/sources/{label}/{endpoint}.json",
                                     params, required_archive=True, cached=True)
            result[endpoint] = payload
            result[endpoint + "_provenance"] = {**prov, "definitions_version": version,
                                                "version_match": version is not None}
        except Exception as exc:
            LOG.warning("Assets %s at version %s unavailable: %s", endpoint, label, exc)
    return result
