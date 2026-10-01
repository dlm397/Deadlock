import bz2
import hashlib
import tempfile
import logging
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zstandard

from .quality import audit_demo_schema
from .io import checksum, save_json, read_json

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
            query = ("SELECT DISTINCT match_id FROM match_salts WHERE replay_salt IS NOT NULL AND metadata_salt IS NOT NULL "
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


def compatible_local_replay(out, interval):
    out = Path(out)
    previous = read_json(out / "summary.json", {})
    script = Path(__file__).resolve().parent / "replay/export.mjs"
    if (previous.get("success") and previous.get("sample_interval") == interval
            and previous.get("exporter_sha256") == hashlib.sha256(script.read_bytes()).hexdigest()
            and all((out / name).exists() for name in
                    ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz"))
            and read_json(out / "schema.json")):
        return previous
    return None


def audit_replay(api, mid, folder, query):
    started = time.perf_counter()
    result = {"replay_available": False, "errors": []}
    schema = None
    try:
        schema, _ = api.json("/v1/matches/demo/schema", f"raw/{mid}/sources/schema/api_demo_schema.json",
                             params={"match_id": mid}, cached=True)
        audit_demo_schema(schema, folder)
        result["replay_available"] = True
    except Exception as exc:
        result["errors"].append(f"demo schema: {exc}")
        LOG.warning("Match %s demo schema: %s", mid, exc)
    if query and schema:
        try:
            result["demo_query"] = query_demo(api, mid, schema)
        except Exception as exc:
            result["errors"].append(f"API demo query: {exc}")
            LOG.warning("Match %s API demo query failed: %s", mid, exc)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def acquire_local_replay(api, mid, interval, save_demo, parse, result, download_error=None):
    folder = api.root / "raw" / str(mid) / "sources"
    compressed, out = folder / "replay.dem.bz2", folder / "replay"
    url = result.get("demo_url")
    previous = compatible_local_replay(out, interval) if parse else None
    if previous:
        result.update(parser_summary=previous, replay_parsed=True, parser_output=str(out),
                      replay_available=True, parser_reused=True)
        LOG.info("Match %s: reusing compatible replay extracts", mid)
    if not url and not compressed.exists() and not previous:
        result["unavailable_reason"] = "No replay URL is available for this match; metadata availability does not imply replay availability"
    if (url or compressed.exists()) and (save_demo or parse and not previous):
        expanded = compressed.with_suffix("")
        try:
            if download_error is not None:
                record = read_json(compressed.with_suffix(compressed.suffix + ".provenance.json"), {})
                if not compressed.exists() or record.get("sha256") != checksum(compressed):
                    raise download_error
                LOG.info("Match %s: using verified local replay after prefetch failed", mid)
            elif url:
                started = time.perf_counter()
                try:
                    api.download(url, compressed)
                finally:
                    result["timings_seconds"]["download"] = time.perf_counter() - started
            result["replay_available"] = True
            result["demo_bytes"] = compressed.stat().st_size
            if parse and not result["replay_parsed"]:
                started = time.perf_counter()
                try:
                    decompress_demo(compressed, expanded)
                finally:
                    result["timings_seconds"]["decompress"] = time.perf_counter() - started
                out.mkdir(parents=True, exist_ok=True)
                script = Path(__file__).resolve().parent / "replay/export.mjs"
                LOG.info("Match %s: parsing replay locally", mid)
                logpath = folder / "parser.log"
                logpath.parent.mkdir(parents=True, exist_ok=True)
                if not (script.parent / "node_modules/deadem").exists():
                    subprocess.run(["npm.cmd" if __import__("os").name == "nt" else "npm", "ci",
                                    "--ignore-scripts", "--no-audit", "--no-fund"],
                                   cwd=script.parent, check=True, timeout=300)
                with tempfile.TemporaryDirectory(prefix=".parse-", dir=folder) as temporary:
                    staging = Path(temporary) / "replay"
                    staging.mkdir()
                    with logpath.open("w", encoding="utf-8") as log:
                        started = time.perf_counter()
                        try:
                            proc = subprocess.run(["node", "--max-old-space-size=6144", str(script),
                                                   str(expanded.resolve()), str(staging.resolve()), str(interval)],
                                                  stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                        finally:
                            result["timings_seconds"]["parse"] = time.perf_counter() - started
                    summary = read_json(staging / "summary.json", {})
                    required = ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz", "schema.json")
                    if proc.returncode or not summary.get("success") or not all((staging / name).exists() for name in required):
                        raise RuntimeError(f"Parser failed; see {logpath}")
                    out.resolve().relative_to(folder.resolve())
                    staging.resolve().relative_to(folder.resolve())
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
    elif previous:
        record = read_json(compressed.with_suffix(compressed.suffix + ".provenance.json"), {})
        result["demo_bytes"] = compressed.stat().st_size if compressed.exists() else record.get("bytes", 0)
        if not save_demo:
            compressed.unlink(missing_ok=True)


def acquire_replay(api, mid, interval, save_demo=True, parse=True, query=True, prefetched_salts=None,
                   prefetched_errors=None):
    started = time.perf_counter()
    result = {"replay_available": False, "replay_parsed": False, "errors": [],
              "timings_seconds": {"download": 0.0, "decompress": 0.0, "parse": 0.0, "remote_audit": 0.0}}
    folder = api.root / "raw" / str(mid) / "sources/schema"
    prefetched_errors = prefetched_errors or {}
    try:
        if "salts" in prefetched_errors:
            result["errors"].append(f"salts: {prefetched_errors['salts']}")
            salts = prefetched_salts or {}
        else:
            salts = get_salts(api, mid) if prefetched_salts is None else prefetched_salts
        result["demo_url"] = salts.get("demo_url")
    except Exception as exc:
        result["errors"].append(f"salts: {exc}")
    # A separate session keeps network audit work independent of local parsing.
    remote_api, executor, future = None, None, None
    if result.get("demo_url") and callable(getattr(api, "fork", None)):
        remote_api = api.fork()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-audit")
        future = executor.submit(audit_replay, remote_api, mid, folder, query)
    try:
        acquire_local_replay(api, mid, interval, save_demo, parse, result, prefetched_errors.get("download"))
        remote = None
        if future:
            remote = future.result()
        elif result.get("demo_url"):
            remote = audit_replay(api, mid, folder, query)
        if remote:
            result["replay_available"] |= remote["replay_available"]
            result["errors"].extend(remote["errors"])
            result["timings_seconds"]["remote_audit"] = remote["elapsed_seconds"]
            if "demo_query" in remote:
                result["demo_query"] = remote["demo_query"]
    finally:
        if executor:
            executor.shutdown(wait=True)
            api.requests.extend(remote_api.requests)
            remote_api.session.close()
    result["timings_seconds"]["total"] = time.perf_counter() - started
    LOG.info("Match %s replay timings: %s", mid,
             ", ".join(f"{'download/cache' if name == 'download' else name}={elapsed:.1f}s"
                       for name, elapsed in result["timings_seconds"].items()))
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
