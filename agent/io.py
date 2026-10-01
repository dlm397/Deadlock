import gzip
import hashlib
import io
import json
import logging
import re
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Event

import pandas as pd
import requests

from .lake_response import LakeUnavailable, query_rows

BASE = "https://api.deadlock-api.com"
LOG = logging.getLogger(__name__)
RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_DELAYS_SECONDS = (2, 4, 60)  # Last retry outlasts the API's one-minute failure cache.
PARALLEL_DOWNLOAD_MIN_BYTES = 16 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
GROUPS = {
    "summaries": ("matches", "players", "game_versions", "metadata_teams", "metadata_damage",
                  "metadata_custom_stats", "metadata_gold_sources"),
    "states": ("player_state", "team_state", "metadata_player_stats", "metadata_player_paths"),
    "purchases": ("purchases", "metadata_item_records"),
    "events": ("events", "metadata_objectives", "metadata_midboss", "legacy_events", "replay_event_counts"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = json_bytes(value)
    temporary.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if path.suffix == ".gz" else data)
    temporary.replace(path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    if not path.exists():
        return default
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as source:
        return json.load(source)


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _retry_after_seconds(response):
    header = response.headers.get("Retry-After") if response is not None else None
    if not header:
        return 0
    try:
        return float(header)
    except ValueError:
        try:
            return (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0


class _RangeUnavailable(ValueError):
    pass


class API:
    def __init__(self, root, save_raw=True, download_workers=4):
        self.root = Path(root)
        self.save_raw = save_raw
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "deadlock-local-research/0.2"
        self.requests = []
        self.mcp_headers = {"Accept": "application/json, text/event-stream"}
        self.mcp_ready = False
        self.download_workers = download_workers

    def fork(self):
        child = API(self.root, self.save_raw, self.download_workers)
        child.session.headers.update(self.session.headers)
        return child

    def _retry(self, operation, description, retry=True):
        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            try:
                return operation()
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError) as exc:
                response = exc.response
                status = response.status_code if response is not None else None
                if (not retry or attempt == len(RETRY_DELAYS_SECONDS)
                        or isinstance(exc, requests.HTTPError) and status not in RETRY_STATUSES):
                    raise
                delay = RETRY_DELAYS_SECONDS[attempt]
                requested = _retry_after_seconds(response)
                if requested > 60:
                    LOG.warning("%s: server requests a %.0fs wait; deferring this request", description, requested)
                    raise
                delay = max(delay, requested)
                if response is not None:
                    response.close()
                LOG.warning("%s: %s; retry %s/%s in %gs", description,
                            f"HTTP {status}" if status is not None else type(exc).__name__,
                            attempt + 1, len(RETRY_DELAYS_SECONDS), delay)
                time.sleep(delay)

    def _rpc(self, method, params=None, notification=False):
        body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notification:
            body["id"] = 1
        response = self.session.post(BASE + "/v1/mcp", json=body, headers=self.mcp_headers, timeout=(20, 180))
        self.requests.append({"url": response.url, "method": "POST", "retrieved_at": now(),
                              "status": response.status_code, "request_body": body,
                              "sha256": hashlib.sha256(response.content).hexdigest()})
        response.raise_for_status()
        if response.headers.get("Mcp-Session-Id"):
            self.mcp_headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
        if notification:
            return
        try:
            if response.text.lstrip().startswith("{"):
                messages = [response.json()]
            else:
                # Streamable HTTP can wrap a JSON-RPC response in an SSE event.
                messages = []
                for event in response.text.replace("\r\n", "\n").split("\n\n"):
                    data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
                    if data:
                        messages.append(json.loads(data))
        except ValueError as exc:
            raise LakeUnavailable("Data lake returned malformed JSON-RPC data", response=response) from exc
        message = next((item for item in messages if isinstance(item, dict) and item.get("id") == body["id"]), {})
        if "error" in message:
            raise RuntimeError(f"Data lake request failed: {message['error']}")
        if "result" not in message:
            raise LakeUnavailable("Data lake returned no JSON-RPC result", response=response)
        return message["result"]

    def lake_query(self, sql):
        # SQL reads are safe to retry; demo-job POST submissions still are not.
        return self._retry(lambda: self._lake_query(sql), "Data lake query")

    def _lake_query(self, sql):
        if not self.mcp_ready:
            result = self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                               "clientInfo": {"name": "deadlock-local-research", "version": "0.2"}})
            self.mcp_headers["MCP-Protocol-Version"] = result["protocolVersion"]
            self._rpc("notifications/initialized", notification=True)
            self.mcp_ready = True
        result = self._rpc("tools/call", {"name": "execute_query", "arguments": {"sql": sql}})
        return query_rows(result)

    def json(self, path, filename=None, params=None, body=None, required_archive=False, cached=False):
        out = self.root / filename if filename else None
        if cached and out and (out.exists() or out.with_name(out.name + ".gz").exists()):
            payload = read_json(out)
            record = read_json(out.with_suffix(out.suffix + ".provenance.json"), {})
            return payload, dict(source=record.get("url", record.get("source")), retrieved_at=record.get("retrieved_at"))
        url = path if path.startswith("http") else BASE + path
        def fetch():
            response = self.session.request("POST" if body is not None else "GET", url,
                                            params=params, json=body, timeout=(20, 180))
            record = dict(url=response.url, method=response.request.method, retrieved_at=now(),
                          status=response.status_code, request_body=body,
                          sha256=hashlib.sha256(response.content).hexdigest())
            self.requests.append(record)
            response.raise_for_status()
            return response, record

        response, record = self._retry(fetch, url, retry=body is None)
        payload = response.json()
        if out and (self.save_raw or required_archive):
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_suffix(out.suffix + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(out)
            save_json(out.with_suffix(out.suffix + ".provenance.json"), record)
        return payload, dict(source=response.url, retrieved_at=record["retrieved_at"])

    def _download_ranges(self, url, temporary, size, validator, workers):
        cancelled = Event()
        with temporary.open("wb") as dest:
            dest.truncate(size)

        def fetch_range(start, end):
            headers = {**self.session.headers, "Accept-Encoding": "identity",
                       "Range": f"bytes={start}-{end}", "If-Range": validator[1]}
            with requests.Session() as session:
                with session.get(url, headers=headers, stream=True, timeout=(20, 180)) as response:
                    if response.status_code in (200, 412, 416):
                        raise _RangeUnavailable("server declined a byte range")
                    response.raise_for_status()
                    actual_range = response.headers.get("Content-Range", "")
                    encoding = response.headers.get("Content-Encoding", "identity").lower()
                    if (response.status_code != 206 or actual_range != f"bytes {start}-{end}/{size}"
                            or encoding != "identity" or response.headers.get(validator[0]) != validator[1]):
                        raise _RangeUnavailable("range headers or entity validator changed")
                    expected = end - start + 1
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and content_length != str(expected):
                        raise _RangeUnavailable("incorrect range Content-Length")
                    received = 0
                    with temporary.open("r+b") as dest:
                        dest.seek(start)
                        for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                            if cancelled.is_set():
                                return
                            received += len(chunk)
                            if received > expected:
                                raise _RangeUnavailable("range body exceeds requested length")
                            dest.write(chunk)
                    if received != expected:
                        raise _RangeUnavailable("incomplete range body")

        # Each worker owns its connection and writes only its disjoint byte range.
        failure = None
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="replay-range") as pool:
            futures = [pool.submit(fetch_range, size * index // workers, size * (index + 1) // workers - 1)
                       for index in range(workers)]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException as exc:
                failure = exc
                cancelled.set()
                for future in futures:
                    future.cancel()
        if failure is not None:
            if isinstance(failure, Exception):
                errors = [future.exception() for future in futures if not future.cancelled()]
                http_errors = [exc for exc in errors if isinstance(exc, requests.HTTPError)]
                if http_errors:
                    # A simultaneous throttle takes precedence over a range fallback.
                    failure = max(http_errors, key=lambda exc: _retry_after_seconds(exc.response))
            raise failure
        return checksum(temporary)

    def download(self, url, target, workers=None):
        workers = self.download_workers if workers is None else workers
        if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
            raise ValueError("download workers must be an integer between 1 and 8")
        target = Path(target)
        label = f"match {target.parent.parent.name} replay" if target.name == "replay.dem.bz2" else target.name
        target.parent.mkdir(parents=True, exist_ok=True)
        sidecar = target.with_suffix(target.suffix + ".provenance.json")
        record = read_json(sidecar, {})
        if target.exists() and record.get("sha256") == checksum(target):
            LOG.info("Reusing verified cached download: %s", label)
            return target
        temporary = target.with_suffix(target.suffix + ".part")
        parallel_allowed = workers > 1

        def fetch():
            nonlocal parallel_allowed
            digest = hashlib.sha256()
            with self.session.get(url, headers={"Accept-Encoding": "identity"}, stream=True,
                                  timeout=(20, 180)) as response:
                response.raise_for_status()
                source = response.url
                length = response.headers.get("Content-Length", "")
                size = int(length) if length.isdigit() else None
                identity = response.headers.get("Content-Encoding", "identity").lower() == "identity"
                etag = response.headers.get("ETag", "")
                modified = response.headers.get("Last-Modified", "")
                validator = ("ETag", etag) if re.fullmatch(r'"[^"\r\n]+"', etag) else None
                if validator is None and modified:
                    try:
                        if parsedate_to_datetime(modified).tzinfo is not None:
                            validator = ("Last-Modified", modified)
                    except (ValueError, TypeError, OverflowError):
                        pass
                use_ranges = (parallel_allowed and response.status_code == 200 and identity
                              and size is not None and size >= PARALLEL_DOWNLOAD_MIN_BYTES
                              and response.headers.get("Accept-Ranges", "").lower() == "bytes" and validator)
                if use_ranges:
                    response.close()
                    LOG.info("Downloading %s: %.1f MiB with %s parallel streams", label, size / 1024**2, workers)
                    try:
                        sha256 = self._download_ranges(source, temporary, size, validator, workers)
                    except _RangeUnavailable as exc:
                        parallel_allowed = False
                        LOG.info("Replay range download unavailable (%s); using one stream", exc)
                        return fetch()
                    except requests.RequestException:
                        # The existing retry policy waits before restarting in one stream.
                        parallel_allowed = False
                        raise
                    record = dict(source=source, retrieved_at=now(), bytes=size, sha256=sha256)
                    temporary.replace(target)
                    save_json(sidecar, record)
                    return target
                if response.status_code != 200 or response.headers.get("Content-Range"):
                    raise requests.exceptions.ChunkedEncodingError("Expected a complete replay response")
                LOG.info("Downloading %s: one stream", label)
                with temporary.open("wb") as dest:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                        dest.write(chunk)
                        digest.update(chunk)
                if identity and size is not None and temporary.stat().st_size != size:
                    raise requests.exceptions.ChunkedEncodingError("Incomplete replay download")
                record = dict(source=response.url, retrieved_at=now(), bytes=temporary.stat().st_size,
                              sha256=digest.hexdigest())
            temporary.replace(target)
            save_json(sidecar, record)
            return target

        try:
            return self._retry(fetch, url)
        finally:
            temporary.unlink(missing_ok=True)


def frame(rows, columns=()):
    df = pd.DataFrame(rows, dtype=object) if rows else pd.DataFrame(columns=list(columns))
    for col in df.columns:
        if any(isinstance(x, (dict, list, tuple)) for x in df[col]):
            df[col] = df[col].map(lambda x: json_bytes(x).decode() if isinstance(x, (dict, list, tuple)) else x)
        kinds = {type(x) for x in df[col] if x is not None and not (isinstance(x, float) and pd.isna(x))}
        if len(kinds) > 1 and not kinds.issubset({int, float, bool}):
            df[col] = df[col].map(lambda x: str(x) if x is not None and not pd.isna(x) else None)
    return df


def write_table(path, rows, columns=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    df = frame(rows, columns)
    df.to_parquet(temporary, index=False, compression="zstd", compression_level=19)
    temporary.replace(path)
    return len(df)


def group_for(name):
    for group, names in GROUPS.items():
        if name in names:
            return group
    raise ValueError(f"Unassigned table: {name}")


def save_match(folder, tables, manifest):
    folder = Path(folder)
    for name in tables:
        group_for(name)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "row_counts": {name: len(tables.get(name, [])) for names in GROUPS.values() for name in names}}
    # Build and verify all four bundles before replacing any saved result.
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=folder) as temporary:
        stage = Path(temporary)
        for group, names in GROUPS.items():
            with zipfile.ZipFile(stage / f"{group}.zip", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name in names:
                    path = stage / f"{name}.parquet"
                    write_table(path, tables.get(name, []), ("match_id",))
                    archive.write(path, path.name)
                    path.unlink()
                if group == "summaries":
                    archive.writestr("manifest.json", json_bytes(manifest))
            with zipfile.ZipFile(stage / f"{group}.zip") as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Archive verification failed: {group}")
        installed, backups = [], []
        try:
            for group in GROUPS:
                target = folder / f"{group}.zip"
                if target.exists():
                    target.replace(stage / f"{group}.previous")
                    backups.append(group)
                (stage / target.name).replace(target)
                installed.append(group)
        except BaseException:
            for group in installed:
                (folder / f"{group}.zip").unlink(missing_ok=True)
            for group in backups:
                (stage / f"{group}.previous").replace(folder / f"{group}.zip")
            raise
    return manifest


def load_table(folder, name, columns=None):
    with zipfile.ZipFile(Path(folder) / f"{group_for(name)}.zip") as archive:
        with archive.open(f"{name}.parquet") as source:
            return pd.read_parquet(io.BytesIO(source.read()), columns=columns, dtype_backend="numpy_nullable")


def compact_sources(folder):
    # Compress exact JSON bytes; read_json also accepts the compressed name.
    for path in Path(folder).rglob("*.json"):
        target = path.with_suffix(".json.gz")
        temporary = target.with_suffix(".gz.tmp")
        with path.open("rb") as source, gzip.open(temporary, "wb", compresslevel=9) as dest:
            shutil.copyfileobj(source, dest, 1024 * 1024)
        with gzip.open(temporary, "rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != checksum(path):
                raise ValueError(f"Compression verification failed: {path}")
        temporary.replace(target)
        path.unlink()
