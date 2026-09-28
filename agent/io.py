import gzip
import hashlib
import io
import json
import logging
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.deadlock-api.com"
LOG = logging.getLogger(__name__)
RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_DELAYS_SECONDS = (2, 4, 60)  # Last retry outlasts the API's one-minute failure cache.
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


class API:
    def __init__(self, root, save_raw=True):
        self.root = Path(root)
        self.save_raw = save_raw
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "deadlock-local-research/0.2"
        self.requests = []
        self.mcp_headers = {"Accept": "application/json, text/event-stream"}
        self.mcp_ready = False

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
                header = response.headers.get("Retry-After") if response is not None else None
                if header:
                    try:
                        requested = float(header)
                    except ValueError:
                        try:
                            requested = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            requested = 0
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
        if response.text.lstrip().startswith("{"):
            messages = [response.json()]
        else:
            # Streamable HTTP can wrap a JSON-RPC response in an SSE event.
            messages = []
            for event in response.text.replace("\r\n", "\n").split("\n\n"):
                data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
                if data:
                    messages.append(json.loads(data))
        message = next((item for item in messages if item.get("id") == body["id"]), {})
        if "error" in message or "result" not in message:
            raise RuntimeError(f"Data lake request failed: {message.get('error', 'missing RPC result')}")
        return message["result"]

    def lake_query(self, sql):
        if not self.mcp_ready:
            result = self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                               "clientInfo": {"name": "deadlock-local-research", "version": "0.2"}})
            self.mcp_headers["MCP-Protocol-Version"] = result["protocolVersion"]
            self._rpc("notifications/initialized", notification=True)
            self.mcp_ready = True
        result = self._rpc("tools/call", {"name": "execute_query", "arguments": {"sql": sql}})
        text = result.get("structuredContent", {}).get("result")
        if text is None:
            text = next((item["text"] for item in result.get("content", []) if item.get("type") == "text"), "")
        if result.get("isError"):
            raise RuntimeError(f"Data lake query failed: {text[:500]}")
        table = json.loads(text)
        if not table.get("success"):
            raise RuntimeError(f"Data lake query failed: {table.get('error', table)}")
        return [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]

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

    def download(self, url, target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        sidecar = target.with_suffix(target.suffix + ".provenance.json")
        record = read_json(sidecar, {})
        if target.exists() and record.get("sha256") == checksum(target):
            return target
        temporary = target.with_suffix(target.suffix + ".part")
        def fetch():
            digest = hashlib.sha256()
            with self.session.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as dest:
                    for chunk in response.iter_content(1024 * 1024):
                        dest.write(chunk)
                        digest.update(chunk)
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
