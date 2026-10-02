import gzip
import hashlib
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

BASE = "https://api.deadlock-api.com"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "deadlock-personal-research/1"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def error_message(error):
    if isinstance(error, requests.HTTPError) and error.response is not None:
        code = error.response.status_code
        url = urlsplit(error.response.url or "")
        endpoint = url.path.rstrip("/").rsplit("/", 1)[-1] or "request"
        service = "Deadlock API" if url.hostname == "api.deadlock-api.com" else "Download server"
        if code in (429, 500, 502, 503, 504):
            return f"{service} temporarily unavailable (HTTP {code}, {endpoint}). Try this match again later."
        return f"{service} rejected {endpoint} (HTTP {code})."
    if isinstance(error, requests.Timeout):
        return "The server took too long to respond. Try this match again later."
    if isinstance(error, requests.ConnectionError):
        return "Could not connect to the server. Check your connection and try again later."
    message = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(error))
    return " ".join(message.split())[:300] or type(error).__name__


def report_error(error, path, label="FAILED"):
    print(f"    {label}: {error_message(error)}")
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        print(f"    Details: {path}")
    except OSError as log_error:
        print(f"    Could not save error details: {error_message(log_error)}")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + ".gz")
    if not path.exists():
        return default
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8-sig") as file:
        return json.load(file)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if path.suffix == ".gz" else data)
    temporary.replace(path)


def fetch_json(url, path=None, params=None, cache_path=None, offline=False, refresh=False):
    for cached in (() if refresh else (cache_path, path)):
        if cached is not None:
            payload = read_json(cached)
            if payload is not None:
                name = str(cached).removesuffix(".gz") + ".provenance.json"
                provenance = read_json(name, {})
                return payload, {**provenance, "source": provenance.get("source", provenance.get("url"))}
    if offline:
        raise FileNotFoundError(f"No archived source: {cache_path or path or url}")
    response = SESSION.get(url if url.startswith("http") else BASE + url, params=params, timeout=(20, 180))
    response.raise_for_status()
    payload = response.json()
    provenance = {"source": response.url, "url": response.url, "retrieved_at": now(),
                  "sha256": hashlib.sha256(response.content).hexdigest()}
    if path is not None:
        path = str(path).removesuffix(".gz")
        archive = Path(path + ".gz")
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(archive) + ".tmp")
        temporary.write_bytes(gzip.compress(response.content, compresslevel=9, mtime=0))
        temporary.replace(archive)
        save_json(path + ".provenance.json.gz", provenance)
    return payload, provenance


def download(url, path, cache_path=None, offline=False):
    path = Path(path)
    for cached in (cache_path, path):
        if cached is not None and Path(cached).is_file():
            return Path(cached)
    if offline:
        raise FileNotFoundError(f"No archived replay: {cache_path or path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".part")
    digest, size = hashlib.sha256(), 0
    with SESSION.get(url, stream=True, headers={"Accept-Encoding": "identity"}, timeout=(20, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as file:
            for chunk in response.iter_content(1024 * 1024):
                file.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if "Content-Length" in response.headers and size != int(response.headers["Content-Length"]):
            raise ValueError("Incomplete replay download")
    temporary.replace(path)
    save_json(str(path) + ".provenance.json.gz", {"source": url, "retrieved_at": now(),
                                                "sha256": digest.hexdigest(), "bytes": size})
    return path


def get_assets(version, output_dir, cache_dir=None, offline=False):
    if version is None:
        return {}
    root, cache = Path(output_dir), Path(cache_dir or output_dir)
    names = ("items", "heroes", "ranks", "generic-data")
    archived = cache / "tables/sources" / str(version)
    if not all(read_json(archived / f"{name}.json") is not None for name in names):
        versions, _ = fetch_json("/v1/assets/client-versions", root / "tables/sources/client_versions.json",
                                 cache_path=cache / "tables/sources/client_versions.json", offline=offline)
        if version not in versions and not offline:
            versions, _ = fetch_json("/v1/assets/client-versions", root / "tables/sources/client_versions.json", refresh=True)
        if version not in versions:
            return {}  # Never attach current definitions to a historical replay.
    result = {}
    for name in names:
        result[name], provenance = fetch_json(f"/v1/assets/{name}",
            root / "tables/sources" / str(version) / f"{name}.json", {"client_version": version},
            archived / f"{name}.json", offline=offline)
        result[name + "_provenance"] = {**provenance, "definitions_version": version, "version_match": True}
    return result


def _cells(line, delimiter=","):
    # JSON decoding handles quoted TOON strings, including escaped commas/quotes.
    values, position = [], 0
    while position < len(line):
        while position < len(line) and line[position] == " ":
            position += 1
        if position < len(line) and line[position] == '"':
            value, position = json.JSONDecoder().raw_decode(line, position)
        else:
            end = line.find(delimiter, position)
            end = len(line) if end < 0 else end
            token, position = line[position:end].strip(), end
            if not token:
                raise ValueError("Empty unquoted SQL value")
            typed = token in ("true", "false", "null", "[]") or re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?", token)
            value = json.loads(token) if typed else token
        values.append(value)
        if position < len(line):
            if line[position] != delimiter:
                raise ValueError("Malformed SQL row")
            position += 1
            if position == len(line):
                raise ValueError("Empty unquoted SQL value")
    return values


def query_rows(result):
    payload = result.get("structuredContent") or {}
    payload = payload.get("result", payload if "success" in payload else None)
    if payload is None:
        payload = "\n".join(block["text"] for block in result.get("content", []) if block.get("type") == "text")
    if result.get("isError"):
        raise RuntimeError(str(payload))
    if isinstance(payload, str) and payload.lstrip().startswith("{"):
        payload = json.loads(payload)
    elif isinstance(payload, str):
        lines, payload, position = payload.splitlines(), {}, 0
        while position < len(lines):
            line = lines[position]
            position += 1
            if not line.strip():
                continue
            header = re.fullmatch(r"(\w+)(?:\[(\d+)([,|\t])?\])?(?:\{(.*)\})?:\s?(.*)", line)
            if header is None:
                raise ValueError("Malformed SQL response")
            key, count, delimiter, fields, inline = header.groups()
            if key in payload:
                raise ValueError("Duplicate SQL response field")
            delimiter = delimiter or ","
            if fields is not None:
                names, values = _cells(fields, delimiter), []
                while position < len(lines) and lines[position].startswith("  "):
                    cells = _cells(lines[position][2:], delimiter)
                    values.append(dict(zip(names, cells, strict=True)))
                    position += 1
            elif count is not None:
                values = _cells(inline, delimiter) if inline else []
            else:
                values = _cells(inline, "\x00")[0]
            if count is not None and len(values) != int(count):
                raise ValueError("Truncated SQL response")
            payload[key] = values
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError(f"Data lake query failed: {payload}")
    columns, rows = payload["columns"], payload["rows"]
    if payload.get("truncated") or payload.get("rowCount", len(rows)) != len(rows) or len(set(columns)) != len(columns):
        raise ValueError("Truncated or malformed SQL table")
    records = [dict(zip(columns, row, strict=True)) if isinstance(row, list) else row for row in rows]
    if any(set(row) != set(columns) for row in records):
        raise ValueError("SQL columns do not match rows")
    return records


def _rpc(method, params=None, notification=False):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if not notification:
        body["id"] = 1
    response = SESSION.post(BASE + "/v1/mcp", json=body, headers=MCP_HEADERS, timeout=(20, 180))
    response.raise_for_status()
    if response.headers.get("Mcp-Session-Id"):
        MCP_HEADERS["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
    if notification:
        return None
    if response.text.lstrip().startswith("{"):
        messages = [response.json()]
    else:
        events = response.text.replace("\r\n", "\n").split("\n\n")
        data = ["\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")) for event in events]
        messages = [json.loads(item) for item in data if item]
    message = next(item for item in messages if item.get("id") == 1)
    if "error" in message:
        raise RuntimeError(message["error"])
    return message["result"]


def sql(query):
    if "MCP-Protocol-Version" not in MCP_HEADERS:
        initialized = _rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                           "clientInfo": {"name": "deadlock-research", "version": "1"}})
        MCP_HEADERS["MCP-Protocol-Version"] = initialized["protocolVersion"]
        _rpc("notifications/initialized", notification=True)
    return query_rows(_rpc("tools/call", {"name": "execute_query", "arguments": {"sql": query}}))


def find_matches(n=5, match_ids=None, completed=()):
    if match_ids is not None:
        yield from dict.fromkeys(int(mid) for mid in match_ids)
        return
    seen, cursor, attempted = set(completed), None, 0
    try:
        while attempted < max(100, n * 10):
            before = f" AND match_id < {cursor}" if cursor else ""
            rows = sql("SELECT DISTINCT match_id FROM match_salts WHERE replay_salt IS NOT NULL "
                       f"AND metadata_salt IS NOT NULL AND cluster_id IS NOT NULL{before} ORDER BY match_id DESC LIMIT 100")
            if not rows:
                return
            for row in rows:
                mid = int(row["match_id"])
                if mid not in seen:
                    seen.add(mid)
                    attempted += 1
                    yield mid
                    if attempted >= max(100, n * 10):
                        return
            cursor = min(row["match_id"] for row in rows)
    except (requests.RequestException, RuntimeError, ValueError) as error:
        print(f"Lake selection unavailable: {error_message(error)} Checking recent matches.")
        recent, _ = fetch_json("/v1/matches/recently-fetched")
        for row in recent:
            if row["match_id"] not in seen:
                yield int(row["match_id"])
