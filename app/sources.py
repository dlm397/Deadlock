"""Small HTTP/download helpers; no data from the original project is used."""
import bz2
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
import zstandard

from .console_log import log

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "deadlock2-personal-research/1"
REPLAY_SUFFIXES = (".dem.bz2", ".dem.zst", ".dem")
_DEFINITIONS = {}


class EligibilityConflict(ValueError):
    """The replay and metadata describe conflicting match identity or modes."""


def get_json(config, endpoint, **params):
    response = SESSION.get(config["api_base"].rstrip("/") + endpoint,
                           params=params, timeout=(20, 90))
    response.raise_for_status()
    return response.json()


def metadata(match_id, config):
    result = get_json(config, f"/v1/matches/{match_id}/metadata",
                      disable_steam=str(config["disable_steam"]).lower())
    info = result.get("match_info") if isinstance(result, dict) else None
    if not isinstance(info, dict) or info.get("match_id") != match_id:
        raise ValueError("metadata is missing or belongs to another match")
    return result


def _match_id(value):
    # Never accept a float: uint64 match IDs must survive without rounding.
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if type(value) is not int or not 0 < value < 2**64:
        raise ValueError("match discovery returned an invalid match ID")
    return value


def _discovery_ids(result):
    """Decode the lake's single-column JSON or TOON result, including empty pages."""
    payload = result.get("structuredContent")
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if payload is None:
        payload = "\n".join(block["text"] for block in result.get("content", [])
                            if block.get("type") == "text")
    if result.get("isError"):
        raise ValueError(f"replay discovery query failed: {str(payload)[:500]}")
    if isinstance(payload, str) and payload.lstrip().startswith("{"):
        payload = json.loads(payload)
    if isinstance(payload, str):
        # Only match_id is selected, so a general-purpose TOON parser is unnecessary.
        rows_header = (re.search(r"^rows\[(\d+)[,|\t]?\]\{match_id\}:[ \t]*$", payload, re.M)
                       or re.search(r"^rows\[(0)\]:[ \t]*$", payload, re.M))
        row_count = re.search(r"^rowCount:\s*(\d+)\s*$", payload, re.M)
        if (not re.search(r"^success:\s*true\s*$", payload, re.M)
                or not re.search(r"^columns\[1\]:\s*match_id\s*$", payload, re.M)
                or not rows_header or not row_count
                or re.search(r"^truncated:\s*true\s*$", payload, re.M)):
            raise ValueError(f"replay discovery returned an invalid or truncated query result: {payload[:500]}")
        rows = []
        for line in payload[rows_header.end():].splitlines():
            if not line.strip():
                continue
            if not line.startswith("  "):
                break
            rows.append(json.loads(line.strip()))
        if len(rows) != int(rows_header[1]) or len(rows) != int(row_count[1]):
            raise ValueError("replay discovery returned an incomplete match list")
    elif isinstance(payload, dict):
        if (payload.get("success") is not True or payload.get("truncated")
                or payload.get("columns") != ["match_id"] or not isinstance(payload.get("rows"), list)):
            raise ValueError(f"replay discovery returned an invalid query result: {str(payload)[:500]}")
        rows = payload["rows"]
        if payload.get("rowCount", len(rows)) != len(rows):
            raise ValueError("replay discovery returned an incomplete match list")
        if any(not isinstance(row, (dict, list)) or len(row) != 1
               or isinstance(row, dict) and "match_id" not in row for row in rows):
            raise ValueError("replay discovery returned malformed match rows")
        rows = [row["match_id"] if isinstance(row, dict) else row[0] for row in rows]
    else:
        raise ValueError("replay discovery returned no query result")
    return [_match_id(value) for value in rows]


def discover(config, exclude=()):
    """Discover full-match candidates with recorded replay salts through MCP.

    Explicit match IDs skip discovery. Large discovery limits can exceed the
    service's MCP query limits; this is independent of the dump pipeline.
    """
    headers = {"Accept": "application/json, text/event-stream"}
    endpoint = config["api_base"].rstrip("/") + "/v1/mcp"

    def rpc(method, params=None, notification=False):
        body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notification:
            body["id"] = 1
        response = SESSION.post(endpoint, json=body, headers=headers, timeout=(20, 180))
        response.raise_for_status()
        if response.headers.get("Mcp-Session-Id"):
            headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
        if notification:
            return None
        if response.text.lstrip().startswith("{"):
            messages = [response.json()]
        else:
            events = response.text.replace("\r\n", "\n").split("\n\n")
            data = ["\n".join(line[5:].lstrip() for line in event.splitlines()
                              if line.startswith("data:")) for event in events]
            messages = [json.loads(item) for item in data if item]
        message = next((item for item in messages if isinstance(item, dict) and item.get("id") == 1), None)
        if message is None or "error" in message or not isinstance(message.get("result"), dict):
            raise ValueError(f"replay discovery MCP request failed: {str(message)[:500]}")
        return message["result"]

    initialized = rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                     "clientInfo": {"name": "deadlock2-research", "version": "1"}})
    if not isinstance(initialized.get("protocolVersion"), str):
        raise ValueError("replay discovery MCP initialization returned no protocol version")
    headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
    rpc("notifications/initialized", notification=True)
    selected, seen, cursor = [], {_match_id(mid) for mid in exclude}, None
    limit = config["discovery_limit"]
    page_size = min(100, limit)
    while len(selected) < limit:
        before = f" AND match_id < {cursor}" if cursor is not None else ""
        query = ("SELECT DISTINCT match_id FROM match_salts WHERE match_id > 0 "
                 "AND replay_salt IS NOT NULL AND replay_salt != 0 "
                 "AND metadata_salt IS NOT NULL AND metadata_salt != 0 "
                 f"AND cluster_id IS NOT NULL AND cluster_id != 0{before} "
                 f"ORDER BY match_id DESC LIMIT {page_size}")
        ids = _discovery_ids(rpc("tools/call", {"name": "execute_query", "arguments": {"sql": query}}))
        if not ids:
            break
        if cursor is not None and any(mid >= cursor for mid in ids):
            raise ValueError("replay discovery returned a nonadvancing match page")
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                selected.append(mid)
                if len(selected) == limit:
                    break
        cursor = min(ids)
        if len(ids) < page_size:
            break
    return selected


def find_replay(match_id, output_dir):
    paths = [Path(output_dir) / "replays" / f"{match_id}{suffix}" for suffix in REPLAY_SUFFIXES]
    existing = [p for p in paths if p.is_file()]
    if len(existing) > 1:
        raise ValueError("multiple retained replays for this match; keep the intended one")
    return existing[0] if existing else None


def replay_suffix(magic):
    if magic.startswith(b"BZh"):
        return ".dem.bz2"
    if magic.startswith(b"\x28\xb5\x2f\xfd"):
        return ".dem.zst"
    if magic.startswith(b"PBDE"):
        return ".dem"
    raise ValueError("response is not a supported Source 2 replay")


def download_replay(match_id, config):
    folder = Path(config["output_dir"]) / "replays"
    folder.mkdir(parents=True, exist_ok=True)
    salts = get_json(config, f"/v1/matches/{match_id}/salts",
                     disable_steam=str(config["disable_steam"]).lower())
    if salts.get("match_id") != match_id or not salts.get("demo_url"):
        raise ValueError("replay URL is unavailable for this match")
    temporary = folder / f"{match_id}.part"
    try:
        with SESSION.get(salts["demo_url"], stream=True, timeout=(20, 120),
                         headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            size, suffix = 0, None
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    if suffix is None:
                        suffix = replay_suffix(chunk[:4])
                    stream.write(chunk)
                    size += len(chunk)
            expected = response.headers.get("Content-Length")
            if not size or expected is not None and size != int(expected):
                raise ValueError("incomplete replay download")
        destination = folder / f"{match_id}{suffix}"
        temporary.replace(destination)
        log(f"  retained replay: {size / 1024**2:.1f} MiB", flush=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def definitions(client_version, config):
    """Only exact-version item/ability definitions; never substitute current assets."""
    if client_version is None:
        return []
    key = (config["api_base"], client_version)
    if key not in _DEFINITIONS:
        available = get_json(config, "/v1/assets/client-versions")
        if client_version not in available:
            _DEFINITIONS[key] = []
        else:
            items = get_json(config, "/v1/assets/items", client_version=client_version)
            if not isinstance(items, list):
                raise ValueError("item definitions did not return a list")
            _DEFINITIONS[key] = items
    return _DEFINITIONS[key]


def parse_replay(replay, payload, sampling_interval_s):
    """Parse an existing file. This function never makes an HTTP request."""
    with tempfile.TemporaryDirectory(prefix="deadlock2-parse-") as temporary:
        folder = Path(temporary)
        with replay.open("rb") as stream:
            suffix = replay_suffix(stream.read(4))
        expanded = replay
        if suffix != ".dem":
            expanded = folder / "replay.dem"
            with expanded.open("wb") as output:
                if suffix == ".dem.bz2":
                    with bz2.open(replay, "rb") as source:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                else:
                    with replay.open("rb") as source:
                        zstandard.ZstdDecompressor().copy_stream(source, output)
        roster = folder / "metadata.json"
        # JavaScript numbers cannot represent every uint64 match ID exactly.
        parser_payload = {**payload, "match_info": {**payload["match_info"],
                          "match_id": str(payload["match_info"]["match_id"])}}
        roster.write_text(json.dumps(parser_payload), encoding="utf-8")
        command = ["node", "--max-old-space-size=6144", str(Path(__file__).with_name("replay.mjs")),
                   str(expanded.resolve()), str(roster), str(folder), str(sampling_interval_s)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        if summary.get("eligibility_conflict"):
            raise EligibilityConflict(summary.get("error", "replay and metadata conflict"))
        if result.returncode or not summary.get("success"):
            detail = summary.get("error") or (result.stderr or result.stdout)[-2500:]
            raise RuntimeError(f"replay parser failed: {detail.strip()}")
        def rows(name):
            with (folder / name).open(encoding="utf-8") as stream:
                return [json.loads(line) for line in stream if line.strip()]
        return summary, rows("samples.ndjson"), rows("replay_events.ndjson")
