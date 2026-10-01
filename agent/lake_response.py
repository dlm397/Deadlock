import json
import re

import requests


class LakeUnavailable(requests.ConnectionError):
    pass


def _value(token):
    token = token.strip()
    if not token:
        raise ValueError("Empty TOON value must be quoted")
    if token.startswith('"') or token in ("true", "false", "null", "[]"):
        return json.loads(token)
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", token):
        return json.loads(token)
    return token


def _cells(line, delimiter=","):
    values, start, quoted, escaped = [], 0, False, False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == delimiter and not quoted:
            values.append(_value(line[start:index]))
            start = index + 1
    if quoted or escaped:
        raise ValueError("Unterminated TOON string")
    values.append(_value(line[start:]))
    return values


def _toon_table(text):
    # Decode the flat SQL response envelope; nested match data travels as base64.
    result = {}
    lines = text.splitlines()
    index = 0
    header = re.compile(r"([A-Za-z_]\w*)(?:\[([0-9]+)([,|\t])?\])?(?:\{(.*)\})?:[ \t]?(.*)")
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip():
            continue
        match = header.fullmatch(line)
        if not match:
            raise ValueError("Unsupported or malformed TOON SQL response")
        key, count, delimiter, fields, inline = match.groups()
        if key in result:
            raise ValueError(f"Duplicate TOON response key: {key}")
        if count is None:
            if fields is not None:
                raise ValueError("TOON table has no row count")
            result[key] = _value(inline)
            continue
        count, delimiter = int(count), delimiter or ","
        if fields is None:
            result[key] = _cells(inline, delimiter) if inline else []
        else:
            if inline.strip():
                raise ValueError("Unexpected inline TOON table data")
            names = _cells(fields, delimiter)
            if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
                raise ValueError("Invalid or duplicate TOON table columns")
            rows = []
            while index < len(lines) and lines[index].startswith("  "):
                values = _cells(lines[index][2:], delimiter)
                if len(values) != len(names):
                    raise ValueError("TOON row width does not match its columns")
                rows.append(dict(zip(names, values, strict=True)))
                index += 1
            result[key] = rows
        if len(result[key]) != count:
            raise ValueError(f"Incomplete TOON {key}: expected {count}, received {len(result[key])}")
    return result


def _failure(message):
    if "NotReadyError" in message or "snapshot catalog is still loading" in message.lower():
        raise LakeUnavailable("Data lake snapshot is still loading")
    raise RuntimeError(f"Data lake query failed: {message[:500]}")


def query_rows(result):
    if not isinstance(result, dict):
        raise RuntimeError("Invalid data lake tool result")
    structured = result.get("structuredContent")
    payload = structured.get("result", structured if "success" in structured else None) if isinstance(structured, dict) else None
    if payload is None:
        payload = "\n".join(item["text"] for item in result.get("content", [])
                            if item.get("type") == "text" and isinstance(item.get("text"), str))
    if result.get("isError"):
        _failure(str(payload))
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            raise LakeUnavailable("Data lake returned an empty query response")
        try:
            table = json.loads(payload) if payload.startswith(("{", "[")) else _toon_table(payload)
        except ValueError as exc:
            raise RuntimeError(f"Invalid data lake query response: {exc}") from exc
    else:
        table = payload
    if not isinstance(table, dict):
        raise RuntimeError("Data lake query response must be an object")
    if table.get("success") is not True:
        _failure(f"{table.get('errorType', '')}: {table.get('error', table)}")
    columns, rows = table.get("columns"), table.get("rows")
    if (not isinstance(columns, list) or not isinstance(rows, list)
            or any(not isinstance(name, str) for name in columns) or len(set(columns)) != len(columns)):
        raise RuntimeError("Data lake response has invalid columns or rows")
    if table.get("truncated") or table.get("rowCount", len(rows)) != len(rows):
        raise RuntimeError("Data lake response was truncated; use a smaller query")
    records = []
    for row in rows:
        if isinstance(row, dict) and set(row) == set(columns):
            records.append({name: row[name] for name in columns})
        elif isinstance(row, list) and len(row) == len(columns):
            records.append(dict(zip(columns, row, strict=True)))
        else:
            raise RuntimeError("Data lake response row does not match its columns")
    return records
