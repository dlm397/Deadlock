import json
import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agent.io import checksum, now, read_json, save_json


TABLES = ("match_player", "match_salts")
LOG = logging.getLogger(__name__)


def _base_type(source_type):
    value = source_type.strip()
    while value.startswith(("Nullable(", "LowCardinality(")) and value.endswith(")"):
        value = value[value.index("(") + 1:-1].strip()
    return value.upper()


def _integer_type(source_type):
    aliases = {"TINYINT": "INT8", "SMALLINT": "INT16", "INTEGER": "INT32", "INT": "INT32",
               "BIGINT": "INT64", "UTINYINT": "UINT8", "USMALLINT": "UINT16",
               "UINTEGER": "UINT32", "UBIGINT": "UINT64"}
    value = aliases.get(source_type, source_type)
    match = re.fullmatch(r"(U?INT)(8|16|32|64)", value)
    return getattr(pa, match[1].lower() + match[2])() if match else None


def _array(values, source_type):
    source_type = _base_type(source_type)
    present = [value for value in values if value is not None]
    nested = (source_type.startswith(("ARRAY(", "MAP(", "STRUCT(", "TUPLE(", "NESTED(", "LIST(", "UNION("))
              or source_type.endswith("[]") or any(isinstance(value, (dict, list, tuple)) for value in present))
    if nested:
        return pa.array([None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                         for value in values], type=pa.string()), "json"
    integer_type = _integer_type(source_type)
    if integer_type is not None:
        integers = []
        for value in values:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, str))):
                raise ValueError(f"Integer column contains a non-integer value: {value!r}")
            integers.append(None if value is None else int(value))
        return pa.array(integers, type=integer_type), "integer"
    if not present and source_type in ("BOOLEAN", "BOOL"):
        return pa.array(values, type=pa.bool_()), "boolean"
    if not present and source_type in ("FLOAT", "DOUBLE", "REAL", "FLOAT32", "FLOAT64"):
        return pa.array(values, type=pa.float64()), "float"
    if present and all(isinstance(value, bool) for value in present):
        return pa.array(values, type=pa.bool_()), "boolean"
    if present and all(isinstance(value, int) and not isinstance(value, bool) for value in present):
        if min(present) >= 0 and max(present) > 2**63 - 1 and max(present) <= 2**64 - 1:
            return pa.array(values, type=pa.uint64()), "integer"
        if min(present) >= -(2**63) and max(present) <= 2**63 - 1:
            return pa.array(values, type=pa.int64()), "integer"
    if present and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
        if all(not isinstance(value, int) or abs(value) <= 2**53 for value in present):
            return pa.array(values, type=pa.float64()), "float"
    if all(isinstance(value, str) for value in present):
        return pa.array(values, type=pa.string()), "text"
    # JSON also preserves unusually large integers and heterogeneous scalar values.
    return pa.array([None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                     for value in values], type=pa.string()), "json"


def _table(rows, schema):
    fields = {entry.get("name", entry.get("column_name")): entry.get("type", entry.get("column_type", ""))
              for entry in schema}
    if None in fields:
        raise ValueError("Lake schema columns require name/type or column_name/column_type")
    names = list(fields)
    for row in rows:
        for name in row:
            if name not in fields:
                names.append(name)
                fields[name] = ""
    arrays, encodings = [], {}
    for name in names:
        array, encoding = _array([row.get(name) for row in rows], fields[name])
        arrays.append(array)
        encodings[name] = encoding
    return pa.Table.from_arrays(arrays, names=names), encodings


def _remove_temporary(path, root):
    if path.resolve().parent != root.resolve() or not path.name.startswith("."):
        raise ValueError(f"Refusing to remove a directory outside the lake staging area: {path}")
    if path.exists():
        shutil.rmtree(path)


def save_lake_match(root, match_id, tables, provenance, compression_level=19):
    if isinstance(match_id, bool) or int(match_id) <= 0 or str(int(match_id)) != str(match_id):
        raise ValueError("match_id must be a positive integer")
    if set(tables) != set(TABLES):
        raise ValueError(f"A complete lake match requires exactly these tables: {', '.join(TABLES)}")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    match_id = int(match_id)
    target = root / str(match_id)
    staging = Path(tempfile.mkdtemp(prefix=f".{match_id}-", dir=root))
    backup = root / f".{match_id}-backup-{uuid.uuid4().hex}"
    schemas = provenance.get("schemas", provenance.get("schema", {}))
    status = {"success": True, "match_id": match_id, "downloaded_at": provenance.get("downloaded_at", now()),
              "files": {}, "row_counts": {}, "schemas": schemas, "column_encodings": {}, "source": provenance}
    committed = False
    try:
        for name in TABLES:
            rows = list(tables[name])
            table, encodings = _table(rows, schemas.get(name, []))
            filename = f"{name}.parquet"
            pq.write_table(table, staging / filename, compression="zstd", compression_level=compression_level)
            status["files"][filename] = checksum(staging / filename)
            status["row_counts"][name] = len(rows)
            status["column_encodings"][name] = encodings
        save_json(staging / "status.json.gz", status)
        try:
            if target.exists():
                target.replace(backup)
            staging.replace(target)
        except BaseException:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        committed = True
    finally:
        _remove_temporary(staging, root)
        if committed and backup.exists():
            try:
                _remove_temporary(backup, root)
            except OSError:
                LOG.warning("Lake match %s saved; old backup remains at %s", match_id, backup)
    return status


def complete_lake_match(folder):
    folder = Path(folder)
    try:
        status = read_json(folder / "status.json", {})
        return (isinstance(status, dict) and isinstance(status.get("files"), dict)
                and status.get("success") is True and str(status.get("match_id")) == folder.name
                and all(status.get("files", {}).get(f"{name}.parquet") == checksum(folder / f"{name}.parquet")
                        for name in TABLES))
    except (OSError, ValueError, TypeError, EOFError):
        return False
