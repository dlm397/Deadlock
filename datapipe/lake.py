import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .sources import fetch_json, now, read_json, save_json, sql, report_error

TABLES = ("match_player", "match_salts")
MANIFEST_URL = "https://data.deadlock-api.com/v1/manifest.json"


def describe(table):
    columns = []
    while True:
        page = sql(f'SELECT column_name, column_type FROM (DESCRIBE "{table}") LIMIT 64 OFFSET {len(columns)}')
        columns.extend(page)
        if len(page) < 64:
            return columns


def row_inventory(query):
    rows, cursor = [], ""
    while True:
        page = sql(f"WITH source_rows AS ({query}) SELECT sha256(payload) AS digest, strlen(payload) AS bytes, "
                   f"count(*) AS copies FROM source_rows GROUP BY payload HAVING sha256(payload) > '{cursor}' "
                   "ORDER BY digest LIMIT 64")
        rows.extend(page)
        if len(page) < 64:
            return rows
        cursor = page[-1]["digest"]


def fetch_table(table, match_id, schema):
    # Base64 fragments avoid the public SQL endpoint's 50 KB response limit.
    query = f'SELECT to_json(source_row) AS payload FROM "{table}" AS source_row WHERE match_id = {int(match_id)}'
    inventory, rows = row_inventory(query), []
    for item in inventory:
        data = bytearray()
        encoded_size = 4 * ((item["bytes"] + 2) // 3)
        for start in range(0, encoded_size, 40000):
            fragment = sql(f"WITH source_rows AS ({query}) SELECT substring(to_base64(encode(payload)), "
                           f"{start + 1}, {min(40000, encoded_size - start)}) AS fragment FROM source_rows "
                           f"WHERE sha256(payload) = '{item['digest']}' LIMIT 1")
            data.extend(base64.b64decode(fragment[0]["fragment"], validate=True))
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["digest"]:
            raise ValueError("Incomplete lake row or source changed during retrieval")
        row = json.loads(data)
        if row["match_id"] != match_id or set(row) != {field["column_name"] for field in schema}:
            raise ValueError("Lake match identity/schema mismatch")
        rows.extend(dict(row) for _ in range(item["copies"]))
    if row_inventory(query) != inventory:
        raise ValueError("Lake snapshot changed during retrieval; rerun this match")
    return rows, inventory


def arrow_table(rows, schema):
    integer_types = {"TINYINT": pa.int8(), "SMALLINT": pa.int16(), "INTEGER": pa.int32(),
                     "BIGINT": pa.int64(), "UTINYINT": pa.uint8(), "USMALLINT": pa.uint16(),
                     "UINTEGER": pa.uint32(), "UBIGINT": pa.uint64()}
    arrays, names = [], []
    for field in schema:
        name, source_type = field["column_name"], field["column_type"]
        values = [row.get(name) for row in rows]
        present = [value for value in values if value is not None]
        nested = source_type.endswith("[]") or source_type.startswith(("MAP(", "STRUCT("))
        if nested or any(isinstance(value, (dict, list)) for value in present):
            values = [None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":")) for value in values]
            datatype = pa.string()
        elif source_type in integer_types:
            if any(type(value) is not int for value in present):
                raise ValueError(f"Noninteger value in {name}; refusing possible identifier rounding")
            datatype = integer_types[source_type]
        elif source_type == "BOOLEAN":
            datatype = pa.bool_()
        elif source_type in ("FLOAT", "DOUBLE"):
            datatype = pa.float64()
        elif all(isinstance(value, str) for value in present):
            datatype = pa.string()
        else:
            datatype = None
        arrays.append(pa.array(values, type=datatype))
        names.append(name)
    return pa.Table.from_arrays(arrays, names=names)


def save_match(root, match_id, tables, schemas, provenance):
    folder = Path(root) / str(match_id)
    folder.mkdir(parents=True, exist_ok=True)
    status = {"success": True, "match_id": match_id, "downloaded_at": now(), "schemas": schemas,
              "source": provenance, "files": {}, "row_counts": {name: len(rows) for name, rows in tables.items()}}
    for name, rows in tables.items():
        temporary = folder / f"{name}.parquet.tmp"
        pq.write_table(arrow_table(rows, schemas[name]), temporary, compression="zstd", compression_level=19)
        with temporary.open("rb") as file:
            status["files"][f"{name}.parquet"] = hashlib.file_digest(file, "sha256").hexdigest()
    # Completion is written last so interrupted replacements cannot look complete.
    for status_name in ("status.json", "status.json.gz"):
        (folder / status_name).unlink(missing_ok=True)
    for name in tables:
        (folder / f"{name}.parquet.tmp").replace(folder / f"{name}.parquet")
    save_json(folder / "status.json.gz", status)
    return status


def candidate_matches(settings, manifest):
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    conditions = [f"duration_s >= {settings.MIN_DURATION_S}"]
    for name in ("game_mode", "match_mode"):
        value = getattr(settings, name.upper(), None)
        if value is not None:
            conditions.append(f"CAST({name} AS VARCHAR) = {quote(value)}")
    for name, operator in (("START_DATE", ">="), ("END_DATE", "<")):
        value = getattr(settings, name, None)
        if value:
            date = datetime.fromisoformat(value.replace("Z", "+00:00"))
            date = date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date.astimezone(timezone.utc)
            conditions.append(f"start_time {operator} TIMESTAMPTZ {quote(date.isoformat())}")
    low, high = getattr(settings, "MIN_MATCH_ID", None) or 1, getattr(settings, "MAX_MATCH_ID", None)
    table = manifest["tables"]["match_player"]
    stride_match = re.fullmatch(r"intDiv\(\s*match_id\s*,\s*(\d+)\s*\)", table.get("partition_expr", ""))
    partitions = [int(file["partition"]) for file in table["files"] if file.get("partition") is not None]
    partitions += [int(part) for file in table["files"] for part in file.get("rows_by_partition", {})]
    stride = int(stride_match[1]) if stride_match and partitions else None
    begin = low
    if stride:
        newest = max(partitions) * stride
        begin = min(newest, high // stride * stride) if high is not None else newest
        begin = max(low, begin)
    roster = f" HAVING count(DISTINCT player_slot) = {settings.EXPECTED_PLAYERS}" if settings.EXPECTED_PLAYERS else ""
    while True:
        bounds = [f"match_id >= {begin}"] + ([f"match_id <= {high}"] if high is not None else [])
        cursor = None
        while True:
            before = [f"match_id < {cursor}"] if cursor else []
            query = "SELECT match_id FROM match_player WHERE " + " AND ".join(conditions + bounds + before)
            rows = sql(query + f" GROUP BY match_id{roster} ORDER BY match_id DESC LIMIT 100")
            yield from (int(row["match_id"]) for row in rows)
            if len(rows) < 100:
                break
            cursor = min(row["match_id"] for row in rows)
        if begin == low:
            return
        high, begin = begin - 1, max(low, begin - stride)


def eligible(rows, settings):
    slots = {row.get("player_slot") for row in rows}
    if None in slots or settings.EXPECTED_PLAYERS is not None and len(slots) != settings.EXPECTED_PLAYERS:
        return False
    for row in rows:
        if row.get("duration_s", 0) < settings.MIN_DURATION_S:
            return False
        for name in ("GAME_MODE", "MATCH_MODE"):
            value = getattr(settings, name, None)
            if value is not None and row.get(name.lower()) != value:
                return False
        for name, operator in (("START_DATE", "before"), ("END_DATE", "after")):
            value = getattr(settings, name, None)
            if value:
                bound = datetime.fromisoformat(value.replace("Z", "+00:00"))
                bound = bound.replace(tzinfo=timezone.utc) if bound.tzinfo is None else bound
                started = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
                if operator == "before" and started < bound or operator == "after" and started >= bound:
                    return False
    return True


def run(settings):
    root = Path(settings.OUTPUT_DIR) / "lake"
    schemas = {table: describe(table) for table in TABLES}
    save_json(root / "tables/schema.json.gz", schemas)
    explicit = settings.MATCH_IDS is not None
    manifest, provenance = (None, {}) if explicit else fetch_json(MANIFEST_URL)
    if manifest:
        save_json(root / "tables" / f"manifest_{manifest['version']}.json.gz", manifest)
    selected = dict.fromkeys(int(mid) for mid in settings.MATCH_IDS) if explicit else candidate_matches(settings, manifest)
    saved = failed = attempted = skipped = already_saved = 0
    for mid in selected:
        folder = root / str(mid)
        if not settings.REPROCESS and read_json(folder / "status.json", {}).get("success") and all((folder / f"{name}.parquet").exists() for name in TABLES):
            already_saved += 1
            continue
        attempted += 1
        print(f"Processing lake match {mid}...")
        try:
            tables, inventories = {}, {}
            for table in TABLES:
                tables[table], inventories[table] = fetch_table(table, mid, schemas[table])
            if not tables["match_player"]:
                raise ValueError("No player records available")
            if not explicit and not eligible(tables["match_player"], settings):
                skipped += 1
                print("    SKIPPED: source records do not satisfy the configured mode, duration, roster or date filters")
            else:
                save_match(root, mid, tables, schemas, {"url": "https://api.deadlock-api.com/v1/mcp",
                           "manifest_version": manifest.get("version") if manifest else None,
                           "manifest_retrieved_at": provenance.get("retrieved_at"), "row_inventory": inventories})
                saved += 1
                print(f"    saved: {len(tables['match_player'])} player records, {len(schemas['match_player'])} columns")
        except Exception as error:
            failed += 1
            report_error(error, folder / "error.log")
        if not explicit and (saved >= settings.N_MATCHES or attempted >= max(100, settings.N_MATCHES * 10)):
            break
    print(f"Completed: {saved}/{len(settings.MATCH_IDS) if explicit else settings.N_MATCHES} | Skipped: {skipped} | Failed: {failed} | Already saved: {already_saved}")
    return {"saved": saved, "failed": failed, "skipped": skipped, "already_saved": already_saved}
