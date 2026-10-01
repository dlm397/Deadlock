import base64
import binascii
import hashlib
import json
import math
import re
import time


TABLES = {"match_player", "match_salts"}
PAGE_ROWS = 64
CHUNK_CHARACTERS = 40000


def table_name(table):
    if table not in TABLES:
        raise ValueError(f"Unsupported match lake table: {table}")
    return f'"{table}"'


def match_id(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**64:
        raise ValueError("match_id must be a positive UInt64 integer")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key in lake response: {key}")
        result[key] = value
    return result


class LakeSource:
    # Each request fits the public SQL interface's 50 KB and 1,024-row limits.
    def __init__(self, api, request_delay=0.0):
        if not isinstance(request_delay, (int, float)) or not math.isfinite(request_delay) or request_delay < 0:
            raise ValueError("request_delay must be a finite nonnegative number")
        self.api = api
        self.schemas = {}
        self.request_delay = request_delay
        self._last_request = -math.inf

    def query(self, sql):
        pause = self.request_delay - (time.monotonic() - self._last_request)
        if pause > 0:
            time.sleep(pause)
        self._last_request = time.monotonic()
        return self.api.lake_query(sql)

    def describe(self, table):
        name = table_name(table)
        if table not in self.schemas:
            columns = []
            while True:
                page = self.query(f"SELECT column_name, column_type FROM (DESCRIBE {name}) "
                                  f"LIMIT {PAGE_ROWS} OFFSET {len(columns)}")
                columns.extend(page)
                if len(page) < PAGE_ROWS:
                    break
            names = [column["column_name"] for column in columns]
            if not columns or "match_id" not in names or len(names) != len(set(names)):
                raise ValueError(f"Invalid or incomplete lake schema for {table}")
            self.schemas[table] = columns
        return self.schemas[table]

    def _rows_sql(self, table, mid):
        return (f"SELECT to_json(source_row) AS payload FROM {table_name(table)} AS source_row "
                f"WHERE match_id = {match_id(mid)}")

    def _inventory(self, rows_sql):
        inventory = []
        cursor = ""
        while True:
            page = self.query(
                f"WITH source_rows AS ({rows_sql}) "
                "SELECT sha256(payload) AS digest, strlen(payload) AS bytes, count(*) AS copies "
                f"FROM source_rows GROUP BY payload HAVING sha256(payload) > '{cursor}' "
                f"ORDER BY digest LIMIT {PAGE_ROWS}")
            for item in page:
                digest = item.get("digest", "")
                if (not re.fullmatch("[0-9a-f]{64}", digest) or digest <= cursor
                        or not isinstance(item.get("bytes"), int) or item["bytes"] <= 0
                        or not isinstance(item.get("copies"), int) or item["copies"] <= 0):
                    raise ValueError("Invalid or unordered lake row inventory")
                cursor = digest
                inventory.append(item)
            if len(page) < PAGE_ROWS:
                return inventory

    def _row(self, rows_sql, item):
        digest, size = item["digest"], item["bytes"]
        data = bytearray()
        encoded_size = 4 * ((size + 2) // 3)
        for start in range(0, encoded_size, CHUNK_CHARACTERS):
            length = min(CHUNK_CHARACTERS, encoded_size - start)
            page = self.query(
                f"WITH source_rows AS ({rows_sql}) "
                f"SELECT substring(to_base64(encode(payload)), {start + 1}, {length}) AS fragment "
                f"FROM source_rows WHERE sha256(payload) = '{digest}' LIMIT 1")
            if len(page) != 1 or not isinstance(page[0].get("fragment"), str):
                raise ValueError("Lake snapshot changed while reading a match")
            fragment = page[0]["fragment"]
            if len(fragment) != length:
                raise ValueError("Incomplete or invalid lake row fragment")
            try:
                data.extend(base64.b64decode(fragment, validate=True))
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Invalid lake row base64 fragment") from exc
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Lake row checksum mismatch")
        row = json.loads(data.decode("utf-8"), object_pairs_hook=_object)
        if not isinstance(row, dict):
            raise ValueError("Lake row is not a JSON object")
        return row

    def fetch_match(self, table, mid):
        expected_columns = {column["column_name"] for column in self.describe(table)}
        rows_sql = self._rows_sql(table, mid)
        inventory = self._inventory(rows_sql)
        rows = []
        for item in inventory:
            row = self._row(rows_sql, item)
            if set(row) != expected_columns:
                raise ValueError(f"Lake schema changed or fields are missing in {table}")
            if row.get("match_id") != mid:
                raise ValueError("Lake response contains a different match")
            # Exact duplicate multiplicity and distinct source versions are retained.
            rows.extend(dict(row) for _ in range(item["copies"]))
        if self._inventory(rows_sql) != inventory:
            raise ValueError("Lake snapshot changed while reading a match; rerun this match")
        return rows
