import base64
import hashlib
import json
import re
import unittest
from collections import Counter
from unittest.mock import patch

from agent.lake_source import LakeSource, match_id, table_name


class FakeAPI:
    def __init__(self, rows):
        self.rows = rows
        self.sql = []
        self.change_after_read = False
        self.inventory_calls = 0
        self.bad_fragment = None

    def _retry(self, operation, description):
        return operation()

    def lake_query(self, sql):
        self.sql.append(sql)
        if "DESCRIBE" in sql:
            columns = [{"column_name": key, "column_type": "VARCHAR"} for key in self.rows[0]]
            limit, offset = map(int, re.search(r"LIMIT (\d+) OFFSET (\d+)", sql).groups())
            return columns[offset:offset + limit]
        payloads = [json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() for row in self.rows]
        data = {hashlib.sha256(payload).hexdigest(): payload for payload in payloads}
        if "GROUP BY payload" in sql:
            self.inventory_calls += 1
            counts = Counter(hashlib.sha256(payload).hexdigest() for payload in payloads)
            cursor = re.search(r"HAVING sha256\(payload\) > '([^']*)'", sql).group(1)
            result = [{"digest": digest, "bytes": len(data[digest]), "copies": counts[digest]}
                      for digest in sorted(data) if digest > cursor][:64]
            if self.change_after_read and self.inventory_calls > 1:
                result[0]["copies"] += 1
            return result
        digest = re.search(r"sha256\(payload\) = '([^']+)'", sql).group(1)
        start, length = map(int, re.search(r"encode\(payload\)\), (\d+), (\d+)\)", sql).groups())
        fragment = base64.b64encode(data[digest]).decode()[start - 1:start - 1 + length]
        if self.bad_fragment is not None:
            fragment = self.bad_fragment(fragment)
        return [{"fragment": fragment}]


class LakeSourceTests(unittest.TestCase):
    def test_preserves_all_fields_large_integers_nested_values_duplicates_and_versions(self):
        row = {"match_id": 2**63 + 41, "slot": 1, "created_at": "2026-09-28 17:45:53+00",
               "nested.field": [{"9223372036854775807": [None, 1.25, 2**64 - 1]}],
               "all_null": None, "text": "\u2603\U0001f31f" * 15000}
        version = {**row, "created_at": "2026-09-28 18:45:53+00", "text": "updated"}
        api = FakeAPI([row, row, version])
        result = LakeSource(api).fetch_match("match_player", row["match_id"])
        self.assertCountEqual(result, [row, row, version])
        self.assertGreater(sum("AS fragment" in query for query in api.sql), 2)
        for query in api.sql:
            if "AS fragment" in query:
                self.assertLessEqual(int(re.search(r"encode\(payload\)\), \d+, (\d+)\)", query).group(1)), 40000)

    def test_inventory_and_schema_pagination_do_not_lose_rows_or_columns(self):
        rows = [{"match_id": 1, "slot": slot, **{f"field_{i}": i for i in range(150)}} for slot in range(130)]
        result = LakeSource(FakeAPI(rows)).fetch_match("match_player", 1)
        self.assertCountEqual(result, rows)

    def test_incomplete_fragment_rejected(self):
        api = FakeAPI([{"match_id": 1}])
        api.bad_fragment = lambda value: value[:-4]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            LakeSource(api).fetch_match("match_player", 1)

    def test_changed_fragment_rejected_by_checksum(self):
        api = FakeAPI([{"match_id": 1}])
        api.bad_fragment = lambda value: ("A" if value[0] != "A" else "B") + value[1:]
        with self.assertRaisesRegex(ValueError, "checksum"):
            LakeSource(api).fetch_match("match_player", 1)

    def test_invalid_base64_rejected(self):
        api = FakeAPI([{"match_id": 1}])
        api.bad_fragment = lambda value: "!" + value[1:]
        with self.assertRaisesRegex(ValueError, "base64"):
            LakeSource(api).fetch_match("match_player", 1)

    def test_snapshot_change_rejected(self):
        api = FakeAPI([{"match_id": 1}])
        api.change_after_read = True
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            LakeSource(api).fetch_match("match_player", 1)

    def test_missing_field_rejected(self):
        api = FakeAPI([{"match_id": 1, "new_column": None}])
        source = LakeSource(api)
        source.schemas["match_player"] = [{"column_name": "match_id", "column_type": "UBIGINT"}]
        with self.assertRaisesRegex(ValueError, "schema changed"):
            source.fetch_match("match_player", 1)

    def test_wrong_match_rejected(self):
        with self.assertRaisesRegex(ValueError, "different match"):
            LakeSource(FakeAPI([{"match_id": 2}])).fetch_match("match_player", 1)

    def test_duplicate_json_keys_rejected(self):
        source = LakeSource(FakeAPI([{"match_id": 1}]))
        data = b'{"match_id":1,"nested":{"a":2,"a":3}}'
        item = {"digest": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        with patch.object(source, "query", return_value=[{"fragment": base64.b64encode(data).decode()}]):
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                source._row("unused", item)

    def test_request_delay_and_validation(self):
        with patch("agent.lake_source.time.monotonic", side_effect=[0, 0, 0.25, 1]), \
                patch("agent.lake_source.time.sleep") as sleep:
            source = LakeSource(FakeAPI([{"match_id": 1}]), request_delay=1)
            source.query("SELECT column_name, column_type FROM (DESCRIBE match_player) LIMIT 64 OFFSET 0")
            source.query("SELECT column_name, column_type FROM (DESCRIBE match_player) LIMIT 64 OFFSET 0")
        sleep.assert_called_once_with(0.75)
        for value in [-1, float("inf"), "1"]:
            with self.assertRaises(ValueError):
                LakeSource(None, request_delay=value)
        for value in [True, 0, -1, 2**64, "1"]:
            with self.assertRaises(ValueError):
                match_id(value)
        with self.assertRaises(ValueError):
            table_name("match_player; DELETE")


if __name__ == "__main__":
    unittest.main()
