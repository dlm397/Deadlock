import copy
import json
import logging
import re
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow.parquet as pq
import requests

from agent import lake
from agent.io import read_json
from agent.lake_storage import complete_lake_match


def players(mid, count=12, **changes):
    result = []
    for slot in range(count):
        row = {"match_id": mid, "player_slot": slot, "account_id": 1000 + slot,
               "team": "Team0" if slot < 6 else "Team1", "hero_id": slot + 1,
               "game_mode": "Normal", "match_mode": "Ranked", "duration_s": 1200,
               "start_time": "2026-09-01T12:00:00Z", "winning_team": "Team0",
               "net_worth": 50000 + slot, "hero_build_id": 2**64 - 1,
               "future_research_field": None,
               "stats.time_stamp_s": [180, 360], "stats.tech_power": [None, 87.5],
               "nested_values": [{"values": [0, None, 2**64 - 1], "label": "research"}]}
        row.update(changes)
        result.append(row)
    return result


def salts(mid):
    return [{"match_id": mid, "cluster_id": 1, "metadata_salt": 2345,
             "replay_salt": None, "created_at": "2026-09-01T13:00:00Z"}]


def schema(rows):
    integer_columns = {"match_id": "UBIGINT", "player_slot": "UINTEGER",
                       "account_id": "UINTEGER", "duration_s": "UINTEGER",
                       "hero_build_id": "UBIGINT", "metadata_salt": "UINTEGER",
                       "replay_salt": "UINTEGER"}
    return [{"column_name": name, "column_type": integer_columns.get(name, "VARCHAR")}
            for name in rows[0]]


class LakePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "lake"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.manifest = {"version": 197, "format_version": 1, "tables": {}}
        self.api = SimpleNamespace(root=self.root, requests=[], session=Mock(),
                                   json=Mock(return_value=(self.manifest, {"fixture": True})))
        self.api_constructor = self.stack.enter_context(patch("agent.lake.API", return_value=self.api))
        self.source = SimpleNamespace(query=Mock(side_effect=self.query),
                                      describe=Mock(side_effect=self.describe),
                                      fetch_match=Mock(side_effect=self.fetch))
        self.source_constructor = self.stack.enter_context(patch("agent.lake.LakeSource", return_value=self.source))
        self.stack.enter_context(patch("agent.lake.logging.basicConfig"))
        self.stack.enter_context(patch("agent.lake.logging.FileHandler", return_value=logging.NullHandler()))
        self.stack.enter_context(patch("agent.lake.LOG"))
        self.candidates = []
        self.rows = {}
        self.failures = {}

    def query(self, sql):
        candidates = sorted(self.candidates, reverse=True)
        low = re.search(r"match_id\s*>=\s*(\d+)", sql)
        high = re.search(r"match_id\s*<=\s*(\d+)", sql)
        if low:
            candidates = [mid for mid in candidates if mid >= int(low[1])]
        if high:
            candidates = [mid for mid in candidates if mid <= int(high[1])]
        cursor = re.search(r"match_id\s*<\s*(\d+)", sql)
        if cursor:
            candidates = [mid for mid in candidates if mid < int(cursor[1])]
        limit = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
        if limit:
            candidates = candidates[:int(limit[1])]
        return [{"match_id": mid} for mid in candidates]

    def describe(self, table):
        return schema(players(1) if table == "match_player" else salts(1))

    def fetch(self, table, mid):
        if (table, mid) in self.failures:
            raise self.failures[table, mid]
        return copy.deepcopy(self.rows.get((table, mid), players(mid) if table == "match_player" else salts(mid)))

    def collect(self, **settings):
        options = {"output_dir": self.root, "request_delay": 0.0}
        options.update(settings)
        return lake.run(**options)

    def snapshot(self, mid):
        return {path.name: path.read_bytes() for path in (self.root / str(mid)).iterdir() if path.is_file()}

    def table_rows(self, mid, table):
        status = read_json(self.root / str(mid) / "status.json")
        encodings = status["column_encodings"][table]
        rows = pq.read_table(self.root / str(mid) / f"{table}.parquet").to_pylist()
        for row in rows:
            for name, encoding in encodings.items():
                if encoding == "json" and row[name] is not None:
                    row[name] = json.loads(row[name])
        return rows

    def test_target_counts_new_successes_after_existing_failed_and_incomplete_matches(self):
        self.collect(match_ids=[110])
        self.source.fetch_match.reset_mock()
        self.candidates = [110, 109, 108, 107, 106, 105]
        self.failures["match_player", 109] = requests.HTTPError("fixture 503")
        self.rows["match_player", 108] = players(108, count=11)
        result = self.collect(n_matches=2)
        self.assertEqual(result, {"saved": 2, "skipped": 2, "failed": 1})
        fetched = {call.args[1] for call in self.source.fetch_match.call_args_list}
        self.assertNotIn(110, fetched)
        self.assertNotIn(105, fetched)
        self.assertTrue(complete_lake_match(self.root / "107"))
        self.assertTrue(complete_lake_match(self.root / "106"))
        self.assertFalse(complete_lake_match(self.root / "108"))
        self.api.session.close.assert_called()

    def test_explicit_ids_bypass_automatic_filters_and_preserve_small_roster(self):
        self.rows["match_player", 101] = players(101, count=8, game_mode="StreetBrawl", duration_s=400)
        result = self.collect(match_ids=[101], game_mode="Normal", min_duration_s=900,
                              start_date="2026-09-02", end_date="2026-09-03", expected_players=12)
        self.assertEqual(result, {"saved": 1, "skipped": 0, "failed": 0})
        self.source.query.assert_not_called()
        self.assertEqual(len(self.table_rows(101, "match_player")), 8)

    def test_empty_explicit_match_is_skipped_and_next_match_saves(self):
        self.rows["match_player", 101] = []
        result = self.collect(match_ids=[101, 102], n_matches=2)
        self.assertEqual(result, {"saved": 1, "skipped": 1, "failed": 0})
        self.assertFalse(complete_lake_match(self.root / "101"))
        self.assertTrue(complete_lake_match(self.root / "102"))

    def test_all_source_columns_nested_values_duplicate_rows_and_variants_survive(self):
        player_rows = players(101)
        player_rows.extend([copy.deepcopy(player_rows[0]), {**copy.deepcopy(player_rows[0]), "net_worth": 99999}])
        salt_rows = salts(101) * 2
        self.rows["match_player", 101] = player_rows
        self.rows["match_salts", 101] = salt_rows
        self.collect(match_ids=[101])
        self.assertEqual(self.table_rows(101, "match_player"), player_rows)
        self.assertEqual(self.table_rows(101, "match_salts"), salt_rows)
        status = read_json(self.root / "101/status.json")
        self.assertEqual(status["schemas"]["match_player"], self.describe("match_player"))
        self.assertEqual(status["row_counts"], {"match_player": 14, "match_salts": 2})

    def test_resume_preserves_success_and_does_not_fetch_it_again(self):
        self.collect(match_ids=[101])
        before = self.snapshot(101)
        self.source.fetch_match.reset_mock()
        result = self.collect(match_ids=[101])
        self.assertEqual(result, {"saved": 0, "skipped": 1, "failed": 0})
        self.source.fetch_match.assert_not_called()
        self.assertEqual(self.snapshot(101), before)

    def test_failed_reprocess_keeps_previous_successful_match_bytes(self):
        self.collect(match_ids=[101])
        before = self.snapshot(101)
        self.failures["match_salts", 101] = requests.HTTPError("fixture salts failure")
        result = self.collect(match_ids=[101], reprocess=True)
        self.assertEqual(result, {"saved": 0, "skipped": 0, "failed": 1})
        self.assertTrue(complete_lake_match(self.root / "101"))
        after = self.snapshot(101)
        for filename, contents in before.items():
            self.assertEqual(after[filename], contents)

    def test_salts_failure_does_not_publish_partial_player_only_match(self):
        self.failures["match_salts", 101] = requests.ConnectionError("fixture interrupted salts")
        result = self.collect(match_ids=[101, 102], n_matches=2)
        self.assertEqual(result, {"saved": 1, "skipped": 0, "failed": 1})
        self.assertFalse(complete_lake_match(self.root / "101"))
        self.assertTrue(complete_lake_match(self.root / "102"))

    def test_absent_replay_salts_does_not_prevent_lake_collection(self):
        self.rows["match_salts", 101] = []
        result = self.collect(match_ids=[101])
        self.assertEqual(result, {"saved": 1, "skipped": 0, "failed": 0})
        self.assertEqual(self.table_rows(101, "match_salts"), [])
        self.assertTrue(complete_lake_match(self.root / "101"))

    def test_automatic_selection_checks_all_rows_not_only_first_row(self):
        mixed = players(103)
        mixed[-1]["duration_s"] = 400
        self.rows["match_player", 103] = mixed
        self.candidates = [103, 102]
        result = self.collect(n_matches=1)
        self.assertEqual(result, {"saved": 1, "skipped": 1, "failed": 0})
        self.assertTrue(complete_lake_match(self.root / "102"))

    def test_player_duplicates_do_not_satisfy_complete_roster(self):
        self.rows["match_player", 103] = players(103, count=6) * 2
        self.candidates = [103, 102]
        result = self.collect(n_matches=1)
        self.assertEqual(result, {"saved": 1, "skipped": 1, "failed": 0})
        self.assertFalse(complete_lake_match(self.root / "103"))

    def test_candidate_pagination_uses_distinct_match_ids_in_descending_order(self):
        self.candidates = [103, 102, 101]
        result = self.collect(n_matches=3, page_size=2)
        self.assertEqual(result, {"saved": 3, "skipped": 0, "failed": 0})
        self.assertEqual(self.source.query.call_count, 2)
        player_fetches = [call.args[1] for call in self.source.fetch_match.call_args_list
                          if call.args[0] == "match_player"]
        self.assertEqual(player_fetches, [103, 102, 101])

    def test_pagination_finishes_current_partition_before_advancing_older(self):
        self.manifest["tables"]["match_player"] = {
            "partition_expr": "intDiv(match_id, 100)", "files": [{"partition": "3"}]}
        self.candidates = [355, 354, 353, 299, 201, 155, 99]
        result = self.collect(n_matches=6, page_size=2, min_match_id=150)
        self.assertEqual(result, {"saved": 6, "skipped": 0, "failed": 0})
        player_fetches = [call.args[1] for call in self.source.fetch_match.call_args_list
                          if call.args[0] == "match_player"]
        self.assertEqual(player_fetches, [355, 354, 353, 299, 201, 155])
        sql = [call.args[0] for call in self.source.query.call_args_list]
        self.assertEqual(len(sql), 5)
        self.assertIn("match_id >= 300", sql[0])
        self.assertIn("match_id < 354", sql[1])
        self.assertIn("match_id >= 200 AND match_id <= 299", sql[2])
        self.assertIn("match_id < 201", sql[3])
        self.assertIn("match_id >= 150 AND match_id <= 199", sql[4])
        self.assertNotRegex(sql[4], r"match_id\s*<\s*\d")

    def test_discovery_searches_empty_and_unlisted_partitions_without_losing_older_matches(self):
        self.manifest["tables"]["match_player"] = {
            "partition_expr": "intDiv(match_id, 100)", "files": [{"partition": "3"}]}
        self.candidates = [350, 150]
        result = self.collect(n_matches=2, page_size=2, min_match_id=100)
        self.assertEqual(result, {"saved": 2, "skipped": 0, "failed": 0})
        self.assertEqual(self.source.query.call_count, 3)
        self.assertTrue(complete_lake_match(self.root / "150"))

    def test_dates_are_start_inclusive_and_end_exclusive(self):
        self.candidates = [105, 104, 103]
        self.rows["match_player", 105] = players(105, start_time="2026-08-31T23:59:59Z")
        self.rows["match_player", 104] = players(104, start_time="2026-09-02T00:00:00Z")
        self.rows["match_player", 103] = players(103, start_time="2026-09-01T00:00:00Z")
        result = self.collect(n_matches=1, start_date="2026-09-01", end_date="2026-09-02")
        self.assertEqual(result, {"saved": 1, "skipped": 2, "failed": 0})
        self.assertTrue(complete_lake_match(self.root / "103"))
        sql = self.source.query.call_args.args[0]
        self.assertIn("start_time >= TIMESTAMPTZ '2026-09-01T00:00:00+00:00'", sql)
        self.assertIn("start_time < TIMESTAMPTZ '2026-09-02T00:00:00+00:00'", sql)

    def test_discovery_applies_modes_id_range_and_utc_offsets(self):
        self.candidates = [105]
        self.rows["match_player", 105] = players(105, match_mode="Private'Lobby")
        self.collect(n_matches=1, match_mode="Private'Lobby", min_match_id=100, max_match_id=110,
                     start_date="2026-09-01T08:00:00-04:00", end_date="2026-09-01T08:01:00-04:00")
        sql = self.source.query.call_args.args[0]
        self.assertIn("CAST(game_mode AS VARCHAR) = 'Normal'", sql)
        self.assertIn("CAST(match_mode AS VARCHAR) = 'Private''Lobby'", sql)
        self.assertIn("match_id >= 100", sql)
        self.assertIn("match_id <= 110", sql)
        self.assertIn("2026-09-01T12:00:00+00:00", sql)
        self.assertIn("2026-09-01T12:01:00+00:00", sql)
        self.assertTrue(complete_lake_match(self.root / "105"))

    def test_candidate_limit_stops_a_run_with_persistent_source_failures(self):
        self.candidates = [103, 102, 101]
        for mid in self.candidates:
            self.failures["match_player", mid] = requests.ConnectionError("fixture lake unavailable")
        result = self.collect(n_matches=1, max_candidates=2)
        self.assertEqual(result, {"saved": 0, "skipped": 0, "failed": 2})
        self.assertEqual([call.args[1] for call in self.source.fetch_match.call_args_list], [103, 102])
        last_run = read_json(self.root / "last_run.json")
        self.assertEqual(last_run["attempted"], 2)
        self.assertEqual(last_run["settings"]["max_candidates"], 2)

    def test_invalid_settings_fail_before_creating_files_or_network_clients(self):
        invalid = [
            {"n_matches": 0}, {"n_matches": True}, {"n_matches": 1.5},
            {"page_size": 0}, {"page_size": 257}, {"page_size": True},
            {"min_duration_s": -1}, {"expected_players": 0}, {"max_candidates": 0},
            {"compression_level": 0}, {"compression_level": 23},
            {"request_delay": -0.1}, {"request_delay": float("inf")},
            {"request_delay": float("nan")}, {"request_delay": True},
            {"game_mode": 1}, {"match_mode": ""},
            {"start_date": "invalid-date"}, {"end_date": "2026-02-30"},
            {"start_date": "2026-09-02", "end_date": "2026-09-01"},
            {"start_date": "2026-09-01", "end_date": "2026-09-01"},
            {"min_match_id": 0}, {"max_match_id": 2**64},
            {"min_match_id": 110, "max_match_id": 100},
            {"match_ids": [True]}, {"match_ids": [1.5]},
            {"match_ids": ["1.5"]}, {"match_ids": ["1; DROP TABLE match_player"]},
        ]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.collect(**settings)
            self.api_constructor.assert_not_called()
            self.source_constructor.assert_not_called()
            self.assertFalse(self.root.exists())


class DiscoveryWindowTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"tables": {"match_player": {
            "partition_expr": "intDiv(match_id, 100)",
            "files": [{"partition": "1"}, {"rows_by_partition": {"3": 12}}]}}}

    def test_windows_include_newer_tail_and_contiguous_older_unlisted_partitions(self):
        self.assertEqual(list(lake.discovery_windows(self.manifest, None, None)),
                         [(300, None), (200, 299), (100, 199), (1, 99)])

    def test_windows_clip_both_bounds_including_exact_partition_boundaries(self):
        cases = [
            ((125, 275), [(200, 275), (125, 199)]),
            ((125, 450), [(300, 450), (200, 299), (125, 199)]),
            ((100, 300), [(300, 300), (200, 299), (100, 199)]),
            ((250, 275), [(250, 275)]),
            ((1, 1), [(1, 1)]),
        ]
        for (low, high), expected in cases:
            with self.subTest(low=low, high=high):
                self.assertEqual(list(lake.discovery_windows(self.manifest, low, high)), expected)

    def test_minimum_newer_than_manifest_keeps_new_lake_matches_in_scope(self):
        self.assertEqual(list(lake.discovery_windows(self.manifest, 450, None)), [(450, None)])
        self.assertEqual(list(lake.discovery_windows(self.manifest, 450, 475)), [(450, 475)])

    def test_missing_or_unsupported_partition_metadata_uses_requested_range(self):
        manifests = [None, {}, {"tables": {"match_player": {"partition_expr": "toYYYYMM(start_time)",
                                                          "files": [{"partition": "3"}]}}},
                     {"tables": {"match_player": {"partition_expr": "intDiv(match_id, 100)", "files": []}}}]
        for manifest in manifests:
            with self.subTest(manifest=manifest):
                self.assertEqual(list(lake.discovery_windows(manifest, 125, 275)), [(125, 275)])

    def test_discovery_rejects_server_rows_outside_current_window(self):
        filters = dict(game_mode=None, match_mode=None, min_duration_s=0, expected_players=None,
                       start_date=None, end_date=None, min_match_id=125, max_match_id=275)
        for mid in (199, 276):
            with self.subTest(mid=mid):
                source = SimpleNamespace(query=Mock(return_value=[{"match_id": mid}]))
                with self.assertRaisesRegex(ValueError, "outside the requested range"):
                    list(lake.candidate_matches(source, filters, 10, self.manifest))


if __name__ == "__main__":
    unittest.main()
