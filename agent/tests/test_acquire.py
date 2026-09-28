import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent import acquire
from agent.io import read_json, save_json


class FakeAPI:
    def __init__(self, root):
        self.root = root
        self.responses = {}
        self.network_calls = []
        self.json = Mock(side_effect=self.get_json)
        self.lake_query = Mock()
        self.download = Mock()

    def get_json(self, path, filename=None, params=None, body=None, cached=False, **kwargs):
        target = self.root / filename if filename else None
        if cached and target:
            value = read_json(target)
            if value is not None:
                return value, {}
        self.network_calls.append(path)
        if path not in self.responses:
            raise AssertionError(f"Unexpected request: {path}")
        value = self.responses[path]
        if isinstance(value, Exception):
            raise value
        if target:
            save_json(target, value)
        return value, {}


class AcquireTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.api = FakeAPI(self.root)
        self.null_salts = {"match_id": 1, "cluster_id": 7, "replay_salt": None, "demo_url": None}
        self.available_salts = {**self.null_salts, "replay_salt": 123,
                                "demo_url": "http://replay7.valve.net/1422450/1_123.dem.bz2"}

    def test_selection_uses_lake_and_skips_completed_matches(self):
        self.api.lake_query.side_effect = [[{"match_id": 90}, {"match_id": 89}, {"match_id": 88}], []]
        self.api.responses["/v1/matches/recently-fetched"] = []
        selected = list(acquire.select_matches(self.api, 5, None, lambda mid: mid == 89))
        self.assertEqual(selected, [90, 88])
        self.api.lake_query.assert_called()
        self.assertNotIn("/v1/sql", self.api.network_calls)
        query = self.api.lake_query.call_args_list[0].args[0]
        self.assertIn("match_salts", query)
        self.assertRegex(query.lower(), r"replay_salt\s+is\s+not\s+null")

    def test_selection_pages_past_completed_matches_with_bounded_queries(self):
        queries = []
        first_minimum = None
        self.api.responses["/v1/matches/recently-fetched"] = []

        def page(query):
            nonlocal first_minimum
            queries.append(query)
            limit = int(re.search(r"\bLIMIT\s+(\d+)", query, re.IGNORECASE).group(1))
            self.assertGreater(limit, 0)
            self.assertLessEqual(limit, 1024)
            if len(queries) == 1:
                ids = list(range(10000, 10000 - limit, -1))
                first_minimum = min(ids)
                return [{"match_id": mid} for mid in ids]
            if len(queries) == 2:
                self.assertRegex(query, rf"match_id\s*<\s*{first_minimum}\b")
                return [{"match_id": 7000}, {"match_id": 6999}]
            return []

        self.api.lake_query.side_effect = page
        selected = list(acquire.select_matches(self.api, 50, None, lambda mid: mid >= 8000))
        self.assertEqual(selected, [7000, 6999])
        self.assertGreaterEqual(len(queries), 2)
        self.assertNotIn("/v1/sql", self.api.network_calls)

    def test_explicit_ids_do_not_use_discovery(self):
        selected = list(acquire.select_matches(self.api, 50, [1, "2", 1]))
        self.assertEqual(selected, [1, 2])
        self.api.lake_query.assert_not_called()
        self.api.json.assert_not_called()

    def test_completed_matches_do_not_consume_candidate_budget(self):
        pages = [[{"match_id": mid} for mid in range(top, top - 30, -1)] for top in (200, 170, 140, 110)]
        self.api.lake_query.side_effect = [*pages, [{"match_id": 80}]]
        self.api.responses["/v1/matches/recently-fetched"] = []
        selected = list(acquire.select_matches(self.api, 1, None, lambda mid: mid > 80))
        self.assertEqual(selected, [80])
        self.assertEqual(self.api.lake_query.call_count, 5)

    def test_recent_fallback_yields_only_verified_replay_candidates(self):
        self.api.lake_query.side_effect = RuntimeError("Data lake unavailable")
        self.api.responses["/v1/matches/recently-fetched"] = [
            {"match_id": 1, "start_time": 4, "game_mode": 1, "duration_s": 1200},
            {"match_id": 2, "start_time": 3, "game_mode": 1, "duration_s": 1200},
            {"match_id": 3, "start_time": 2, "game_mode": 1, "duration_s": 1200},
            {"match_id": 4, "start_time": 1, "game_mode": 1, "duration_s": 30},
        ]
        self.api.responses["/v1/matches/1/salts"] = self.null_salts
        self.api.responses["/v1/matches/2/salts"] = {**self.available_salts, "match_id": 2}
        with patch("agent.acquire.LOG"):
            selected = list(acquire.select_matches(self.api, 5, None, lambda mid: mid == 3))
        self.assertEqual(selected, [2])
        self.assertFalse(any("/metadata" in path or path == "/v1/sql" for path in self.api.network_calls))
        self.assertNotIn("/v1/matches/3/salts", self.api.network_calls)
        self.assertNotIn("/v1/matches/4/salts", self.api.network_calls)

    def test_metadata_only_fallback_does_not_require_replay_salts(self):
        self.api.lake_query.side_effect = RuntimeError("Data lake unavailable")
        self.api.responses["/v1/matches/recently-fetched"] = [
            {"match_id": 1, "start_time": 4, "game_mode": 1, "duration_s": 1200},
        ]
        with patch("agent.acquire.LOG"):
            selected = list(acquire.select_matches(self.api, 5, None, require_replay=False))
        self.assertEqual(selected, [1])
        self.assertEqual(self.api.network_calls, ["/v1/matches/recently-fetched"])

    def test_fresh_missing_salts_are_not_immediately_requested_twice(self):
        self.api.responses["/v1/matches/1/salts"] = self.null_salts
        salts = acquire.get_salts(self.api, 1)
        self.assertIsNone(salts["demo_url"])
        self.assertEqual(self.api.network_calls, ["/v1/matches/1/salts"])

    def test_cached_missing_salts_are_refreshed_once(self):
        save_json(self.root / "raw/1/sources/api/salts.json.gz", self.null_salts)
        self.api.responses["/v1/matches/1/salts"] = self.available_salts
        salts = acquire.get_salts(self.api, 1)
        self.assertEqual(salts["demo_url"], self.available_salts["demo_url"])
        self.assertEqual(self.api.network_calls, ["/v1/matches/1/salts"])

    def test_cached_available_salts_require_no_request(self):
        save_json(self.root / "raw/1/sources/api/salts.json.gz", self.available_salts)
        salts = acquire.get_salts(self.api, 1)
        self.assertEqual(salts["demo_url"], self.available_salts["demo_url"])
        self.assertEqual(self.api.network_calls, [])

    def test_missing_replay_is_unavailable_without_attempting_parser(self):
        self.api.responses["/v1/matches/1/salts"] = self.null_salts
        with patch("agent.acquire.subprocess.run") as parser:
            result = acquire.acquire_replay(self.api, 1, 10)
        self.assertFalse(result["replay_available"])
        self.assertFalse(result["replay_parsed"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["unavailable_reason"])
        self.api.download.assert_not_called()
        parser.assert_not_called()
        self.assertEqual(self.api.network_calls, ["/v1/matches/1/salts"])


if __name__ == "__main__":
    unittest.main()
