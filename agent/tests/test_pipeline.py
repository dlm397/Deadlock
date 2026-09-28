import logging
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from agent import pipeline
from agent.io import API, GROUPS, checksum, read_json, save_json, save_match


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.fingerprint = self.stack.enter_context(patch("agent.pipeline.signature", return_value="test-v1"))
        self.stack.enter_context(patch("agent.pipeline.compact_sources"))
        self.stack.enter_context(patch("agent.pipeline.logging.basicConfig"))
        self.stack.enter_context(patch("agent.pipeline.logging.FileHandler", return_value=logging.NullHandler()))
        self.stack.enter_context(patch("agent.pipeline.LOG"))
        self.calls = []
        self.failures = set()
        self.builder = self.stack.enter_context(patch("agent.pipeline.build_match", side_effect=self.build))

    def build(self, api, mid, interval, parse, query, save_demo, offline, fingerprint):
        self.calls.append(mid)
        if mid in self.failures:
            raise ValueError(f"Fixture failure for {mid}")
        folder = api.root / "raw" / str(mid)
        status = save_match(folder, {"matches": [{"match_id": mid}], "player_state": []},
                            {"match_id": mid, "processed_success": True, "signature": fingerprint,
                             "sample_interval": interval})
        status["files"] = {f"{group}.zip": checksum(folder / f"{group}.zip") for group in GROUPS}
        save_json(folder / "sources/status.json", status)
        return status

    def run_matches(self, matches, **settings):
        return pipeline.run(output_dir=self.root, match_ids=matches, offline=True,
                            parse_replays=False, query_demo_api=False, **settings)

    def test_failed_match_does_not_stop_later_matches(self):
        self.failures.add(2)
        result = self.run_matches([1, 2, 3])
        self.assertEqual(result, {"saved": 2, "skipped": 0, "failed": 1})
        self.assertEqual(self.calls, [1, 2, 3])
        self.assertTrue(pipeline.complete(self.root / "raw/1", "test-v1"))
        self.assertFalse(pipeline.complete(self.root / "raw/2", "test-v1"))
        self.assertTrue(pipeline.complete(self.root / "raw/3", "test-v1"))
        catalog = read_json(self.root / "tables/catalog.json")
        self.assertTrue(catalog["1"]["processed_success"])
        self.assertFalse(catalog["2"]["processed_success"])
        self.assertTrue(catalog["3"]["processed_success"])
        self.assertIn("Fixture failure", read_json(self.root / "raw/2/sources/last_error.json")["error"])

    def test_unavailable_replay_is_skipped_and_later_matches_are_saved(self):
        def build(api, mid, *args):
            if mid == 2:
                raise pipeline.ReplayUnavailable("No replay URL is available")
            return self.build(api, mid, *args)

        self.builder.side_effect = build
        result = self.run_matches([1, 2, 3])
        self.assertEqual(result, {"saved": 2, "skipped": 1, "failed": 0})
        self.assertFalse(pipeline.complete(self.root / "raw/2"))
        self.assertTrue(pipeline.complete(self.root / "raw/3", "test-v1"))
        skipped = read_json(self.root / "raw/2/sources/last_error.json")
        self.assertTrue(skipped["skipped"])
        self.assertIn("No replay URL", skipped["reason"])

    def test_complete_match_is_skipped_unless_reprocess_requested(self):
        self.run_matches([1])
        before = {path.name: path.read_bytes() for path in (self.root / "raw/1").glob("*.zip")}
        self.calls.clear()
        self.assertEqual(self.run_matches([1]), {"saved": 0, "skipped": 1, "failed": 0})
        self.assertEqual(self.calls, [])
        after = {path.name: path.read_bytes() for path in (self.root / "raw/1").glob("*.zip")}
        self.assertEqual(before, after)
        self.assertEqual(self.run_matches([1], reprocess=True), {"saved": 1, "skipped": 0, "failed": 0})
        self.assertEqual(self.calls, [1])

    def test_signature_change_rebuilds_completed_match(self):
        self.run_matches([1])
        self.calls.clear()
        self.fingerprint.return_value = "test-v2"
        self.assertEqual(self.run_matches([1]), {"saved": 1, "skipped": 0, "failed": 0})
        self.assertEqual(self.calls, [1])
        self.assertTrue(pipeline.complete(self.root / "raw/1", "test-v2"))

    def test_corrupted_archive_is_not_treated_as_complete(self):
        self.run_matches([1])
        (self.root / "raw/1/events.zip").write_bytes(b"incomplete archive")
        self.calls.clear()
        self.assertEqual(self.run_matches([1]), {"saved": 1, "skipped": 0, "failed": 0})
        self.assertEqual(self.calls, [1])
        self.assertTrue(pipeline.complete(self.root / "raw/1", "test-v1"))

    def test_corrupted_status_is_rebuilt_without_stopping_run(self):
        self.run_matches([1])
        (self.root / "raw/1/sources/status.json").write_bytes(b"not valid JSON")
        self.calls.clear()
        self.assertEqual(self.run_matches([1, 2]), {"saved": 2, "skipped": 0, "failed": 0})
        self.assertEqual(self.calls, [1, 2])
        self.assertTrue(pipeline.complete(self.root / "raw/1", "test-v1"))

    def test_failed_reprocess_preserves_previous_success(self):
        self.run_matches([1])
        before = {path.name: path.read_bytes() for path in (self.root / "raw/1").glob("*.zip")}
        self.failures.add(1)
        self.assertEqual(self.run_matches([1], reprocess=True), {"saved": 0, "skipped": 0, "failed": 1})
        after = {path.name: path.read_bytes() for path in (self.root / "raw/1").glob("*.zip")}
        self.assertEqual(before, after)
        self.assertTrue(pipeline.complete(self.root / "raw/1", "test-v1"))
        self.assertTrue(read_json(self.root / "tables/catalog.json")["1"]["processed_success"])


class ArchiveRequestTests(unittest.TestCase):
    def test_failed_response_preserves_archived_successful_bytes(self):
        with tempfile.TemporaryDirectory() as temporary, patch("agent.io.requests.Session") as constructor, patch("agent.io.time.sleep"):
            root = Path(temporary)
            relative = "raw/1/sources/api/metadata.json"
            path = root / relative
            path.parent.mkdir(parents=True)
            successful = b'{"match_info":{"match_id":1}}'
            path.write_bytes(successful)
            sidecar = path.with_suffix(".json.provenance.json")
            sidecar.write_bytes(b'{"retrieved_at":"previous-success"}')
            response = Mock(url="https://api.deadlock-api.com/v1/matches/1/metadata",
                            status_code=503, content=b'{"error":"temporarily unavailable"}')
            response.request.method = "GET"
            response.headers = {}
            response.raise_for_status.side_effect = requests.HTTPError("503", response=response)
            constructor.return_value.request.return_value = response
            api = API(root)
            with self.assertRaises(requests.HTTPError):
                api.json("/v1/matches/1/metadata", relative)
            self.assertEqual(path.read_bytes(), successful)
            self.assertEqual(sidecar.read_bytes(), b'{"retrieved_at":"previous-success"}')
            response.json.assert_not_called()
            self.assertEqual(api.requests[-1]["status"], 503)

    def test_invalid_json_response_preserves_archived_successful_bytes(self):
        with tempfile.TemporaryDirectory() as temporary, patch("agent.io.requests.Session") as constructor:
            root = Path(temporary)
            path = root / "metadata.json"
            path.write_bytes(b'{"valid":true}')
            response = Mock(url="https://example.invalid/metadata", status_code=200, content=b"not json")
            response.request.method = "GET"
            response.json.side_effect = ValueError("Invalid JSON")
            constructor.return_value.request.return_value = response
            with self.assertRaises(ValueError):
                API(root).json("/metadata", "metadata.json")
            self.assertEqual(path.read_bytes(), b'{"valid":true}')


if __name__ == "__main__":
    unittest.main()
