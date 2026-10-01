import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import acquire
from agent.io import read_json, save_json


class FakeAPI:
    def __init__(self, root, schema=None):
        self.root = Path(root)
        self.requests = []
        self.session = SimpleNamespace(close=Mock())
        self.schema = schema or {"tables": [{"name": "HeroKilledEvent", "kind": "event"}]}
        self.children = []
        self.on_schema = lambda: None
        self.json = Mock(side_effect=self.get_json)
        self.download = Mock(side_effect=self.get_download)

    def fork(self):
        child = FakeAPI(self.root, self.schema)
        child.on_schema = self.on_schema
        self.children.append(child)
        return child

    def get_json(self, path, filename=None, **kwargs):
        if path != "/v1/matches/demo/schema":
            raise AssertionError(f"Unexpected request: {path}")
        self.on_schema()
        self.requests.append({"url": path})
        if isinstance(self.schema, Exception):
            raise self.schema
        save_json(self.root / filename, self.schema)
        return self.schema, {}

    def get_download(self, url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PBDEtest")
        return target


class AcquireOverlapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.api = FakeAPI(self.root)
        self.salts = {"demo_url": "https://example.invalid/replay.dem.bz2"}
        self.out = self.root / "raw/1/sources/replay"

    def write_extracts(self, out=None, interval=20):
        out = out or self.out
        out.mkdir(parents=True, exist_ok=True)
        script = Path(acquire.__file__).parent / "replay/export.mjs"
        summary = {"success": True, "sample_interval": interval,
                   "exporter_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
        save_json(out / "summary.json.gz", summary)
        save_json(out / "schema.json.gz", {"tables": []})
        for name in ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz"):
            (out / name).write_bytes(b"fixture")
        return summary

    def test_schema_and_query_overlap_download_and_parser(self):
        schema_started = threading.Event()
        download_started = threading.Event()
        query_started = threading.Event()
        parser_finished = threading.Event()

        def schema():
            schema_started.set()
            self.assertTrue(download_started.wait(3), "Schema blocked the local download")

        def download(url, target):
            download_started.set()
            self.assertTrue(schema_started.wait(3))
            self.assertTrue(query_started.wait(3), "Query did not start while downloading")
            return self.api.get_download(url, target)

        def query(api, mid, schema):
            query_started.set()
            self.assertTrue(parser_finished.wait(3), "Remote query blocked local parsing")
            api.requests.append({"url": "query"})
            return {"status": "done", "path": "audit.parquet"}

        def parse(command, **kwargs):
            if command[0] == "node":
                self.write_extracts(Path(command[4]))
                save_json(Path(command[4]) / "schema.json", {"tables": []})
                parser_finished.set()
            return SimpleNamespace(returncode=0)

        self.api.on_schema = schema
        self.api.download.side_effect = download
        with patch("agent.acquire.query_demo", side_effect=query), patch("agent.acquire.subprocess.run", side_effect=parse):
            result = acquire.acquire_replay(self.api, 1, 20, save_demo=False, prefetched_salts=self.salts)
        self.assertTrue(result["replay_parsed"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["demo_query"]["status"], "done")
        self.assertEqual(len(self.api.children), 1)
        child = self.api.children[0]
        self.assertIsNot(self.api.session, child.session)
        self.assertIsNot(self.api.requests, child.requests)
        child.session.close.assert_called_once()
        self.assertEqual(self.api.requests, [{"url": "/v1/matches/demo/schema"}, {"url": "query"}])
        self.assertEqual(set(result["timings_seconds"]), {"download", "decompress", "parse", "remote_audit", "total"})
        self.assertFalse((self.out.parent / "replay.dem.bz2").exists())
        self.assertFalse((self.out.parent / "replay.dem").exists())
        self.assertEqual(read_json(self.out.parent / "schema/replay_status.json"), result)

    def test_query_disabled_still_archives_and_audits_schema(self):
        with patch("agent.acquire.query_demo") as query:
            result = acquire.acquire_replay(self.api, 1, 20, save_demo=False, parse=False,
                                            query=False, prefetched_salts=self.salts)
        query.assert_not_called()
        self.api.download.assert_not_called()
        self.assertTrue(result["replay_available"])
        self.assertTrue(read_json(self.out.parent / "schema/api_demo_schema.json"))
        self.assertTrue(read_json(self.out.parent / "schema/declared_event_tables.json"))

    def test_compatible_extracts_reused_without_downloading_discarded_demo(self):
        previous = self.write_extracts()
        with patch("agent.acquire.subprocess.run") as parser:
            result = acquire.acquire_replay(self.api, 1, 20, save_demo=False, query=False,
                                            prefetched_salts=self.salts)
        parser.assert_not_called()
        self.api.download.assert_not_called()
        self.assertEqual(result["parser_summary"], previous)
        self.assertTrue(result["replay_parsed"])
        self.assertTrue(result["parser_reused"])
        self.assertTrue(read_json(self.out.parent / "schema/declared_event_tables.json"))

    def test_compatible_extracts_work_when_replay_url_has_expired(self):
        self.write_extracts()
        result = acquire.acquire_replay(self.api, 1, 20, save_demo=False, prefetched_salts={})
        self.assertTrue(result["replay_available"])
        self.assertTrue(result["replay_parsed"])
        self.assertNotIn("unavailable_reason", result)
        self.api.json.assert_not_called()
        self.api.download.assert_not_called()

    def test_keeping_demo_still_downloads_when_extracts_are_cached(self):
        self.write_extracts()
        result = acquire.acquire_replay(self.api, 1, 20, save_demo=True, query=False,
                                        prefetched_salts=self.salts)
        self.api.download.assert_called_once()
        self.assertTrue(result["replay_parsed"])
        self.assertTrue((self.out.parent / "replay.dem.bz2").exists())

    def test_schema_failure_does_not_discard_local_replay(self):
        self.api.schema = RuntimeError("Service unavailable")
        with patch("agent.acquire.LOG"):
            result = acquire.acquire_replay(self.api, 1, 20, parse=False, prefetched_salts=self.salts)
        self.assertTrue(result["replay_available"])
        self.assertEqual(result["errors"], ["demo schema: Service unavailable"])
        self.assertTrue((self.out.parent / "replay.dem.bz2").exists())
        self.api.children[0].session.close.assert_called_once()

    def test_query_failure_does_not_discard_cached_extracts(self):
        self.write_extracts()
        with patch("agent.acquire.query_demo", side_effect=RuntimeError("Query unavailable")), patch("agent.acquire.LOG"):
            result = acquire.acquire_replay(self.api, 1, 20, save_demo=False, prefetched_salts=self.salts)
        self.assertTrue(result["replay_parsed"])
        self.assertEqual(result["errors"], ["API demo query: Query unavailable"])

    def test_download_failure_still_joins_and_records_remote_audit(self):
        self.api.download.side_effect = RuntimeError("Download unavailable")
        with patch("agent.acquire.query_demo", return_value={"status": "done"}), patch("agent.acquire.LOG"):
            result = acquire.acquire_replay(self.api, 1, 20, prefetched_salts=self.salts)
        self.assertFalse(result["replay_parsed"])
        self.assertEqual(result["demo_query"], {"status": "done"})
        self.assertEqual(result["errors"], ["local replay: Download unavailable"])
        self.api.children[0].session.close.assert_called_once()

    def test_prefetched_salts_failure_is_not_requested_again(self):
        self.write_extracts()
        result = acquire.acquire_replay(self.api, 1, 20, save_demo=False,
                                        prefetched_errors={"salts": RuntimeError("Rate limited")})
        self.api.json.assert_not_called()
        self.api.download.assert_not_called()
        self.assertTrue(result["replay_parsed"])
        self.assertEqual(result["errors"], ["salts: Rate limited"])

    def test_prefetched_download_failure_is_not_retried_and_audit_finishes(self):
        with patch("agent.acquire.query_demo", return_value={"status": "done"}), patch("agent.acquire.LOG"):
            result = acquire.acquire_replay(self.api, 1, 20, prefetched_salts=self.salts,
                                            prefetched_errors={"download": RuntimeError("Rate limited")})
        self.api.download.assert_not_called()
        self.assertFalse(result["replay_parsed"])
        self.assertEqual(result["demo_query"], {"status": "done"})
        self.assertEqual(result["errors"], ["local replay: Rate limited"])
        self.api.children[0].session.close.assert_called_once()

    def test_prefetched_download_failure_can_reuse_verified_local_demo(self):
        compressed = self.out.parent / "replay.dem.bz2"
        self.api.get_download(self.salts["demo_url"], compressed)
        save_json(compressed.with_suffix(compressed.suffix + ".provenance.json"),
                  {"sha256": hashlib.sha256(compressed.read_bytes()).hexdigest()})
        result = acquire.acquire_replay(self.api, 1, 20, parse=False, query=False,
                                        prefetched_salts=self.salts,
                                        prefetched_errors={"download": RuntimeError("Rate limited")})
        self.api.download.assert_not_called()
        self.assertTrue(result["replay_available"])
        self.assertEqual(result["demo_bytes"], compressed.stat().st_size)
        self.assertEqual(result["errors"], [])

    def test_prefetched_download_failure_does_not_accept_corrupt_local_demo(self):
        compressed = self.out.parent / "replay.dem.bz2"
        self.api.get_download(self.salts["demo_url"], compressed)
        save_json(compressed.with_suffix(compressed.suffix + ".provenance.json"), {"sha256": "wrong"})
        with patch("agent.acquire.decompress_demo") as decompress, patch("agent.acquire.LOG"):
            result = acquire.acquire_replay(self.api, 1, 20, query=False, prefetched_salts=self.salts,
                                            prefetched_errors={"download": RuntimeError("Rate limited")})
        self.api.download.assert_not_called()
        decompress.assert_not_called()
        self.assertFalse(result["replay_parsed"])
        self.assertEqual(result["errors"], ["local replay: Rate limited"])

    def test_incomplete_or_different_interval_extracts_are_not_reused(self):
        self.write_extracts()
        self.assertIsNone(acquire.compatible_local_replay(self.out, 10))
        (self.out / "events.ndjson.gz").unlink()
        self.assertIsNone(acquire.compatible_local_replay(self.out, 20))

    def test_changed_exporter_invalidates_extracts(self):
        previous = self.write_extracts()
        save_json(self.out / "summary.json.gz", {**previous, "exporter_sha256": "old"})
        self.assertIsNone(acquire.compatible_local_replay(self.out, 20))

    def test_cached_completed_query_is_not_submitted_again(self):
        folder = self.root / "raw/1/sources/api/demo_query"
        save_json(folder / "status.json.gz", {"status": "done", "job_id": "123"})
        save_json(folder / "submission.json.provenance.json.gz",
                  {"request_body": {"query": "SELECT * FROM HeroKilledEvent"}})
        (folder / "hero_killed.parquet").write_bytes(b"fixture")
        result = acquire.query_demo(self.api, 1, self.api.schema)
        self.assertEqual(result["job_id"], "123")
        self.api.json.assert_not_called()
        self.api.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
