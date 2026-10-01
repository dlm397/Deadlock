import logging
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import requests

from agent import pipeline
from agent.io import GROUPS, checksum, read_json, save_json, save_match


def metadata(eligible=True):
    return {"match_info": {"game_mode": 1 if eligible else 4, "duration_s": 1200,
                           "players": [{} for _ in range(12)], "winning_team": 0}}, {}


class PrepareCandidateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.child = SimpleNamespace(root=self.root, json=Mock(return_value=metadata()),
                                     download=Mock(), requests=[{"url": "metadata"}], session=Mock())
        self.api = SimpleNamespace(root=self.root, fork=Mock(return_value=self.child),
                                   json=Mock(), download=Mock(), requests=[])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.salts = self.stack.enter_context(patch("agent.pipeline.get_salts", return_value={"demo_url": "https://example.invalid/demo"}))
        self.cached = self.stack.enter_context(patch("agent.pipeline.compatible_local_replay", return_value=None))

    def prepare(self, **settings):
        options = dict(interval=20, parse=True, save_demo=True, automatic=True)
        options.update(settings)
        return pipeline.prepare_candidate(self.api, 1, **options)

    def test_metadata_is_fetched_once_with_isolated_session_and_provenance(self):
        result = self.prepare()
        self.api.fork.assert_called_once_with()
        self.child.json.assert_called_once_with("/v1/matches/1/metadata", "raw/1/sources/api/metadata.json",
                                                {"disable_steam": "true"}, cached=True)
        self.salts.assert_called_once_with(self.child, 1)
        self.child.download.assert_called_once_with("https://example.invalid/demo", self.root / "raw/1/sources/replay.dem.bz2")
        self.api.json.assert_not_called()
        self.api.download.assert_not_called()
        self.child.session.close.assert_called_once_with()
        self.assertIs(result["requests"], self.child.requests)
        self.assertEqual(self.api.requests, [])
        self.assertGreaterEqual(result["seconds"], 0)

    def test_ineligible_automatic_match_does_not_download(self):
        self.child.json.return_value = metadata(False)
        result = self.prepare()
        self.assertIn("metadata", result)
        self.salts.assert_not_called()
        self.child.download.assert_not_called()
        self.child.session.close.assert_called_once_with()

    def test_explicit_match_bypasses_automatic_eligibility(self):
        self.child.json.return_value = metadata(False)
        self.prepare(automatic=False)
        self.child.download.assert_called_once()

    def test_cached_extract_avoids_unwanted_demo_download(self):
        self.cached.return_value = {"success": True}
        self.prepare(save_demo=False)
        self.cached.assert_called_once_with(self.root / "raw/1/sources/replay", 20)
        self.child.download.assert_not_called()

    def test_metadata_only_collection_does_not_fetch_replay(self):
        self.prepare(parse=False, save_demo=False)
        self.salts.assert_not_called()
        self.child.download.assert_not_called()

    def test_metadata_error_is_returned_once_and_session_is_closed(self):
        failure = requests.ConnectionError("fixture metadata failure")
        self.child.json.side_effect = failure
        result = self.prepare()
        self.assertIs(result["error"], failure)
        self.child.json.assert_called_once()
        self.salts.assert_not_called()
        self.child.session.close.assert_called_once_with()

    def test_prefetch_http_failures_are_forwarded_without_another_attempt(self):
        for stage in ("salts", "download"):
            with self.subTest(stage=stage):
                failure = requests.HTTPError("fixture 503")
                self.salts.reset_mock(side_effect=True)
                self.child.download.reset_mock(side_effect=True)
                failing = self.salts if stage == "salts" else self.child.download
                failing.side_effect = failure
                result = self.prepare()
                self.assertNotIn("error", result)
                self.assertIs(result["prefetch_errors"][stage], failure)
                failing.assert_called_once()
                if stage == "salts":
                    self.child.download.assert_not_called()


class PreparedCandidatesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.api = SimpleNamespace(root=Path(temporary.name))

    def test_only_one_next_candidate_overlaps_current_processing(self):
        started = {mid: Event() for mid in (1, 2, 3)}
        release = {mid: Event() for mid in (2, 3)}

        def prepare(mid):
            started[mid].set()
            if mid in release and not release[mid].wait(5):
                raise AssertionError("Fixture preparation was not released")
            return {"match_id": mid}

        stream = pipeline.prepared_candidates(self.api, [1, 2, 3], prepare, lambda: True, lambda mid: False)
        try:
            self.assertEqual(next(stream)[0], 1)
            self.assertTrue(started[2].wait(5))
            self.assertFalse(started[3].is_set())
            release[2].set()
            self.assertEqual(next(stream)[0], 2)
            self.assertTrue(started[3].wait(5))
            release[3].set()
            self.assertEqual(next(stream)[0], 3)
            self.assertEqual(list(stream), [])
        finally:
            for event in release.values():
                event.set()
            stream.close()

    def test_shutdown_joins_pending_work_and_archives_its_requests(self):
        started, release, closing, closed = (Event() for _ in range(4))
        failures = []

        def prepare(mid):
            if mid == 2:
                started.set()
                if not release.wait(5):
                    raise AssertionError("Fixture preparation was not released")
            return {"match_id": mid, "requests": [{"url": f"match/{mid}"}]}

        stream = pipeline.prepared_candidates(self.api, [1, 2], prepare, lambda: True, lambda mid: False)
        self.assertEqual(next(stream)[0], 1)
        self.assertTrue(started.wait(5))

        def close():
            closing.set()
            try:
                stream.close()
            except BaseException as exc:
                failures.append(exc)
            finally:
                closed.set()

        thread = Thread(target=close)
        thread.start()
        try:
            self.assertTrue(closing.wait(5))
            self.assertFalse(closed.is_set())
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        path = self.api.root / "raw/2/sources/prefetch_requests.json"
        self.assertEqual(read_json(path), [{"url": "match/2"}])
        self.assertTrue(path.with_suffix(".json.gz").exists())

    def test_disabled_lookahead_does_not_start_next_candidate_until_advance(self):
        prepare = Mock(side_effect=lambda mid: {"match_id": mid})
        stream = pipeline.prepared_candidates(self.api, [1, 2], prepare, lambda: False, lambda mid: False)
        try:
            self.assertEqual(next(stream)[0], 1)
            self.assertEqual([call.args[0] for call in prepare.call_args_list], [1])
            self.assertEqual(next(stream)[0], 2)
            self.assertEqual([call.args[0] for call in prepare.call_args_list], [1, 2])
        finally:
            stream.close()


class PrefetchPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.api = SimpleNamespace(root=self.root, requests=[], json=Mock(return_value=([], {})), session=Mock())
        self.selection = SimpleNamespace(root=self.root, requests=[], session=Mock())
        self.api.fork = Mock(return_value=self.selection)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("agent.pipeline.API", return_value=self.api))
        self.stack.enter_context(patch("agent.pipeline.signature", return_value="new-code"))
        self.stack.enter_context(patch("agent.pipeline.logging.basicConfig"))
        self.stack.enter_context(patch("agent.pipeline.logging.FileHandler", return_value=logging.NullHandler()))
        self.stack.enter_context(patch("agent.pipeline.LOG"))
        self.stack.enter_context(patch("agent.pipeline.gc.collect"))
        self.prepared = []
        self.built = []
        self.preparer = self.stack.enter_context(patch("agent.pipeline.prepare_candidate", side_effect=self.prepare))
        self.builder = self.stack.enter_context(patch("agent.pipeline.build_match", side_effect=self.build))

    def prepare(self, api, mid, *args):
        self.prepared.append(mid)
        return {"match_id": mid, "metadata": metadata(), "seconds": 0, "requests": []}

    def build(self, api, mid, interval, parse, query, save_demo, offline, fingerprint, prepared=None):
        self.built.append(mid)
        folder = self.root / "raw" / str(mid)
        status = save_match(folder, {"matches": [{"match_id": mid}]},
                            {"match_id": mid, "processed_success": True, "signature": fingerprint,
                             "sample_interval": interval})
        status["files"] = {f"{group}.zip": checksum(folder / f"{group}.zip") for group in GROUPS}
        save_json(folder / "sources/status.json", status)
        return status

    def test_errors_and_skips_do_not_consume_target_or_overfetch(self):
        def prepare(api, mid, *args):
            result = self.prepare(api, mid, *args)
            if mid == 1:
                result["error"] = requests.ConnectionError("fixture unavailable metadata")
            if mid == 2:
                result["metadata"] = metadata(False)
            return result

        self.preparer.side_effect = prepare
        with patch("agent.pipeline.select_matches", return_value=iter([1, 2, 3, 4, 5])):
            result = pipeline.run(n_matches=2, output_dir=self.root)
        self.assertEqual(result, {"saved": 2, "skipped": 1, "failed": 1})
        self.assertEqual(self.prepared, [1, 2, 3, 4])
        self.assertEqual(self.built, [3, 4])
        self.assertIn("fixture unavailable metadata", read_json(self.root / "raw/1/sources/last_error.json")["error"])
        self.api.json.assert_called_once()
        self.api.session.close.assert_called_once()
        self.selection.session.close.assert_called_once()

    def test_automatic_discovery_excludes_complete_matches_with_same_signature(self):
        self.build(self.api, 1, 20, True, True, True, False, "new-code")
        self.built.clear()

        def select(api, count, matches, done, **kwargs):
            self.assertTrue(done(1))
            self.assertFalse(done(2))
            return iter([2])

        with patch("agent.pipeline.select_matches", side_effect=select):
            result = pipeline.run(n_matches=1, output_dir=self.root)
        self.assertEqual(result["saved"], 1)
        self.assertEqual(self.prepared, [2])
        self.assertEqual(self.built, [2])
        self.assertTrue(pipeline.complete(self.root / "raw/1", "new-code"))

    def test_automatic_collection_rebuilds_complete_match_with_old_signature(self):
        self.build(self.api, 1, 20, True, True, True, False, "old-code")
        self.built.clear()

        def select(api, count, matches, done, **kwargs):
            self.assertFalse(done(1))
            return iter([1])

        with patch("agent.pipeline.select_matches", side_effect=select):
            result = pipeline.run(n_matches=1, output_dir=self.root)
        self.assertEqual(result, {"saved": 1, "skipped": 0, "failed": 0})
        self.assertEqual(self.built, [1])
        self.assertTrue(pipeline.complete(self.root / "raw/1", "new-code"))

    def test_explicit_collection_rebuilds_complete_match_with_old_signature(self):
        self.build(self.api, 1, 20, True, True, True, False, "old-code")
        self.built.clear()
        with patch("agent.pipeline.select_matches", return_value=iter([1])):
            result = pipeline.run(match_ids=[1], output_dir=self.root)
        self.assertEqual(result, {"saved": 1, "skipped": 0, "failed": 0})
        self.assertEqual(self.built, [1])
        self.assertTrue(pipeline.complete(self.root / "raw/1", "new-code"))


class ReferenceCacheTests(unittest.TestCase):
    def test_same_definitions_are_written_once_and_new_version_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, written = Path(temporary), set()

            def definitions(version):
                provenance = {"definitions_version": version, "source": "fixture", "retrieved_at": "now"}
                return {"items": [{"id": 10, "type": "ability", "name": f"Ability {version}"}],
                        "heroes": [{"id": 20, "name": f"Hero {version}"}],
                        "items_provenance": provenance, "heroes_provenance": provenance}

            with patch("agent.pipeline.write_table", wraps=pipeline.write_table) as writer:
                pipeline.references(root, definitions(1), written)
                self.assertEqual(writer.call_count, 3)
                pipeline.references(root, definitions(1), written)
                self.assertEqual(writer.call_count, 3)
                pipeline.references(root, definitions(2), written)
                self.assertEqual(writer.call_count, 6)
            for name in ("heroes", "items", "abilities"):
                table = pd.read_parquet(root / "tables" / f"{name}.parquet")
                self.assertEqual(set(table["definitions_version"]), {1, 2})


class BuildPreparedTests(unittest.TestCase):
    def test_build_reuses_prefetched_metadata_and_forwards_network_errors(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            api = SimpleNamespace(root=root, json=Mock(return_value=([1], {})))
            failure = requests.ConnectionError("fixture interrupted download")
            prepared = {"metadata": metadata(), "salts": {"demo_url": "fixture"},
                        "prefetch_errors": {"download": failure}, "seconds": 0.1}
            replay = {"replay_parsed": True, "parser_summary": {"game_version": 1}}
            acquisition = stack.enter_context(patch("agent.pipeline.acquire_replay", return_value=replay))
            definitions = {name: [] for name in ("items", "heroes", "ranks", "generic-data")}
            definitions["items_provenance"] = {"definitions_version": 1}
            static = stack.enter_context(patch("agent.pipeline.static_data", return_value=definitions))
            tables = {"matches": [{}], "players": [], "player_state": []}
            stack.enter_context(patch("agent.pipeline.normalize_metadata", side_effect=lambda *args: dict(tables)))
            for name in ("audit_metadata", "merge_replay", "aggregate_teams", "references"):
                stack.enter_context(patch(f"agent.pipeline.{name}"))
            stack.enter_context(patch("agent.pipeline.validate_tables", return_value=[]))
            stack.enter_context(patch("agent.pipeline.save_match", side_effect=lambda folder, tables, status: status))
            stack.enter_context(patch("agent.pipeline.checksum", return_value="fixture-digest"))
            for mid in (1, 2):
                status = pipeline.build_match(api, mid, 20, True, True, True, False, "fixture", prepared)
                self.assertEqual(status["match_id"], mid)
            api.json.assert_called_once_with("/v1/assets/client-versions", "tables/sources/client_versions.json",
                                              required_archive=True, cached=True)
            static.assert_called_once_with(api, 1)
            self.assertEqual(acquisition.call_count, 2)
            for call in acquisition.call_args_list:
                self.assertIs(call.kwargs["prefetched_salts"], prepared["salts"])
                self.assertIs(call.kwargs["prefetched_errors"]["download"], failure)


if __name__ == "__main__":
    unittest.main()
