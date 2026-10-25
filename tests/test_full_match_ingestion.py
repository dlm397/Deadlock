"""Bounded, offline ingestion checks; real DuckDB, synthetic parser observations.

Run: python -m unittest discover -s tests -p test_full_match_ingestion.py -v
"""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb

from app import full_match, sources


def metadata(match_id):
    # Deliberately reverse the roster: alignment must use slot, not list position.
    return {"match_info": {
        "match_id": match_id, "duration_s": 1800, "game_mode": 1,
        "match_mode": 1, "match_outcome": 0, "winning_team": 1,
        "players": [{"player_slot": slot, "account_id": 1000 + slot,
                     "hero_id": slot + 1, "team": slot // 6,
                     "kills": slot, "deaths": 1, "assists": 2,
                     "player_rank_data": {"initial_display_rank": 50}}
                    for slot in reversed(range(12))],
    }}


def parsed(replay, payload, interval):
    return (
        {"success": True, "parser": "deadem", "parser_version": "4.0.1",
         "client_version": 6730, "interval_s": interval,
         "final_observed_game_time_s": 1800, "replay_reaches_match_end": True,
         "tick_interval_s": 1 / 64, "clock_method": "synthetic_clock"},
        [{"player_slot": slot, "game_time_s": time,
          "observed_time_s": time if time != 20 else 19.984375,
          "is_final_observation": time == 1800, "net_worth": 1000 + slot,
          "alive": True, "x": float(slot)}
         for time in (0, 20, 1800) for slot in range(12)],
        [],
    )


class FullMatchIngestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="full-match-db-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "full_match"
        self.config = {
            "output_dir": self.output, "match_ids": [123],
            "sampling_interval_s": 20, "batch_size": 1,
            "retry_failures": False, "rebuild_outputs": False,
            "api_base": "https://unused.invalid", "disable_steam": True,
        }
        self.seed_replay(123)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(sources.SESSION, "request",
                                        side_effect=AssertionError("network is forbidden")))
        self.fetch = stack.enter_context(patch.object(
            sources, "metadata", side_effect=lambda mid, config: metadata(mid)))
        self.parse = stack.enter_context(patch.object(sources, "parse_replay", side_effect=parsed))
        stack.enter_context(patch.object(sources, "definitions", return_value=[]))
        self.download = stack.enter_context(patch.object(
            sources, "download_replay", side_effect=AssertionError("replay downloads are forbidden")))
        self.discover = stack.enter_context(patch.object(
            sources, "discover", side_effect=AssertionError("remote discovery is forbidden")))
        stack.enter_context(patch.object(full_match, "log"))

    def seed_replay(self, match_id):
        replay = self.output / "replays" / f"{match_id}.dem"
        replay.parent.mkdir(parents=True, exist_ok=True)
        replay.write_bytes(b"PBDEsynthetic replay fixture")
        return replay

    @property
    def database(self):
        return self.output / "full_match.duckdb"

    def query(self, sql, params=None):
        with duckdb.connect(str(self.database), read_only=True) as connection:
            return connection.execute(sql, params or []).fetchall()

    def test_success_rerun_skips_without_duplicates_or_reparsing(self):
        self.assertEqual(full_match.run(self.config, "ingest")["complete"], 1)
        before = self.query("SELECT * FROM player_snapshots ORDER BY game_time_s, player_slot")
        self.assertEqual(len(before), 36)
        self.assertEqual(full_match.run(self.config, "ingest")["complete"], 0)
        self.assertEqual(self.query("SELECT * FROM player_snapshots ORDER BY game_time_s, player_slot"), before)
        self.assertEqual(self.query("SELECT count(*) FROM matches"), [(1,)])
        self.assertEqual(self.query("SELECT count(*) FROM processing_attempts"), [(1,)])
        self.parse.assert_called_once()
        self.fetch.assert_called_once()
        self.download.assert_not_called()
        self.discover.assert_not_called()
        self.assertEqual(self.query("SELECT count(*) FROM player_snapshots WHERE observed_time_s > game_time_s"), [(0,)])
        self.assertEqual(self.query("SELECT player_slot, hero_id FROM match_players ORDER BY player_slot"),
                         [(slot, slot + 1) for slot in range(12)])
        self.assertTrue((self.output / "replays/123.dem").exists())
        self.assertFalse(list(self.output.rglob("*.csv")))
        self.assertFalse(list(self.output.rglob("*.json")))
        self.assertFalse(list(self.output.rglob("*.parquet")))

    def test_failure_is_recorded_and_retry_recovers(self):
        self.parse.side_effect = RuntimeError("synthetic parser failure")
        self.assertEqual(full_match.run(self.config, "ingest")["failed"], 1)
        self.assertEqual(self.query("SELECT count(*) FROM matches"), [(0,)])
        self.assertEqual(self.query("SELECT status FROM processing_attempts"), [("failed",)])
        self.parse.side_effect = parsed
        self.assertEqual(full_match.run(self.config, "retry")["complete"], 1)
        self.assertEqual(self.query("SELECT count(*) FROM matches"), [(1,)])
        self.assertEqual(self.query("SELECT count(*) FROM processing_attempts"), [(2,)])

    def test_failed_reprocessing_preserves_successful_data_and_interval(self):
        full_match.run(self.config, "ingest")
        before = self.query("SELECT * FROM matches")
        states = self.query("SELECT * FROM player_snapshots ORDER BY game_time_s, player_slot")
        self.config["sampling_interval_s"] = 10
        self.parse.side_effect = RuntimeError("synthetic reprocess failure")
        self.assertEqual(full_match.run(self.config, "reprocess")["failed"], 1)
        self.assertEqual(self.query("SELECT * FROM matches"), before)
        self.assertEqual(self.query("SELECT * FROM player_snapshots ORDER BY game_time_s, player_slot"), states)
        self.assertEqual(self.query("SELECT a.sampling_interval_s FROM processing_attempts a "
                                    "JOIN matches m ON a.attempt_id = m.successful_attempt_id"), [(20,)])

    def test_bounded_batch_advances_past_successful_local_replays(self):
        self.seed_replay(456)
        self.seed_replay(789)
        self.config["match_ids"] = []
        self.assertEqual(full_match.run(self.config, "ingest")["complete"], 1)
        self.assertEqual(full_match.run(self.config, "ingest")["complete"], 1)
        self.assertEqual(self.query("SELECT count(*) FROM matches"), [(2,)])
        self.assertEqual(self.parse.call_count, 2)
        self.assertEqual(self.fetch.call_count, 2)

    def test_reprocess_eligibility_rejection_preserves_previous_success(self):
        full_match.run(self.config, "ingest")
        before = self.query("SELECT * FROM matches")
        rejected = metadata(123)
        rejected["match_info"]["duration_s"] = 600
        self.fetch.side_effect = None
        self.fetch.return_value = rejected
        full_match.run(self.config, "reprocess")
        self.assertEqual(self.query("SELECT * FROM matches"), before)
        self.assertEqual(self.query("SELECT count(*) FROM player_snapshots"), [(36,)])
        self.assertEqual(self.parse.call_count, 1)

    def test_future_observation_fails_without_publishing_match(self):
        def future(replay, payload, interval):
            summary, samples, events = parsed(replay, payload, interval)
            samples[0]["observed_time_s"] = samples[0]["game_time_s"] + 0.1
            return summary, samples, events
        self.parse.side_effect = future
        self.assertEqual(full_match.run(self.config, "ingest")["failed"], 1)
        self.assertEqual(self.query("SELECT count(*) FROM matches"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM player_snapshots"), [(0,)])

    def test_resume_recovers_an_interrupted_attempt(self):
        self.parse.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            full_match.run(self.config, "ingest")
        self.assertEqual(self.query("SELECT count(*) FROM matches"), [(0,)])
        self.parse.side_effect = parsed
        self.config["match_ids"] = []
        self.assertEqual(full_match.run(self.config, "resume")["complete"], 1)
        self.assertEqual(self.query("SELECT count(*) FROM matches"), [(1,)])
        self.assertEqual(self.query("SELECT count(*) FROM processing_attempts"), [(2,)])
        self.assertEqual(self.query("SELECT count(*) FROM processing_attempts WHERE status = 'running'"), [(0,)])

    def test_uint64_match_id_remains_exact(self):
        match_id = 2**63 + 17
        self.seed_replay(match_id)
        self.config["match_ids"] = [match_id]
        self.assertEqual(full_match.run(self.config, "ingest")["complete"], 1)
        self.assertEqual(self.query("SELECT match_id FROM matches"), [(match_id,)])

    def test_explicit_replay_outside_collection_uses_selected_path(self):
        replay = self.root / "external-recording.dem"
        original = b"PBDEexternal synthetic replay"
        replay.write_bytes(original)
        self.config["replay_path"] = replay
        self.assertEqual(full_match.run(self.config, "ingest")["complete"], 1)
        self.assertEqual(self.parse.call_args.args[0], replay.resolve())
        self.assertEqual(self.query("SELECT path FROM replay_sources"), [(str(replay.resolve()),)])
        self.assertEqual(replay.read_bytes(), original)

    def test_missing_replay_failure_retries_after_file_is_supplied(self):
        self.config["match_ids"] = [456]
        self.assertEqual(full_match.run(self.config, "ingest")["failed"], 1)
        self.assertEqual(self.query("SELECT status,stage FROM processing_attempts"), [("failed", "source")])
        self.fetch.assert_not_called()
        self.parse.assert_not_called()
        supplied = self.output / "replays" / "456.dem.zst"
        supplied.write_bytes(b"\x28\xb5\x2f\xfdsynthetic compressed fixture")
        self.config["match_ids"] = []
        self.assertEqual(full_match.run(self.config, "retry")["complete"], 1)
        self.assertEqual(self.parse.call_args.args[0], supplied.resolve())
        self.assertEqual(self.query("SELECT match_id FROM matches"), [(456,)])

    def test_ingest_requires_explicit_reprocessing_to_change_interval(self):
        full_match.run(self.config, "ingest")
        before = self.query("SELECT * FROM matches")
        self.config["sampling_interval_s"] = 10
        with self.assertRaisesRegex(ValueError, "reprocess"):
            full_match.run(self.config, "ingest")
        self.assertEqual(self.query("SELECT * FROM matches"), before)
        self.assertEqual(self.parse.call_count, 1)

    def test_resume_recovers_interrupted_reprocessing_at_recorded_interval(self):
        full_match.run(self.config, "ingest")
        before = self.query("SELECT * FROM matches")
        self.config["sampling_interval_s"] = 10
        self.parse.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            full_match.run(self.config, "reprocess")
        self.assertEqual(self.query("SELECT * FROM matches"), before)
        self.parse.side_effect = parsed
        self.config["sampling_interval_s"] = 20
        self.config["match_ids"] = []
        self.assertEqual(full_match.run(self.config, "resume")["complete"], 1)
        self.assertEqual(self.parse.call_args.args[2], 10)
        self.assertEqual(self.query("SELECT a.sampling_interval_s FROM processing_attempts a "
                                    "JOIN matches m ON a.attempt_id = m.successful_attempt_id"), [(10,)])
        self.assertEqual(self.query("SELECT count(*) FROM player_snapshots"), [(36,)])

    def test_retry_recovers_failed_reprocessing_at_recorded_interval(self):
        full_match.run(self.config, "ingest")
        self.config["sampling_interval_s"] = 10
        self.parse.side_effect = RuntimeError("failed resampling")
        self.assertEqual(full_match.run(self.config, "reprocess")["failed"], 1)
        self.parse.side_effect = parsed
        self.config["sampling_interval_s"] = 20
        self.config["match_ids"] = []
        self.assertEqual(full_match.run(self.config, "retry")["complete"], 1)
        self.assertEqual(self.parse.call_args.args[2], 10)
        self.assertEqual(self.query("SELECT a.sampling_interval_s FROM processing_attempts a "
                                    "JOIN matches m ON a.attempt_id = m.successful_attempt_id"), [(10,)])

    def test_resume_scan_skips_completed_matches_with_other_intervals(self):
        self.config["sampling_interval_s"] = 10
        full_match.run(self.config, "ingest")
        self.seed_replay(456)
        self.config.update(match_ids=[], sampling_interval_s=20)
        self.assertEqual(full_match.run(self.config, "resume")["complete"], 1)
        self.assertEqual(self.query("SELECT m.match_id,a.sampling_interval_s FROM matches m "
                                    "JOIN processing_attempts a ON a.attempt_id=m.successful_attempt_id "
                                    "ORDER BY match_id"), [(123, 10), (456, 20)])


if __name__ == "__main__":
    unittest.main()
