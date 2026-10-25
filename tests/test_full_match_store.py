"""Small offline DuckDB checks; no real replay or network work."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import full_match_store as store


def metadata():
    return {"match_info": {"match_id": 123, "duration_s": 1800, "game_mode": 1,
        "match_mode": 4, "match_outcome": 0, "winning_team": 1,
        "players": [{"player_slot": slot, "account_id": 1000 + slot, "hero_id": slot + 1,
                     "team": slot // 6, "kills": 0, "deaths": 0,
                     "player_rank_data": {"initial_display_rank": 0 if slot == 0 else 50}}
                    for slot in range(12)]},
        "hero_build_ids": {"1000": 2**63 + 5}, "pregame_hero_ids": {"1000": 9},
        "banned_hero_ids": [22, 22, 33]}


def samples():
    return [{"player_slot": slot, "game_time_s": time, "observed_time_s": time,
             "is_final_observation": time == 20, "net_worth": slot * 100,
             "kills": 0, "alive": True, "x": float(slot),
             "global": {"observed_time_s": time, "game_state": 7, "paused": False, "server_tick": 100 + time}}
            for time in (0, 20) for slot in range(12)]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.replay = self.root / "123.dem"
        self.replay.write_bytes(b"PBDEsynthetic")
        self.connection = store.connect(self.root / "full_match.duckdb")
        self.addCleanup(self.connection.close)
        self.payload = metadata()
        self.summary = {"interval_s": 20, "client_version": 1234, "final_observed_game_time_s": 20,
                        "replay_reaches_match_end": False, "clock_method": "synthetic", "warnings": []}

    def attempt(self):
        attempt = store.start_attempt(self.connection, 123, self.replay, 20,
                                      metadata_source="https://unused.invalid/123", parser_revision="synthetic")
        store.record_metadata(self.connection, attempt, self.payload)
        return attempt

    def save(self, attempt=None, **kwargs):
        attempt = attempt or self.attempt()
        return store.replace_match(self.connection, attempt, self.payload, self.summary,
                                   kwargs.get("samples", samples()), kwargs.get("events", []),
                                   kwargs.get("definitions", []))

    def test_typed_write_alignment_missing_and_derived_teams(self):
        self.save()
        self.assertEqual(self.connection.execute("SELECT count(*) FROM player_snapshots").fetchone(), (24,))
        self.assertEqual(self.connection.execute("SELECT count(*) FROM global_snapshots").fetchone(), (2,))
        self.assertEqual(self.connection.execute("SELECT hero_build_id,pregame_hero_id,initial_rank_badge "
                                                "FROM match_players WHERE player_slot=0").fetchone(),
                         (2**63 + 5, 9, None))
        self.assertEqual(self.connection.execute("SELECT count(*) FROM match_bans").fetchone(), (2,))
        self.assertEqual(self.connection.execute("SELECT net_worth,unspent_souls,observed_player_count "
                                                "FROM team_snapshots WHERE team_id=0 AND game_time_s=20").fetchone(),
                         (1500, None, 6))
        self.assertFalse(store.needs_processing(self.connection, 123, 20))
        with self.assertRaisesRegex(ValueError, "reprocess"):
            store.needs_processing(self.connection, 123, 10)

    def test_successful_replace_and_failed_transaction_preserve_previous(self):
        self.save()
        self.save()
        self.assertEqual(self.connection.execute("SELECT count(*) FROM player_snapshots").fetchone(), (24,))
        successful = store.status(self.connection, 123)["successful_attempt_id"]
        latest = self.attempt()
        original = store._insert
        def fail_mid_write(connection, table, rows):
            if table == "global_snapshots":
                raise RuntimeError("synthetic write interruption after deletes")
            original(connection, table, rows)
        with patch.object(store, "_insert", side_effect=fail_mid_write):
            with self.assertRaisesRegex(RuntimeError, "synthetic write"):
                self.save(latest)
        store.finish_attempt(self.connection, latest, "failed", "write", "synthetic")
        result = store.status(self.connection, 123)
        self.assertEqual(result["successful_attempt_id"], successful)
        self.assertEqual((result["status"], result["latest_status"]), ("complete", "failed"))
        self.assertTrue(store.needs_processing(self.connection, 123, 20, retry=True))
        self.assertEqual(self.connection.execute("SELECT count(*) FROM player_snapshots").fetchone(), (24,))
        self.assertEqual(self.connection.execute("SELECT count(*) FROM global_snapshots").fetchone(), (2,))

    def test_future_duplicate_and_misaligned_players_rejected(self):
        for mutation, message in (("future", "future backfill"), ("duplicate", "duplicate time-slice"),
                                  ("missing", "complete roster"), ("identity", "conflicts")):
            with self.subTest(mutation=mutation):
                rows = samples()
                if mutation == "future":
                    rows[0]["observed_time_s"] = 0.1
                elif mutation == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif mutation == "missing":
                    rows.pop()
                else:
                    rows[0]["account_id"] = 123456
                with self.assertRaisesRegex(ValueError, message):
                    self.save(samples=rows)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM matches").fetchone(), (0,))

    def test_sparse_events_keep_precision_and_participants(self):
        self.payload["match_info"]["players"][1]["death_details"] = [
            {"game_time_s": 10, "killer_player_slot": 0, "death_duration_s": 12,
             "death_pos": {"x": 1, "y": 2, "z": 3}}]
        self.payload["match_info"]["players"][0]["items"] = [{"item_id": 42, "game_time_s": 2}]
        events = [
            {"event_type": "k_EUserMsg_HeroKilled", "game_time_s": 10.125,
             "actor_player_slot": 0, "target_player_slot": 1, "assister_player_slots": [2],
             "scorer_player_slot": 0, "attacker_player_slot": 3, "tick": 77, "network_tick": 88},
            {"event_type": "k_EUserMsg_BossKilled", "game_time_s": 12.25, "actor_player_slot": 0,
             "data": {"objectiveTeam": 2, "objectiveMaskChange": 8, "entityKilledClass": 99,
                      "bossesRemaining": 0, "entityPosition": {"x": 1, "y": 2, "z": 3}}},
            {"event_type": "k_EUserMsg_RejuvStatus", "game_time_s": 13.75, "actor_player_slot": 0,
             "data": {"eventType": 0, "killingTeam": 2, "userTeam": 3}},
        ]
        self.save(events=events, definitions=[{"id": 42, "name": "Synthetic", "type": "upgrade"}])
        self.assertEqual(self.connection.execute("SELECT event_time_s,scorer_player_slot,attacker_player_slot "
                                                "FROM hero_kills").fetchone(), (10.125, 0, 3))
        self.assertEqual(self.connection.execute("SELECT player_slot FROM kill_assists").fetchone(), (2,))
        self.assertEqual(self.connection.execute("SELECT objective_team_code,bosses_remaining FROM boss_notifications").fetchone(), (2, 0))
        self.assertEqual(self.connection.execute("SELECT event_type_code FROM rejuv_events").fetchone(), (0,))
        self.assertEqual(self.connection.execute("SELECT event_type,item_id FROM item_events").fetchone(), ("item_acquired", 42))

    def test_fractional_final_time_allows_missing_player_and_global_requires_evidence(self):
        rows = samples()
        for row in rows[12:]:
            row["game_time_s"] = row["observed_time_s"] = 20.25
            row["global"]["observed_time_s"] = 20.25
        missing = rows[-1]
        missing.update(observed_time_s=None, is_final_observation=False, net_worth=None,
                       kills=None, alive=None, x=None)
        self.save(samples=rows)
        self.assertEqual(self.connection.execute("SELECT observed_time_s FROM player_snapshots "
                                                "WHERE player_slot=11 AND game_time_s=20.25").fetchone(), (None,))
        rows = samples()
        for row in rows[:12]:
            row["global"]["observed_time_s"] = None
        with self.assertRaisesRegex(ValueError, "unobserved global"):
            self.save(samples=rows)

    def test_missing_source_attempt_is_recorded(self):
        attempt = store.start_attempt(self.connection, 456, self.root / "456.dem", 20)
        store.finish_attempt(self.connection, attempt, "failed", "replay_lookup", "missing")
        self.assertEqual(store.status(self.connection, 456)["status"], "failed")
        self.assertEqual(self.connection.execute("SELECT size_bytes FROM replay_sources WHERE match_id=456").fetchone(), (None,))


if __name__ == "__main__":
    unittest.main()
