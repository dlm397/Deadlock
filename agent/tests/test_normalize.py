# Check missingness, timing and item semantics used by statistical models.

import unittest

from agent.normalize import aggregate_team_state, normalize_metadata
from agent.replay_data import enrich_purchase_state, merge_player_fields


def fixture(players=None, **extra):
    return {"match_info": {"match_id": 1, "duration_s": 400, "winning_team": 0,
                           "players": players or [], **extra}}


def player(account=10, team=0, stats=None, **extra):
    return {"account_id": account, "player_slot": account, "team": team, "hero_id": 3,
            "stats": stats or [], "items": [], **extra}


class NormalizeTests(unittest.TestCase):
    def normalize(self, data, definitions=None):
        return normalize_metadata(data, {"source": "test", "game_version": 1}, definitions or [], 10)

    def test_sparse_asof_and_missing_not_zero(self):
        result = self.normalize(fixture([player(stats=[{"time_stamp_s": 180, "net_worth": 0, "kills": None},
                                                       {"time_stamp_s": 360, "net_worth": 500}])]))
        panel = {r["game_time"]: r for r in result["player_state"]}
        self.assertIsNone(panel[170]["net_worth"])
        self.assertEqual(panel[180]["net_worth"], 0)
        self.assertIsNone(panel[180]["kills"])
        self.assertEqual(panel[350]["net_worth"], 0)
        self.assertEqual(panel[350]["source_game_time"], 180)
        self.assertEqual(panel[350]["state_age"], 170)
        self.assertEqual(panel[360]["net_worth"], 500)
        self.assertIsNone(result["players"][0]["player_damage"])
        self.assertFalse(result["players"][0]["final_stats_at_match_end"])

    def test_team_sum_requires_every_player(self):
        data = fixture([player(10, 0, [{"time_stamp_s": 180, "net_worth": 0}]),
                        player(11, 0, [{"time_stamp_s": 360, "net_worth": 5}]),
                        player(20, 1, [{"time_stamp_s": 180, "net_worth": 2}])])
        teams = {(r["game_time"], r["team"]): r for r in self.normalize(data)["team_state"]}
        self.assertIsNone(teams[(180, 0)]["team_net_worth"])
        self.assertEqual(teams[(180, 0)]["net_worth_observed_players"], 1)
        self.assertIsNone(teams[(180, 1)]["net_worth_advantage"])
        self.assertEqual(teams[(360, 0)]["team_net_worth"], 5)
        self.assertEqual(teams[(360, 0)]["net_worth_advantage"], 3)
        self.assertFalse(teams[(360, 0)]["economy_simultaneous_observations"])

    def test_absent_player_is_not_a_complete_team(self):
        rows = aggregate_team_state([player(10), player(11)],
                                    [{"game_time": 10, "team": 0, "net_worth": 50,
                                      "source_game_time": 10, "alive": True}], {})
        self.assertFalse(rows[0]["roster_complete"])
        self.assertIsNone(rows[0]["team_net_worth"])
        self.assertIsNone(rows[0]["alive_players"])
        self.assertEqual(rows[0]["net_worth_observed_players"], 1)

    def test_ability_is_not_purchase_and_removal_not_assumed_sale(self):
        items = [{"item_id": 1, "game_time_s": 5, "upgrade_id": 1, "sold_time_s": 30, "flags": 1},
                 {"item_id": 2, "game_time_s": 10, "upgrade_id": 0, "sold_time_s": 0},
                 {"item_id": 2, "game_time_s": 20, "upgrade_id": 888, "sold_time_s": 0}]
        definitions = [{"id": 1, "type": "upgrade", "name": "Shop item", "cost": 500},
                       {"id": 2, "type": "ability", "name": "Ability", "cost": 0}]
        result = self.normalize(fixture([player(items=items)]), definitions)
        self.assertEqual([r["transaction_type"] for r in result["purchases"]], ["purchase", "item_removed"])
        self.assertEqual(result["purchases"][1]["removal_reason"], "sale_or_upgrade_unknown")
        self.assertIsNone(result["purchases"][0]["amount_paid"])
        self.assertEqual(result["purchases"][0]["item_cost"], 500)
        self.assertEqual([r["event_type"] for r in result["events"]], ["purchase", "item_removed", "ability_unlock", "ability_upgrade"])
        self.assertEqual(next(r for r in result["player_state"] if r["game_time"] == 20)["inventory"], [1])
        self.assertEqual(next(r for r in result["player_state"] if r["game_time"] == 30)["inventory"], [])

    def test_unknown_definition_has_no_guessed_price_or_type(self):
        result = self.normalize(fixture([player(items=[{"item_id": 1, "game_time_s": 5, "upgrade_id": 1, "sold_time_s": 0}])]))
        self.assertEqual(result["purchases"][0]["transaction_type"], "unclassified_item_record")
        self.assertIsNone(result["purchases"][0]["item_cost"])
        self.assertFalse(result["players"][0]["final_inventory_complete"])

    def test_midboss_claim_team_does_not_invent_claim_time(self):
        result = self.normalize(fixture(mid_boss=[{"destroyed_time_s": 100, "team_killed": 0, "team_claimed": 1}]))
        claim = next(r for r in result["events"] if r["event_type"] == "rejuvenator_claim_team")
        self.assertIsNone(claim["game_time"])
        self.assertEqual(claim["team"], 1)
        self.assertEqual(result["matches"][0]["first_midboss_time"], 100)
        self.assertIsNone(result["matches"][0]["first_rejuvenator_claim_time"])

    def test_paths_percent_health_and_death_sentinel(self):
        data = fixture([player()], duration_s=20, match_paths={"interval_s": 10, "x_resolution": 10,
                        "y_resolution": 10, "version": 1, "paths": [{"player_slot": 10,
                        "x_min": -100, "x_max": 100, "y_min": -50, "y_max": 50,
                        "x_pos": [5, 0, 1], "y_pos": [10, 0, 1], "health": [100, 0, 50]}]})
        result = self.normalize(data)
        rows = result["player_state"]
        self.assertEqual(rows[0]["x"], 0)
        self.assertEqual(rows[0]["y"], 50)
        self.assertEqual(rows[0]["health_percent"], 100)
        self.assertIsNone(rows[0]["health"])
        self.assertFalse(rows[1]["alive"])
        self.assertIsNone(rows[1]["x"])
        self.assertIsNone(rows[0]["net_worth"])

    def test_final_stats_preserved_and_no_imputed_assist_events(self):
        p = player(net_worth=123, kills=0, stats=[{"time_stamp_s": 400, "net_worth": 123,
                   "player_damage": 8, "boss_damage": 0, "shots_hit": 0, "shots_missed": 0,
                   "unrecognized_future_stat": 9}])
        result = self.normalize(fixture([p], new_match_field={"hello": [1, 2]}))
        self.assertEqual(result["players"][0]["unrecognized_future_stat"], 9)
        self.assertEqual(result["players"][0]["objective_damage"], 0)
        self.assertIsNone(result["players"][0]["shot_accuracy"])
        self.assertEqual(result["matches"][0]["new_match_field"], {"hello": [1, 2]})

    def test_spirit_power_alias_keeps_sparse_observation_time(self):
        data = fixture([player(stats=[{"time_stamp_s": 180, "tech_power": 0, "weapon_power": 15},
                                      {"time_stamp_s": 400, "tech_power": 73, "weapon_power": 40}])])
        result = self.normalize(data)
        panel = {r["game_time"]: r for r in result["player_state"]}
        self.assertIsNone(panel[170]["spirit_power"])
        self.assertIsNone(panel[170]["spirit_power_source_game_time"])
        self.assertEqual(panel[350]["spirit_power"], 0)
        self.assertEqual(panel[350]["tech_power"], 0)
        self.assertEqual(panel[350]["spirit_power_source_game_time"], 180)
        self.assertEqual(panel[350]["spirit_power_state_age"], 170)
        self.assertEqual(panel[350]["weapon_power_source_game_time"], 180)
        self.assertEqual(panel[400]["spirit_power"], 73)
        self.assertEqual(result["players"][0]["spirit_power"], 73)
        self.assertEqual(result["metadata_player_stats"][0]["spirit_power"], 0)
        self.assertNotIn("spirit_power", data["match_info"]["players"][0]["stats"][0])

    def test_replay_currencies_and_sparse_power_have_separate_times(self):
        tables = self.normalize(fixture([player(stats=[{"time_stamp_s": 180, "tech_power": 12,
                                                        "weapon_power": 15, "kills": 3}])]))
        metadata = next(r for r in tables["player_state"] if r["game_time"] == 350)
        row = {**metadata, "state_age": 0.01, "source_game_time": 349.99,
               "field_provenance": dict(metadata["field_provenance"])}
        raw = {"source_game_time": 349.99, "raw_controller": {"m_PlayerDataGlobal.m_iDenies": 2},
               "raw_pawn": {"m_nCurrencies.0000": 0, "m_nCurrencies.0001": 3,
                            "m_nSpentCurrencies.0000": 1000, "m_bNetworkDisconnected": False}}
        merge_player_fields(row, metadata, raw)
        self.assertEqual(row["unspent_souls"], 0)
        self.assertEqual(row["spent_souls"], 1000)
        self.assertEqual(row["unspent_ability_points"], 3)
        self.assertEqual(row["unspent_souls_source_game_time"], 349.99)
        self.assertEqual(row["spirit_power"], 12)
        self.assertEqual(row["spirit_power_source_game_time"], 180)
        self.assertEqual(row["spirit_power_state_age"], 170)
        self.assertEqual(row["field_provenance"]["spirit_power"], "metadata_stats:tech_power")
        self.assertEqual(row["denies"], 2)
        self.assertEqual(row["kills"], 3)
        self.assertEqual(row["field_provenance"]["kills"], "metadata_asof")
        self.assertIsNone(row["last_hits"])
        self.assertFalse(row["network_disconnected"])

    def test_missing_replay_currency_is_not_imputed_from_net_worth(self):
        row = {"game_time": 10, "state_age": 0}
        merge_player_fields(row, {}, {"source_game_time": 10, "net_worth": 1000})
        self.assertEqual(row["net_worth"], 1000)
        self.assertIsNone(row["unspent_souls"])
        self.assertIsNone(row["unspent_souls_source_game_time"])

    def test_purchase_balance_is_only_the_last_past_sample(self):
        purchase = {"match_id": 1, "player_id": 10, "game_time": 25, "souls_before": None, "souls_after": None}
        states = [{"match_id": 1, "player_id": 10, "game_time": 20, "source_game_time": 19.99,
                   "net_worth": 2000, "unspent_souls": 500, "unspent_souls_source_game_time": 19.99},
                  {"match_id": 1, "player_id": 10, "game_time": 30, "source_game_time": 30,
                   "net_worth": 2500, "unspent_souls": 0}]
        enrich_purchase_state([purchase], states)
        self.assertEqual(purchase["unspent_souls"], 500)
        self.assertEqual(purchase["unspent_souls_source_time"], 19.99)
        self.assertEqual(purchase["unspent_souls_quality"], "last_observed_before_transaction_not_exact")
        self.assertIsNone(purchase["souls_before"])
        self.assertIsNone(purchase["souls_after"])


if __name__ == "__main__":
    unittest.main()
