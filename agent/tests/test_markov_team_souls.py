import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.analysis.markov import make_transitions, run_analysis
from markov_team_souls import prepare_match, predict_state_distribution


def roster():
    return pd.DataFrame({"player_slot": [0, 1, 2, 3], "team": [0, 0, 1, 1]})


def observations(match_id=1, scale=1):
    rows = []
    for time, totals in zip([0, 20, 40, 60], [(100, 100), (130, 110), (90, 130), (100, 100)]):
        for team in (0, 1):
            source_time = max(0, time - 0.015625)
            rows.append({"match_id": match_id, "team": team, "game_time": time,
                         "team_net_worth": float(totals[team] * scale),
                         "net_worth_advantage": (totals[team] - totals[1 - team]) * scale,
                         "player_count": 2, "expected_players": 2, "net_worth_observed_players": 2,
                         "roster_complete": True, "economy_complete": True,
                         "economy_simultaneous_observations": True,
                         "source_game_time_min": source_time, "source_game_time_max": source_time,
                         "state_age_max": time - source_time,
                         "player_state_sources": '["local_replay:deadem"]'})
    return pd.DataFrame(rows)


class MarkovTeamSoulsTests(unittest.TestCase):
    def test_relative_wealth_is_signed_half_advantage_with_tied_ranks(self):
        panel = prepare_match(observations(), roster())
        paired = panel.groupby("game_time")
        np.testing.assert_allclose(paired.relative_net_worth.sum(), 0)
        np.testing.assert_allclose(panel.net_worth_advantage, 2 * panel.relative_net_worth)
        np.testing.assert_allclose(panel.opponent_net_worth + panel.net_worth_advantage,
                                   panel.team_net_worth)
        at_twenty = panel.loc[panel.game_time == 20].sort_values("team")
        np.testing.assert_allclose(at_twenty.relative_net_worth, [10, -10])
        np.testing.assert_allclose(at_twenty.net_worth_rank, [1, 2])
        np.testing.assert_allclose(at_twenty.net_worth_percentile, [1, 0])
        tied = panel.loc[panel.game_time == 0]
        np.testing.assert_allclose(tied.net_worth_rank, [1.5, 1.5])
        np.testing.assert_allclose(tied.net_worth_percentile, [0.5, 0.5])

    def test_unusable_team_drops_both_perspectives_at_that_time(self):
        defects = [
            {"economy_complete": False},
            {"roster_complete": False},
            {"net_worth_observed_players": 1},
            {"team_net_worth": np.nan},
            {"team_net_worth": -1},
            {"economy_simultaneous_observations": False},
            {"source_game_time_min": 0, "source_game_time_max": 0, "state_age_max": 20},
            {"source_game_time_min": 20.5, "source_game_time_max": 20.5, "state_age_max": -0.5},
            {"source_game_time_min": 19.5, "source_game_time_max": 19.5, "state_age_max": 0.5},
            {"player_state_sources": '["https://example.test/metadata"]'},
            {"player_state_sources": '["local_replay:deadem", "metadata"]'},
            {"player_state_sources": '[]'},
        ]
        for defect in defects:
            with self.subTest(defect=defect):
                source = observations()
                source.loc[(source.team == 0) & (source.game_time == 20), list(defect)] = list(defect.values())
                panel = prepare_match(source, roster())
                self.assertNotIn(20, set(panel.game_time))
                self.assertEqual(len(panel), 6)

    def test_missing_opponent_and_inconsistent_saved_advantage(self):
        source = observations()
        source = source.loc[~((source.team == 1) & (source.game_time == 20))]
        self.assertNotIn(20, set(prepare_match(source, roster()).game_time))
        source = observations()
        source.loc[0, "net_worth_advantage"] = 5
        with self.assertRaisesRegex(ValueError, "advantage disagrees"):
            prepare_match(source, roster())

    def test_duplicate_observations_and_invalid_rosters_are_rejected(self):
        source = observations()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_match(pd.concat([source, source.iloc[:1]]), roster())
        bad_rosters = [roster().iloc[:-1], roster().assign(team=0),
                       roster().assign(player_slot=[0, 0, 2, 3]),
                       roster().assign(team=[0, 0, 1, 2])]
        for bad in bad_rosters:
            with self.subTest(roster=bad.to_dict()), self.assertRaises(ValueError):
                prepare_match(source, bad)

    def test_transitions_do_not_bridge_gaps_teams_or_matches(self):
        rows = []
        for match_id, team, times in ((1, 0, [0, 20, 60, 80]),
                                      (1, 1, [0, 20, 40, 60]),
                                      (2, 0, [0, 20, 40, 60])):
            for time in times:
                rows.append({"match_id": match_id, "team": team, "game_time": time,
                             "state": team + 1, "relative_net_worth": (2 * team - 1) * 100})
        transitions = make_transitions(pd.DataFrame(rows).sample(frac=1, random_state=3), [20, 60], 20,
                                       entity_col="team")
        self.assertEqual(sum(transitions.horizon_seconds == 20), 8)
        self.assertEqual(sum(transitions.horizon_seconds == 60), 2)
        np.testing.assert_array_equal(transitions.state, transitions.next_state)
        gapped = transitions.loc[(transitions.match_id == 1) & (transitions.team == 0)]
        self.assertEqual(set(zip(gapped.game_time, gapped.next_game_time)), {(0, 20), (60, 80)})

    def test_team_forecast_reports_full_opponent_advantage(self):
        model = {"entity_name": "team", "step_seconds": 20,
                 "state_labels": ["Behind", "Ahead"], "time_bin_starts": [0],
                 "probabilities_overall": [[0.75, 0.25], [0.25, 0.75]],
                 "probabilities_by_time_bin": [[[0.75, 0.25], [0.25, 0.75]]],
                 "representatives_by_time_bin": [[-100, 100]]}
        result = predict_state_distribution("Behind", 20, 20, model)
        self.assertEqual(result["probabilities"], {"Behind": 0.75, "Ahead": 0.25})
        self.assertEqual(result["expected_relative_net_worth"], -50)
        self.assertEqual(result["expected_net_worth_advantage"], -100)

    def test_complete_match_holdout_and_training_parameters_ignore_test_wealth(self):
        panel = pd.concat([prepare_match(observations(mid, mid), roster()) for mid in range(1, 5)])
        panel = panel.assign(native_step_seconds=20, match_start_time=0, game_version=1)
        audit = pd.DataFrame({"match_id": [1, 2, 3, 4], "used": True})
        settings = {"n_states": 3, "game_time_bins_minutes": [0, 1], "direct_horizons_seconds": [60],
                    "validation_horizons_seconds": [60], "test_match_fraction": 0.25, "random_seed": 42,
                    "smoothing_alpha": 0.5, "max_state_age_seconds": 1, "n_examples": 1,
                    "example_horizon_seconds": 60, "log_loss_epsilon": 1e-12,
                    "script_name": "markov_team_souls", "entity_col": "team",
                    "net_worth_col": "team_net_worth", "entity_name": "team"}
        with tempfile.TemporaryDirectory() as temporary, patch("agent.analysis.markov.make_figures"), contextlib.redirect_stdout(io.StringIO()):
            settings["output_dir"] = Path(temporary) / "first"
            model = run_analysis(panel, 20, audit, settings)
            saved = pd.read_parquet(settings["output_dir"] / "modeling_dataset.parquet")
            self.assertTrue(saved.groupby("match_id").split.nunique().eq(1).all())
            self.assertTrue(saved.groupby(["match_id", "split"]).team.nunique().eq(2).all())
            self.assertEqual(len(model["test_match_ids"]), 1)
            self.assertFalse(set(model["test_match_ids"]) & set(model["training_match_ids"]))
            self.assertEqual(np.asarray(model["training_counts_overall"]).sum(), 18)
            changed = panel.copy()
            held_out = changed.match_id.isin(model["test_match_ids"])
            changed.loc[held_out, "relative_net_worth"] *= 1000
            settings["output_dir"] = Path(temporary) / "second"
            repeated = run_analysis(changed, 20, audit, settings)
            for key in ("cutpoints", "training_counts_overall", "probabilities_overall",
                        "probabilities_by_time_bin", "representatives_by_time_bin"):
                self.assertEqual(model[key], repeated[key], key)


if __name__ == "__main__":
    unittest.main()
