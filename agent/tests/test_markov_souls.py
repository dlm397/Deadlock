import unittest

import numpy as np
import pandas as pd

from markov_souls import (
    assign_states,
    empirical_probabilities,
    fit_boundaries,
    forecast_probabilities,
    make_transitions,
    transition_counts,
)


class MarkovSoulsTests(unittest.TestCase):
    def test_quantile_ties_do_not_create_duplicate_or_empty_edge_states(self):
        values = np.array([0] * 12 + [10] * 6 + [20] * 2, dtype=float)
        cuts = fit_boundaries(values, 5)
        self.assertTrue(np.all(np.diff(cuts) > 0))
        self.assertTrue(np.all((cuts > values.min()) & (cuts < values.max())))
        self.assertLessEqual(len(cuts), 4)
        constant_cuts = fit_boundaries(np.full(20, 7.0), 5)
        self.assertEqual(len(constant_cuts), 0)
        np.testing.assert_array_equal(assign_states([7.0, 7.0], constant_cuts), [1, 1])

    def test_boundaries_have_deterministic_right_hand_membership(self):
        cuts = np.array([-100.0, 100.0])
        actual = assign_states([-1000, -100, 0, 100, 1000], cuts)
        np.testing.assert_array_equal(actual, [1, 2, 2, 3, 3])

    def test_transitions_do_not_bridge_gaps_players_or_matches(self):
        rows = []
        for match_id, slot, times in ((1, 0, [0, 20, 60, 80]),
                                      (1, 1, [0, 20, 40, 60]),
                                      (2, 0, [0, 20, 40, 60])):
            for time in times:
                rows.append({"match_id": match_id, "player_slot": slot,
                             "game_time": time, "state": 1 + int(time >= 40),
                             "relative_net_worth": float(time)})
        panel = pd.DataFrame(rows).sample(frac=1, random_state=13)
        transitions = make_transitions(panel, [20, 60], 20)
        self.assertEqual(len(transitions.loc[transitions.horizon_seconds == 20]), 8)
        self.assertEqual(len(transitions.loc[transitions.horizon_seconds == 60]), 2)
        gapped = transitions.loc[(transitions.match_id == 1) & (transitions.player_slot == 0)]
        self.assertEqual(set(zip(gapped.game_time, gapped.next_game_time)), {(0, 20), (60, 80)})
        self.assertTrue((transitions.next_game_time - transitions.game_time
                         == transitions.horizon_seconds).all())
        self.assertEqual(len(transitions), len(transitions.drop_duplicates(
            ["match_id", "player_slot", "game_time", "horizon_seconds"])))

    def test_counts_keep_current_state_in_rows_and_next_state_in_columns(self):
        transitions = pd.DataFrame({"state": [1, 1, 1, 2], "next_state": [1, 2, 2, 1]})
        np.testing.assert_array_equal(transition_counts(transitions, 2), [[1, 2], [1, 0]])

    def test_empirical_empty_rows_are_unknown_not_zero_probabilities(self):
        probabilities = empirical_probabilities(np.array([[2, 1], [0, 0]]))
        np.testing.assert_allclose(probabilities[0], [2 / 3, 1 / 3])
        self.assertTrue(np.isnan(probabilities[1]).all())

    def model(self):
        return {"step_seconds": 20, "state_labels": ["Behind", "Ahead"],
                "time_bin_starts": [0, 600],
                "probabilities_overall": [[0.9, 0.1], [0.1, 0.9]],
                "probabilities_by_time_bin": [[[0.75, 0.25], [0.0, 1.0]],
                                              [[0.2, 0.8], [0.6, 0.4]]]}

    def test_prediction_changes_stage_matrix_at_boundary(self):
        # The 580-to-600 step uses the early matrix; the next uses the late one.
        actual = forecast_probabilities(self.model(), 1, 580, 40)
        np.testing.assert_allclose(actual, [0.3, 0.7])
        np.testing.assert_allclose(actual.sum(), 1.0)
        at_boundary = forecast_probabilities(self.model(), 1, 600, 20)
        np.testing.assert_allclose(at_boundary, [0.2, 0.8])

    def test_overall_prediction_and_zero_horizon(self):
        overall = forecast_probabilities(self.model(), 1, 580, 40, use_time_bins=False)
        np.testing.assert_allclose(overall, [0.82, 0.18])
        unchanged = forecast_probabilities(self.model(), 2, 580, 0)
        np.testing.assert_array_equal(unchanged, [0.0, 1.0])

    def test_prediction_rejects_negative_and_partial_step_horizons(self):
        for horizon in (-20, 30):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                forecast_probabilities(self.model(), 1, 580, horizon)

    def test_prediction_rejects_invalid_state_ids(self):
        for state in (0, 1.5, 3, np.nan, np.inf):
            with self.subTest(state=state), self.assertRaises(ValueError):
                forecast_probabilities(self.model(), state, 580, 20)


if __name__ == "__main__":
    unittest.main()
