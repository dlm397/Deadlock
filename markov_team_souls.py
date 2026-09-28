import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agent.analysis.markov import predict_state_distribution as predict_from_model, run_analysis
from dataset import match_ids, read_match
from main import OUTPUT_DIR as DATA_DIR

# Edit these settings, then run this file.
OUTPUT_DIR = Path(DATA_DIR) / "processed/markov_team_souls"
MATCH_IDS = None
N_STATES = 5
BASE_STEP_SECONDS = None  # Infer a common grid without interpolation.
GAME_TIME_BINS_MINUTES = [0, 10, 20, 30]
DIRECT_HORIZONS_SECONDS = [60]
VALIDATION_HORIZONS_SECONDS = [60, 300]
TEST_MATCH_FRACTION = 0.25
RANDOM_SEED = 42
SMOOTHING_ALPHA = 0.5
MAX_STATE_AGE_SECONDS = 1.0
N_EXAMPLES = 3
EXAMPLE_HORIZON_SECONDS = 300
LOG_LOSS_EPSILON = 1e-12


def replay_sources(value):
    try:
        sources = json.loads(value) if isinstance(value, str) else value
        return isinstance(sources, list) and bool(sources) and all(
            isinstance(source, str) and source.startswith("local_replay:") for source in sources)
    except (TypeError, ValueError):
        return False


def prepare_match(state, roster, max_age=MAX_STATE_AGE_SECONDS):
    # Team totals are usable only when both complete rosters were observed together.
    if (roster[["player_slot", "team"]].isna().any().any()
            or roster.player_slot.duplicated().any() or not roster.team.isin([0, 1]).all()):
        raise ValueError("Roster must contain unique player slots assigned to teams 0 and 1")
    counts = roster.groupby("team").size()
    if set(counts.index) != {0, 1} or counts.nunique() != 1:
        raise ValueError("Both teams must have equally sized, nonempty rosters")
    if state.duplicated(["team", "game_time"]).any():
        raise ValueError("Duplicate team/time observations")
    state = state.copy()
    for column in ("game_time", "team_net_worth", "source_game_time_min", "source_game_time_max",
                   "state_age_max", "net_worth_advantage"):
        state[column] = pd.to_numeric(state[column], errors="coerce").astype(float)
    expected = state.team.map(counts)
    age = state.game_time - state.source_game_time_min
    complete = (state.roster_complete.fillna(False) & state.economy_complete.fillna(False)
                & state.economy_simultaneous_observations.fillna(False)
                & state.player_count.eq(expected) & state.expected_players.eq(expected)
                & state.net_worth_observed_players.eq(expected))
    fresh = (age.ge(-1e-6) & age.le(max_age) & state.state_age_max.ge(-1e-6)
             & state.state_age_max.le(max_age)
             & np.isclose(state.source_game_time_min, state.source_game_time_max, rtol=0, atol=1e-6))
    valid = (state.team.isin([0, 1]) & complete & fresh & state.game_time.ge(0)
             & np.isfinite(state.game_time) & np.isfinite(state.team_net_worth)
             & state.team_net_worth.ge(0) & state.player_state_sources.map(replay_sources))
    state = state.loc[valid.fillna(False)].copy()
    grouped = state.groupby("game_time")
    paired = grouped.team.transform("nunique").eq(2)
    simultaneous = np.isclose(grouped.source_game_time_min.transform("min"),
                              grouped.source_game_time_max.transform("max"), rtol=0, atol=1e-6)
    state = state.loc[paired & simultaneous].copy()
    grouped = state.groupby("game_time")
    state["match_mean_team_net_worth"] = grouped.team_net_worth.transform("mean")
    state["relative_net_worth"] = state.team_net_worth - state.match_mean_team_net_worth
    advantage = 2 * state.relative_net_worth
    known = state.net_worth_advantage.notna()
    if not np.isclose(state.loc[known, "net_worth_advantage"], advantage.loc[known], rtol=0, atol=1e-6).all():
        raise ValueError("Saved team advantage disagrees with paired team net-worth totals")
    state["net_worth_advantage"] = advantage
    state["opponent_net_worth"] = state.team_net_worth - advantage
    state["net_worth_rank"] = grouped.team_net_worth.rank(method="average", ascending=False)
    state["net_worth_percentile"] = 2 - state.net_worth_rank
    state["roster_size"] = state.team.map(counts)
    return state.sort_values(["team", "game_time"]).reset_index(drop=True)


def load_observations():
    columns = ["match_id", "team", "game_time", "team_net_worth", "net_worth_advantage",
               "player_count", "expected_players", "net_worth_observed_players", "roster_complete",
               "economy_complete", "economy_simultaneous_observations", "source_game_time_min",
               "source_game_time_max", "state_age_max", "player_state_sources"]
    selected = match_ids(DATA_DIR) if MATCH_IDS is None else list(dict.fromkeys(int(mid) for mid in MATCH_IDS))
    frames, audit = [], []
    for mid in selected:
        entry = {"match_id": mid, "used": False}
        try:
            state = read_match(mid, "team_state", columns=columns, output_dir=DATA_DIR)
            roster = read_match(mid, "players", columns=["player_slot", "team"], output_dir=DATA_DIR)
            match = read_match(mid, "matches", columns=["start_time", "game_version"], output_dir=DATA_DIR).iloc[0]
            entry["input_observations"] = len(state)
            if not state.match_id.eq(mid).fillna(False).all():
                raise ValueError("Team-state match IDs disagree with the selected match")
            state = prepare_match(state, roster, MAX_STATE_AGE_SECONDS)
            times = np.sort(state.game_time.unique())
            if len(times) < 2:
                raise ValueError("Fewer than two paired, fresh replay timestamps")
            native_step = float(pd.Series(np.round(np.diff(times), 6)).mode().iloc[0])
            state["native_step_seconds"] = native_step
            state["match_start_time"] = int(match.start_time)
            state["game_version"] = match.game_version
            entry.update(used=True, native_step_seconds=native_step, paired_replay_observations=len(state))
            frames.append(state)
        except Exception as exc:
            entry["reason"] = str(exc)
        audit.append(entry)
    if not frames:
        raise ValueError("No complete matches with usable paired team net-worth observations")
    panel = pd.concat(frames, ignore_index=True)
    native = sorted(panel.native_step_seconds.unique())
    if BASE_STEP_SECONDS is None:
        if any(not np.isclose(value, round(value)) for value in native):
            raise ValueError("Set BASE_STEP_SECONDS explicitly for fractional source intervals")
        step = math.lcm(*(int(round(value)) for value in native))
    else:
        step = float(BASE_STEP_SECONDS)
    if not np.isfinite(step) or step <= 0 or any(not np.isclose(step / value, round(step / value)) for value in native):
        raise ValueError(f"BASE_STEP_SECONDS must be a common multiple of source intervals {native}")
    panel = panel.loc[np.isclose(np.mod(panel.game_time, step), 0)].copy()
    if panel.empty:
        raise ValueError("No observations lie on the selected analysis grid")
    panel = panel.sort_values(["match_id", "team", "game_time"]).reset_index(drop=True)
    grid_counts = panel.groupby("match_id").size()
    for entry in audit:
        entry["model_observations"] = int(grid_counts.get(entry["match_id"], 0))
        if entry["used"] and not entry["model_observations"]:
            entry.update(used=False, reason="No observations on the selected analysis grid")
    return panel, step, pd.DataFrame(audit)


def predict_state_distribution(current_state, current_game_time, prediction_horizon, model=None):
    # Times are seconds; the returned advantage is over the opposing team.
    if model is None:
        model = json.loads((OUTPUT_DIR / "model.json").read_text(encoding="utf-8"))
    return predict_from_model(current_state, current_game_time, prediction_horizon, model)


def main():
    panel, step, audit = load_observations()
    return run_analysis(panel, step, audit, {
        "output_dir": OUTPUT_DIR, "n_states": N_STATES,
        "game_time_bins_minutes": GAME_TIME_BINS_MINUTES,
        "direct_horizons_seconds": DIRECT_HORIZONS_SECONDS,
        "validation_horizons_seconds": VALIDATION_HORIZONS_SECONDS,
        "test_match_fraction": TEST_MATCH_FRACTION, "random_seed": RANDOM_SEED,
        "smoothing_alpha": SMOOTHING_ALPHA, "max_state_age_seconds": MAX_STATE_AGE_SECONDS,
        "n_examples": N_EXAMPLES, "example_horizon_seconds": EXAMPLE_HORIZON_SECONDS,
        "log_loss_epsilon": LOG_LOSS_EPSILON, "script_name": "markov_team_souls",
        "entity_col": "team", "net_worth_col": "team_net_worth", "entity_name": "team",
    })


if __name__ == "__main__":
    main()
