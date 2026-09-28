import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import match_ids, read_match
from main import OUTPUT_DIR as DATA_DIR
from agent.analysis.markov import (
    assign_states, empirical_probabilities, fit_boundaries, forecast_probabilities,
    make_transitions, transition_counts, run_analysis,
    predict_state_distribution as _predict_state_distribution,
)

# Edit these settings, then run this file.
OUTPUT_DIR = Path(DATA_DIR) / "processed/markov_souls"
MATCH_IDS = None
N_STATES = 5
BASE_STEP_SECONDS = None  # Infer a common grid; never interpolate finer observations.
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


def load_observations():
    columns = ["match_id", "player_slot", "player_id", "hero_id", "team", "game_time", "net_worth",
               "state_source", "source_game_time", "field_provenance"]
    available = match_ids(DATA_DIR)
    selected = available if MATCH_IDS is None else list(dict.fromkeys(MATCH_IDS))
    frames, audit = [], []
    for mid in selected:
        entry = {"match_id": mid, "used": False}
        try:
            state = read_match(mid, "player_state", columns=columns, output_dir=DATA_DIR)
            roster = read_match(mid, "players", columns=["player_slot"], output_dir=DATA_DIR)
            match = read_match(mid, "matches", columns=["start_time", "game_version"], output_dir=DATA_DIR).iloc[0]
            entry["input_observations"] = len(state)
            slots = set(roster.player_slot.dropna().astype(int))
            if len(slots) != len(roster) or len(slots) < 2:
                raise ValueError("Roster does not have unique, nonmissing player slots")
            if state.duplicated(["player_slot", "game_time"]).any():
                raise ValueError("Duplicate player/time observations")
            for col in ("game_time", "net_worth", "source_game_time"):
                state[col] = pd.to_numeric(state[col], errors="coerce").astype(float)
            provenance = state.field_provenance.map(lambda x: json.loads(x).get("net_worth", "") if pd.notna(x) else "")
            age = state.game_time - state.source_game_time
            valid = (state.player_slot.isin(slots) & state.state_source.eq("replay").fillna(False)
                     & provenance.str.startswith("replay") & np.isfinite(state.net_worth)
                     & state.net_worth.ge(0) & state.game_time.ge(0)
                     & age.ge(-1e-6) & age.le(MAX_STATE_AGE_SECONDS))
            state = state.loc[valid].drop(columns="field_provenance").copy()
            entry["quality_observations"] = len(state)
            full = state.groupby("game_time").player_slot.nunique().eq(len(slots))
            state = state.loc[state.game_time.isin(full.index[full])].copy()
            times = np.sort(state.game_time.unique())
            if len(times) < 2:
                raise ValueError("Fewer than two complete-roster, fresh replay timestamps")
            differences = pd.Series(np.round(np.diff(times), 6))
            native_step = float(differences.mode().iloc[0])
            state["native_step_seconds"] = native_step
            state["roster_size"] = len(slots)
            state["match_start_time"] = int(match.start_time)
            state["game_version"] = match.game_version
            entry.update(used=True, native_step_seconds=native_step, complete_roster_observations=len(state))
            frames.append(state)
        except Exception as exc:
            entry["reason"] = str(exc)
        audit.append(entry)
    if not frames:
        raise ValueError("No complete matches with usable replay net-worth observations")
    frame = pd.concat(frames, ignore_index=True)
    native = sorted(frame.native_step_seconds.unique())
    if BASE_STEP_SECONDS is None:
        if any(not np.isclose(value, round(value)) for value in native):
            raise ValueError("Set BASE_STEP_SECONDS explicitly for fractional-second source intervals")
        step = math.lcm(*(int(round(value)) for value in native))
    else:
        step = float(BASE_STEP_SECONDS)
    if step <= 0 or any(not np.isclose(step / value, round(step / value)) for value in native):
        raise ValueError(f"BASE_STEP_SECONDS must be a common multiple of source intervals {native}")
    frame = frame.loc[np.isclose(np.mod(frame.game_time, step), 0)].copy()
    group = frame.groupby(["match_id", "game_time"])
    frame["match_mean_net_worth"] = group.net_worth.transform("mean")
    frame["relative_net_worth"] = frame.net_worth - frame.match_mean_net_worth
    frame["net_worth_rank"] = group.net_worth.rank(method="average", ascending=False)
    frame["net_worth_percentile"] = (frame.roster_size - frame.net_worth_rank) / (frame.roster_size - 1)
    frame = frame.sort_values(["match_id", "player_slot", "game_time"]).reset_index(drop=True)
    grid_counts = frame.groupby("match_id").size()
    for entry in audit:
        entry["model_observations"] = int(grid_counts.get(entry["match_id"], 0))
    return frame, step, pd.DataFrame(audit)


def predict_state_distribution(current_state, current_game_time, prediction_horizon, model=None):
    # Times and horizons are seconds.
    if model is None:
        model = json.loads((OUTPUT_DIR / "model.json").read_text(encoding="utf-8"))
    return _predict_state_distribution(current_state, current_game_time, prediction_horizon, model)


def main():
    panel, step, audit = load_observations()
    settings = {
        "output_dir": OUTPUT_DIR, "n_states": N_STATES,
        "game_time_bins_minutes": GAME_TIME_BINS_MINUTES,
        "direct_horizons_seconds": DIRECT_HORIZONS_SECONDS,
        "validation_horizons_seconds": VALIDATION_HORIZONS_SECONDS,
        "test_match_fraction": TEST_MATCH_FRACTION, "random_seed": RANDOM_SEED,
        "smoothing_alpha": SMOOTHING_ALPHA, "max_state_age_seconds": MAX_STATE_AGE_SECONDS,
        "n_examples": N_EXAMPLES, "example_horizon_seconds": EXAMPLE_HORIZON_SECONDS,
        "log_loss_epsilon": LOG_LOSS_EPSILON, "script_name": "markov_souls",
        "entity_col": "player_slot", "net_worth_col": "net_worth", "entity_name": "player",
    }
    return run_analysis(panel, step, audit, settings)


if __name__ == "__main__":
    main()
