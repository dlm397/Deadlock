import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datapipe.dataset import match_ids, read_match
from main import OUTPUT_DIR

# Settings: times and horizons are seconds; bins below are minutes.
DATA_DIR = OUTPUT_DIR  # Override here to analyze a different dataset.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "data/processed"
MATCH_IDS = None
LEVEL = "player"  # "player" or "team"
SAMPLE_INTERVAL = 20
N_STATES = 5
GAME_TIME_BINS = [0, 10, 20, 30]
HORIZONS = [60, 300]
TEST_FRACTION = 0.25
RANDOM_SEED = 42
SMOOTHING = 0.5
MAX_STATE_AGE = 1.0
N_EXAMPLES = 3


def load_observations(data_dir=DATA_DIR, ids=None, level=LEVEL, interval=SAMPLE_INTERVAL):
    if level not in ("player", "team") or interval <= 0:
        raise ValueError("Use player/team and a positive sampling interval")
    columns = ["match_id", "player_slot", "player_id", "hero_id", "team", "game_time",
               "net_worth", "state_source", "source_game_time", "field_provenance"]
    frames, audit = [], []
    for mid in match_ids(data_dir) if ids is None else dict.fromkeys(ids):
        try:
            state = read_match(mid, "player_state", columns=columns, output_dir=data_dir)
            roster = read_match(mid, "players", output_dir=data_dir)
            match = read_match(mid, "matches", output_dir=data_dir).iloc[0]
            if (roster.player_slot.isna().any() or roster.player_slot.duplicated().any()
                    or not roster.team.isin([0, 1]).all()):
                raise ValueError("Roster needs unique slots and teams 0/1")
            sizes = roster.groupby("team").size()
            if len(sizes) != 2 or sizes.nunique() != 1:
                raise ValueError("Two equally sized teams are required")
            if state.duplicated(["player_slot", "game_time"]).any() or not state.match_id.eq(mid).fillna(False).all():
                raise ValueError("Duplicate player/time rows or mismatched match IDs")
            known = state.player_slot.isin(roster.player_slot)
            expected = state.player_slot.map(roster.set_index("player_slot").team)
            if not state.loc[known, "team"].eq(expected[known]).fillna(False).all():
                raise ValueError("State teams disagree with the roster")
            for column in ("game_time", "source_game_time", "net_worth"):
                state[column] = pd.to_numeric(state[column], errors="coerce").astype(float)
            source = state.field_provenance.map(lambda x: json.loads(x).get("net_worth", "") if pd.notna(x) else "")
            age = state.game_time - state.source_game_time
            valid = (known & state.state_source.eq("replay").fillna(False) & source.str.startswith("replay").fillna(False)
                     & age.between(-1e-6, MAX_STATE_AGE) & state.game_time.ge(0)
                     & np.isfinite(state.net_worth) & state.net_worth.ge(0))
            state = state.loc[valid.fillna(False)].drop(columns="field_provenance").copy()
            full = state.groupby("game_time").player_slot.transform("nunique").eq(len(roster))
            state = state.loc[full].copy()
            if level == "team":
                times = state.groupby("game_time").source_game_time
                state = state.loc[np.isclose(times.transform("min"), times.transform("max"), atol=1e-6, rtol=0)]
                state = state.groupby(["match_id", "team", "game_time"], as_index=False).agg(
                    net_worth=("net_worth", "sum"), source_game_time=("source_game_time", "min"))
                state["team_net_worth"] = state.net_worth
            times = np.sort(state.game_time.unique())
            if len(times) < 2:
                raise ValueError("Too few complete, fresh replay timestamps")
            native = float(pd.Series(np.round(np.diff(times), 6)).mode().iloc[0])
            if not np.isclose(interval / native, round(interval / native)):
                raise ValueError(f"Analysis interval {interval}s is not a multiple of the native {native}s grid")
            state = state.loc[np.isclose(state.game_time % interval, 0)].copy()
            state["game_version"], state["match_start_time"] = match.game_version, match.start_time
            group = state.groupby(["match_id", "game_time"]).net_worth
            state["relative_net_worth"] = state.net_worth - group.transform("mean")
            state["net_worth_rank"] = group.rank(ascending=False, method="average")
            state["net_worth_percentile"] = (group.transform("size") - state.net_worth_rank) / (group.transform("size") - 1)
            if level == "team":
                state["net_worth_advantage"] = 2 * state.relative_net_worth
            frames.append(state)
            audit.append({"match_id": int(mid), "observations": len(state), "players": len(roster), "error": ""})
        except Exception as error:
            print(f"Analysis match {mid}: skipped ({error})")
            audit.append({"match_id": int(mid), "observations": 0, "players": 0, "error": str(error)})
    if not frames:
        raise ValueError("No usable replay observations")
    entity = "team" if level == "team" else "player_slot"
    return pd.concat(frames, ignore_index=True).sort_values(["match_id", entity, "game_time"]), pd.DataFrame(audit)


def make_transitions(panel, horizons, interval, entity):
    frame = panel.reset_index(drop=True)
    gap = frame.groupby(["match_id", entity]).game_time.diff()
    sequences = (~np.isclose(gap, interval)).cumsum()
    grouped = frame.groupby(sequences)[["game_time", "state", "relative_net_worth"]]
    results = []
    for horizon in horizons:
        if horizon <= 0 or not np.isclose(horizon / interval, round(horizon / interval)):
            raise ValueError("Horizons must be positive multiples of the sampling interval")
        future = grouped.shift(-int(round(horizon / interval)))
        valid = np.isclose(future.game_time - frame.game_time, horizon)
        rows = frame.loc[valid, ["match_id", entity, "game_time", "state", "time_bin", "split"]].copy()
        rows["next_game_time"] = future.loc[valid, "game_time"]
        rows["next_state"] = future.loc[valid, "state"].astype(int)
        rows["next_relative_net_worth"] = future.loc[valid, "relative_net_worth"]
        rows["horizon_seconds"] = horizon
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def transition_counts(rows, number):
    counts = np.zeros((number, number), dtype=np.int64)
    np.add.at(counts, (rows.state.to_numpy(dtype=int) - 1, rows.next_state.to_numpy(dtype=int) - 1), 1)
    return counts


def probabilities(counts, alpha=0, fallback=None):
    totals = counts.sum(axis=1, keepdims=True)
    result = np.divide(counts + alpha, totals + alpha * len(counts),
                       out=np.full(counts.shape, np.nan), where=(totals + alpha * len(counts)) > 0)
    if fallback is not None:
        result[totals[:, 0] == 0] = fallback[totals[:, 0] == 0]
    return result


def forecast(model, current_state, current_time, horizon, by_stage=True):
    step = model["step_seconds"]
    if (not np.isfinite(current_time) or current_time < 0 or not np.isfinite(horizon)
            or horizon < 0 or not np.isclose(horizon / step, round(horizon / step))):
        raise ValueError("Use nonnegative seconds and a horizon divisible by the model step")
    number = len(model["state_labels"])
    if current_state != int(current_state) or not 1 <= current_state <= number:
        raise ValueError("State IDs start at 1")
    distribution = np.eye(number)[int(current_state) - 1]
    for offset in range(int(round(horizon / step))):
        stage = np.searchsorted(model["time_bin_starts"], current_time + offset * step, side="right") - 1
        matrix = model["probabilities_by_time_bin"][stage] if by_stage else model["probabilities_overall"]
        distribution = distribution @ matrix
    return distribution


def predict_state_distribution(current_state, current_game_time, prediction_horizon, model=None):
    if model is None:
        model = json.loads((RESULTS_DIR / f"markov_{LEVEL}_souls/model.json").read_text(encoding="utf-8"))
    if isinstance(current_state, str):
        current_state = model["state_labels"].index(current_state) + 1
    distribution = forecast(model, current_state, current_game_time, prediction_horizon)
    stage = np.searchsorted(model["time_bin_starts"], current_game_time + prediction_horizon, side="right") - 1
    result = {"probabilities": dict(zip(model["state_labels"], distribution.tolist())),
              "expected_relative_net_worth": float(distribution @ model["representatives_by_time_bin"][stage])}
    if model["level"] == "team":
        result["expected_net_worth_advantage"] = 2 * result["expected_relative_net_worth"]
    return result


def evaluate(model, transitions, output):
    scores, predictions = [], []
    number = len(model["state_labels"])
    for horizon, rows in transitions.loc[transitions.split == "test"].groupby("horizon_seconds"):
        truth = rows.next_state.to_numpy(dtype=int) - 1
        for name in ("time_binned_markov", "overall_markov", "persistence"):
            cache = {}
            vectors = []
            for row in rows.itertuples():
                key = (row.state, row.game_time)
                if key not in cache:
                    cache[key] = (np.eye(number)[row.state - 1] if name == "persistence"
                                  else forecast(model, row.state, row.game_time, horizon, name == "time_binned_markov"))
                vectors.append(cache[key])
            vectors = np.asarray(vectors)
            scored = rows[["match_id"]].copy()
            scored["accuracy"] = vectors.argmax(axis=1) == truth
            scored["log_loss"] = -np.log(np.clip(vectors[np.arange(len(rows)), truth], 1e-12, 1))
            per_match = scored.groupby("match_id").agg(accuracy=("accuracy", "mean"), log_loss=("log_loss", "mean"), transitions=("accuracy", "size")).reset_index()
            per_match["model"], per_match["horizon_seconds"] = name, horizon
            scores.append(per_match)
            if name == "time_binned_markov":
                predicted = rows.reset_index(drop=True).copy()
                for index in range(number):
                    predicted[f"probability_state_{index + 1}"] = vectors[:, index]
                stages = np.searchsorted(model["time_bin_starts"], rows.next_game_time, side="right") - 1
                predicted["expected_relative_net_worth"] = (vectors * np.asarray(model["representatives_by_time_bin"])[stages]).sum(axis=1)
                predictions.append(predicted)
    by_match = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(columns=["match_id", "model", "horizon_seconds", "accuracy", "log_loss", "transitions"])
    by_match.to_csv(output / "validation_by_match.csv", index=False)
    metrics = []
    for (name, horizon), rows in by_match.groupby(["model", "horizon_seconds"]):
        metrics.append({"model": name, "horizon_seconds": horizon, "matches": len(rows), "transitions": rows.transitions.sum(),
                        "accuracy": np.average(rows.accuracy, weights=rows.transitions), "log_loss": np.average(rows.log_loss, weights=rows.transitions),
                        "match_macro_accuracy": rows.accuracy.mean(), "match_macro_log_loss": rows.log_loss.mean()})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(output / "heldout_predictions.parquet", index=False, compression="zstd", compression_level=19)
    return metrics


def figures(panel, transitions, model, matrices, output):
    labels, starts, step = model["state_labels"], model["time_bin_starts"], model["step_seconds"]
    colors = plt.get_cmap("RdYlBu")(np.linspace(0.08, 0.92, len(labels)))
    for horizon, collection in matrices.items():
        for name, selected in (("overall", collection[:1]), ("stages", collection[1:])):
            fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 5), squeeze=False, layout="constrained")
            for index, (ax, matrix) in enumerate(zip(axes.flat, selected)):
                ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues")
                ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                       title=f"{horizon:g}s: " + ("overall" if name == "overall" else f"from {starts[index] / 60:g} min"), xlabel="Next state", ylabel="Current state")
                ax.tick_params(axis="x", rotation=35)
                for i, j in np.ndindex(matrix.shape):
                    ax.text(j, i, f"{matrix[i, j]:.2f}" if np.isfinite(matrix[i, j]) else "-", ha="center", va="center", color="white" if matrix[i, j] > .55 else "black")
            fig.savefig(output / f"transition_{name}_{horizon:g}s.png", dpi=130)
            plt.close(fig)
    entity = "team" if model["level"] == "team" else "player_slot"
    eligible = transitions.loc[transitions.horizon_seconds == transitions.horizon_seconds.max()]
    heldout = eligible.loc[eligible.split == "test"]
    eligible = heldout if len(heldout) else eligible
    for index, (_, rows) in enumerate(eligible.groupby("match_id")):
        if index >= N_EXAMPLES:
            break
        example = rows.iloc[(rows.game_time - 720).abs().argmin()]
        track = panel.loc[(panel.match_id == example.match_id) & panel[entity].eq(example[entity])]
        clock = np.arange(example.game_time, example.next_game_time + step / 2, step)
        actual = track.set_index("game_time").loc[clock]
        predicted = np.asarray([forecast(model, int(example.state), example.game_time, t - example.game_time) for t in clock])
        stages = np.searchsorted(starts, clock, side="right") - 1
        wealth = (predicted * np.asarray(model["representatives_by_time_bin"])[stages]).sum(axis=1)
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
        axes[0].plot(track.game_time / 60, track.net_worth, color="gray", linewidth=.7)
        for state, label in enumerate(labels, 1):
            selected = track.loc[track.state == state]
            axes[0].scatter(selected.game_time / 60, selected.net_worth, s=12, color=colors[state - 1], label=label)
        axes[0].set(ylabel="Net worth", title=f"Match {int(example.match_id)}, {model['level']} {int(example[entity])}; {example.split}")
        axes[0].legend(ncol=min(5, len(labels)), fontsize=8)
        axes[1].stackplot(clock / 60, predicted.T, colors=colors, alpha=.8)
        axes[1].set(ylabel="Predicted probability", ylim=(0, 1))
        actual_axis = axes[1].twinx()
        actual_axis.step(clock / 60, actual.state, color="black", where="post")
        actual_axis.set(ylabel="Actual state (black)", yticks=range(1, len(labels) + 1), ylim=(.5, len(labels) + .5))
        axes[2].plot(clock / 60, actual.relative_net_worth, color="black", label="Actual")
        axes[2].plot(clock / 60, wealth, "--", label="Predicted state-mean proxy")
        axes[2].set(ylabel="Relative net worth", xlabel="Game time (minutes)")
        axes[2].legend()
        fig.savefig(output / f"trajectory_{index + 1}.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.bar(labels, model["representatives_overall"], color=colors)
    ax.set(ylabel="Mean relative net worth", title="State representatives: training matches")
    fig.savefig(output / "state_means.png", dpi=140)
    plt.close(fig)


def run(data_dir=DATA_DIR, output_dir=None, ids=None, level=LEVEL, interval=SAMPLE_INTERVAL,
        n_states=N_STATES, bins=GAME_TIME_BINS, horizons=HORIZONS, test_fraction=TEST_FRACTION,
        seed=RANDOM_SEED, smoothing=SMOOTHING):
    if n_states < 1 or not 0 < test_fraction < 1 or smoothing < 0:
        raise ValueError("Use positive state count, test fraction between 0/1, and nonnegative smoothing")
    starts = np.asarray(bins, dtype=float) * 60
    if not len(starts) or starts[0] != 0 or not np.all(np.diff(starts) > 0):
        raise ValueError("Time bins must start at zero and increase")
    output = Path(output_dir) if output_dir is not None else RESULTS_DIR / f"markov_{level}_souls"
    output.mkdir(parents=True, exist_ok=True)
    panel, audit = load_observations(data_dir, ids, level, interval)
    entity = "team" if level == "team" else "player_slot"
    ids = np.sort(panel.match_id.unique().astype(int))
    count = min(len(ids) - 2, max(1, math.ceil(len(ids) * test_fraction))) if len(ids) >= 3 else 0
    test_ids = np.random.default_rng(seed).permutation(ids)[:count]
    panel["split"] = np.where(panel.match_id.isin(test_ids), "test", "train")
    train_values = panel.loc[panel.split == "train", "relative_net_worth"]
    cuts = np.unique(np.quantile(train_values, np.arange(1, n_states) / n_states))
    cuts = cuts[(cuts > train_values.min()) & (cuts < train_values.max())]
    number = len(cuts) + 1
    labels = ["Far Behind", "Behind", "Average", "Ahead", "Far Ahead"] if number == 5 else [f"State {i + 1}" for i in range(number)]
    panel["state"] = np.searchsorted(cuts, panel.relative_net_worth, side="right") + 1
    panel["state_label"] = panel.state.map(dict(enumerate(labels, 1)))
    panel["time_bin"] = np.searchsorted(starts, panel.game_time, side="right") - 1
    requested = sorted(set([interval, *horizons]))
    transitions = make_transitions(panel, requested, interval, entity)
    base = transitions.loc[transitions.horizon_seconds == interval]
    train = panel.loc[panel.split == "train"]
    training_counts = transition_counts(base.loc[base.split == "train"], number)
    overall = probabilities(training_counts, smoothing, np.full((number, number), 1 / number))
    means = train.groupby("state").relative_net_worth.mean().reindex(range(1, number + 1)).fillna(train.relative_net_worth.mean()).to_numpy()
    model = {"level": level, "step_seconds": interval, "state_labels": labels, "cutpoints": cuts.tolist(),
             "time_bin_starts": starts.tolist(), "smoothing_alpha": smoothing,
             "training_match_ids": sorted(set(ids.tolist()) - set(test_ids.tolist())), "test_match_ids": test_ids.tolist(),
             "probabilities_overall": overall.tolist(), "representatives_overall": means.tolist(),
             "probabilities_by_time_bin": [], "representatives_by_time_bin": []}
    for stage in range(len(starts)):
        rows = base.loc[(base.split == "train") & (base.time_bin == stage)]
        model["probabilities_by_time_bin"].append(probabilities(transition_counts(rows, number), smoothing, overall).tolist())
        values = train.loc[train.time_bin == stage].groupby("state").relative_net_worth.mean()
        model["representatives_by_time_bin"].append([float(values.get(i + 1, means[i])) for i in range(number)])
    (output / "model.json").write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
    next_values = base.drop(columns=["state", "time_bin", "split", "horizon_seconds"])
    modeling = panel.merge(next_values, on=["match_id", entity, "game_time"], how="left", validate="one_to_one")
    modeling["next_state"] = modeling.next_state.astype("Int64")
    modeling.to_parquet(output / "modeling_dataset.parquet", index=False, compression="zstd", compression_level=19)
    transitions.to_parquet(output / "transitions.parquet", index=False, compression="zstd", compression_level=19)
    audit.to_csv(output / "input_audit.csv", index=False)
    definitions = pd.DataFrame({"state": range(1, number + 1), "label": labels, "lower_bound": np.r_[-np.inf, cuts], "upper_bound": np.r_[cuts, np.inf], "relative_net_worth_mean": means})
    definitions.to_csv(output / "state_definitions.csv", index=False)
    matrices, stage_differences = {}, []
    for horizon in requested:
        rows = transitions.loc[transitions.horizon_seconds == horizon]
        matrices[horizon] = []
        suffix = "" if horizon == interval else f"_{horizon:g}s"
        for stage in [None, *range(len(starts))]:
            counts = transition_counts(rows if stage is None else rows.loc[rows.time_bin == stage], number)
            matrix = probabilities(counts)
            matrices[horizon].append(matrix)
            stage_suffix = "" if stage is None else f"_from_{starts[stage] / 60:g}min"
            for name, values in (("matrix", matrix), ("counts", counts)):
                pd.DataFrame(values, index=labels, columns=labels).rename_axis("current_state").to_csv(output / f"transition_{name}{suffix}{stage_suffix}.csv")
            if stage is not None and counts.sum():
                distance = .5 * np.nansum(np.abs(matrix - matrices[horizon][0]), axis=1)
                stage_differences.append({"horizon_seconds": horizon, "from_minute": starts[stage] / 60,
                                          "transitions": int(counts.sum()), "weighted_total_variation": np.average(distance, weights=counts.sum(axis=1))})
    pd.DataFrame(stage_differences).to_csv(output / "stage_comparison.csv", index=False)
    metrics = evaluate(model, transitions, output)
    figures(panel, transitions, model, matrices, output)
    persistence = sorted(zip(labels, np.diag(matrices[interval][0])), key=lambda x: -np.nan_to_num(x[1], nan=-1))
    report = [f"Exploratory {level} Markov model: {len(ids)} matches ({len(ids) - count} training, {count} held out).",
              f"{int(audit.players.sum())} player-match appearances; {panel.groupby(['match_id', entity]).ngroups} {level} trajectories; {len(panel)} observations; {len(base)} {interval:g}s transitions.",
              "State boundaries and representatives use training matches only; intervals are [lower, upper).",
              definitions.to_string(index=False), "All-match empirical transition matrix:",
              pd.DataFrame(matrices[interval][0], index=labels, columns=labels).to_string(float_format=lambda x: f"{x:.3f}"),
              "Persistence: " + "; ".join(f"{name} {value:.1%}" for name, value in persistence),
              "Game-stage differences (total variation vs. overall; descriptive, not a significance test):",
              pd.DataFrame(stage_differences).to_string(index=False), "Preliminary held-out performance:",
              metrics.to_string(index=False) if len(metrics) else "Too few matches or eligible future observations for validation.",
              "Repeated players, teams and overlapping transitions are dependent. Match-macro scores weight matches equally; no independent-row uncertainty estimates are reported.",
              "Fresh replay net worth requires a complete roster; observations are never interpolated and transitions never cross missing times.",
              "Descriptive matrices use all matches. Forecast matrices use training matches with additive smoothing; empty stage rows use the overall training row.",
              "Forecasts condition on match continuation and observed future coverage. Expected wealth is a training state-mean proxy, not a prediction interval.",
              "Pooled states weight longer games more heavily; fixed thresholds may create stage differences. A fit does not establish the Markov property."]
    if level == "team":
        report.append("Team relative wealth is half the lead over the opponent. The two team perspectives are dependent; forecasts are marginal, not joint.")
    (output / "summary.txt").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print("\n\n".join(report[:14]))
    print(f"Outputs: {output.resolve()}")
    return model


def describe_match(match_id=None, data_dir=DATA_DIR, output_dir=None):
    mid = match_ids(data_dir)[0] if match_id is None else match_id
    output = Path(output_dir) if output_dir is not None else RESULTS_DIR / "plots"
    output.mkdir(parents=True, exist_ok=True)
    state, teams, purchases, events = [read_match(mid, name, output_dir=data_dir) for name in ("player_state", "team_state", "purchases", "events")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), layout="constrained")
    for slot, rows in state.groupby("player_slot"):
        axes[0].plot(rows.game_time / 60, rows.net_worth, label=f"Slot {slot}", linewidth=1)
    axes[0].set(title=f"Match {mid}: player net worth", ylabel="Net worth")
    axes[0].legend(ncol=6, fontsize=8)
    totals = teams.pivot(index="game_time", columns="team", values="team_net_worth").sort_index()
    totals.set_axis(totals.index / 60).plot(ax=axes[1])
    axes[1].set(ylabel="Team net worth")
    axes[2].plot(totals.index / 60, totals[0] - totals[1])
    axes[2].axhline(0, color="gray", linewidth=.5)
    axes[2].set(xlabel="Game time (minutes)", ylabel="Team 0 minus team 1")
    fig.savefig(output / f"{mid}_economy.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout="constrained")
    if len(purchases):
        for kind, rows in purchases.dropna(subset=["game_time", "player_slot"]).groupby("transaction_type", dropna=False):
            axes[0].scatter(rows.game_time / 60, rows.player_slot, label=str(kind), s=10)
        axes[0].legend(fontsize=8)
    axes[0].set(title="Recorded transactions", ylabel="Player slot")
    if len(events):
        text = events.event_type.astype("string").fillna("") + " " + events.target.astype("string").fillna("")
        objectives = events.loc[text.str.contains("mid.?boss|rejuven|guardian|walker|patron|shrine|urn|objective|base", case=False)]
        for kind, rows in objectives.dropna(subset=["game_time"]).groupby("event_type"):
            axes[1].scatter(rows.game_time / 60, [kind] * len(rows), s=20)
    axes[1].set(title="Recorded major objectives", xlabel="Game time (minutes)")
    fig.savefig(output / f"{mid}_timelines.png", dpi=140)
    plt.close(fig)
    return output


