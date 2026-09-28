import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def state_labels(number):
    if number == 5:
        return ["Far Behind", "Behind", "Average", "Ahead", "Far Ahead"]
    return ["Average"] if number == 1 else [f"State {i + 1}" for i in range(number)]


def fit_boundaries(values, n_states):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or n_states < 1:
        raise ValueError("State estimation needs finite observations and n_states >= 1")
    cuts = np.unique(np.quantile(values, np.arange(1, n_states) / n_states))
    return cuts[(cuts > values.min()) & (cuts < values.max())]


def assign_states(values, cuts):
    return np.searchsorted(cuts, np.asarray(values, dtype=float), side="right") + 1


def empirical_probabilities(counts):
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)


def transition_counts(transitions, n_states):
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    if len(transitions):
        np.add.at(counts, (transitions.state.to_numpy(dtype=int) - 1,
                          transitions.next_state.to_numpy(dtype=int) - 1), 1)
    return counts


def make_transitions(panel, horizons, step_seconds, entity_col="player_slot"):
    frame = panel.sort_values(["match_id", entity_col, "game_time"]).reset_index(drop=True).copy()
    gap = frame.groupby(["match_id", entity_col]).game_time.diff()
    frame["sequence_id"] = (~np.isclose(gap.to_numpy(dtype=float), step_seconds)).cumsum()
    grouped = frame.groupby("sequence_id", sort=False)
    results = []
    for horizon in sorted(set(horizons)):
        steps = horizon / step_seconds
        if horizon <= 0 or not np.isclose(steps, round(steps)):
            raise ValueError("Transition horizons must be positive multiples of the base step")
        future = grouped[["game_time", "state", "relative_net_worth"]].shift(-int(round(steps)))
        valid = np.isclose(future.game_time.to_numpy(dtype=float) - frame.game_time.to_numpy(dtype=float), horizon)
        rows = frame.loc[valid].copy()
        rows["next_game_time"] = future.loc[valid, "game_time"]
        rows["next_state"] = future.loc[valid, "state"].astype(int)
        rows["next_relative_net_worth"] = future.loc[valid, "relative_net_worth"]
        rows["horizon_seconds"] = horizon
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def bin_index(game_time, starts):
    return np.searchsorted(starts, game_time, side="right") - 1


def forecast_probabilities(model, current_state, current_time, horizon, use_time_bins=True):
    step = model["step_seconds"]
    steps = horizon / step
    if current_time < 0 or not np.isfinite(current_time) or horizon < 0 or not np.isfinite(steps) or not np.isclose(steps, round(steps)):
        raise ValueError("Time must be nonnegative and horizon must be a nonnegative multiple of the model step")
    if not np.isfinite(current_state) or current_state != int(current_state) or not 1 <= current_state <= len(model["state_labels"]):
        raise ValueError("State IDs start at 1 and must be present in the model")
    distribution = np.eye(len(model["state_labels"]))[int(current_state) - 1]
    for k in range(int(round(steps))):
        stage = int(bin_index(current_time + k * step, model["time_bin_starts"]))
        matrix = model["probabilities_by_time_bin"][stage] if use_time_bins else model["probabilities_overall"]
        distribution = distribution @ np.asarray(matrix)
    return distribution


def predict_state_distribution(current_state, current_game_time, prediction_horizon, model):
    # Times are seconds. The returned wealth value is a state-mean proxy.
    if isinstance(current_state, str):
        current_state = model["state_labels"].index(current_state) + 1
    probabilities = forecast_probabilities(model, current_state, current_game_time, prediction_horizon)
    stage = int(bin_index(current_game_time + prediction_horizon, model["time_bin_starts"]))
    values = np.asarray(model["representatives_by_time_bin"][stage])
    result = {"probabilities": dict(zip(model["state_labels"], probabilities.tolist())),
              "expected_relative_net_worth": float(probabilities @ values),
              "interpretation": "Probability-weighted training state means in the target time bin; not a prediction interval",
              "horizon_seconds": prediction_horizon, "step_seconds": model["step_seconds"]}
    if model.get("entity_name") == "team":
        result["expected_net_worth_advantage"] = 2 * result["expected_relative_net_worth"]
        result["interpretation"] += "; a per-team marginal forecast, not a joint forecast of both teams"
    return result


def fit_model(train, train_transitions, cuts, labels, step, starts, smoothing_alpha=0.5):
    number = len(labels)
    counts = transition_counts(train_transitions, number)

    def smooth(values, fallback):
        probabilities = (values + smoothing_alpha) / np.maximum(values.sum(axis=1, keepdims=True) + smoothing_alpha * number, 1e-15)
        empty = values.sum(axis=1) == 0
        probabilities[empty] = fallback[empty]
        return probabilities

    overall = smooth(counts, np.full((number, number), 1 / number))
    means = train.groupby("state").relative_net_worth.mean().reindex(range(1, number + 1))
    means = means.fillna(train.relative_net_worth.mean()).to_numpy(dtype=float)
    by_stage, stage_counts, representatives = [], [], []
    for stage in range(len(starts)):
        values = transition_counts(train_transitions.loc[train_transitions.time_bin == stage], number)
        by_stage.append(smooth(values, overall).tolist())
        stage_counts.append(values.tolist())
        stage_means = train.loc[train.time_bin == stage].groupby("state").relative_net_worth.mean()
        representatives.append([float(stage_means.get(i + 1, means[i])) for i in range(number)])
    return {"step_seconds": step, "state_labels": labels, "cutpoints": cuts.tolist(),
            "time_bin_starts": starts.tolist(), "smoothing_alpha": smoothing_alpha,
            "probabilities_overall": overall.tolist(), "probabilities_by_time_bin": by_stage,
            "training_counts_overall": counts.tolist(), "training_counts_by_time_bin": stage_counts,
            "representatives_overall": means.tolist(), "representatives_by_time_bin": representatives,
            "training_match_ids": sorted(train.match_id.unique().astype(int).tolist())}


def stage_names(starts):
    return [f"{left / 60:g}\u2013{starts[i + 1] / 60:g} min" if i + 1 < len(starts) else f"{left / 60:g}+ min"
            for i, left in enumerate(starts)]


def save_matrices(transitions, labels, starts, step, horizons, output_dir):
    records, matrices = [], {}
    for horizon in horizons:
        rows = transitions.loc[transitions.horizon_seconds == horizon]
        counts = transition_counts(rows, len(labels))
        overall = empirical_probabilities(counts)
        prefix = "" if horizon == step else f"_{horizon:g}s"
        pd.DataFrame(overall, index=labels, columns=labels).rename_axis("current_state").to_csv(output_dir / f"transition_matrix{prefix}.csv")
        pd.DataFrame(counts, index=labels, columns=labels).rename_axis("current_state").to_csv(output_dir / f"transition_counts{prefix}.csv")
        matrices[horizon] = [overall]
        for stage, name in enumerate(stage_names(starts)):
            subset = rows.loc[rows.time_bin == stage]
            stage_counts = transition_counts(subset, len(labels))
            probabilities = empirical_probabilities(stage_counts)
            matrices[horizon].append(probabilities)
            filename = f"{starts[stage] / 60:g}min{prefix}"
            pd.DataFrame(probabilities, index=labels, columns=labels).rename_axis("current_state").to_csv(output_dir / "time_bins" / f"transition_matrix_{filename}.csv")
            pd.DataFrame(stage_counts, index=labels, columns=labels).rename_axis("current_state").to_csv(output_dir / "time_bins" / f"transition_counts_{filename}.csv")
            active = stage_counts.sum(axis=1) > 0
            tv = 0.5 * np.nansum(np.abs(probabilities - overall), axis=1)
            weight = stage_counts.sum(axis=1)
            records.append({"horizon_seconds": horizon, "time_bin": name, "transitions": len(subset),
                            "weighted_total_variation_vs_overall": np.average(tv[active], weights=weight[active]) if active.any() else np.nan,
                            "maximum_row_total_variation": tv[active].max() if active.any() else np.nan,
                            "stay_probability": np.trace(stage_counts) / len(subset) if len(subset) else np.nan})
    comparison = pd.DataFrame(records)
    comparison.to_csv(output_dir / "stage_comparison.csv", index=False)
    return matrices, comparison


def evaluate(model, transitions, train, horizons, settings):
    output_dir = Path(settings["output_dir"])
    entity_col = settings["entity_col"]
    smoothing_alpha = settings["smoothing_alpha"]
    log_loss_epsilon = settings["log_loss_epsilon"]
    identity_columns = ["match_id", entity_col] + (["player_id"] if entity_col == "player_slot" else [])
    test = transitions.loc[(transitions.split == "test") & transitions.horizon_seconds.isin(horizons)].copy()
    metric_rows, match_rows, forecasts = [], [], []
    number = len(model["state_labels"])
    marginal = np.bincount(train.state.to_numpy(dtype=int) - 1, minlength=number) + smoothing_alpha
    marginal = marginal / marginal.sum()
    for horizon, rows in test.groupby("horizon_seconds", sort=True):
        cache = {}
        predictions = {"time_binned_markov": [], "overall_markov": [], "persistence": [], "training_marginal": []}
        for row in rows.itertuples():
            for name, use_bins in (("time_binned_markov", True), ("overall_markov", False)):
                key = (name, row.state, row.game_time)
                if key not in cache:
                    cache[key] = forecast_probabilities(model, row.state, row.game_time, horizon, use_bins)
                predictions[name].append(cache[key])
            predictions["persistence"].append(np.eye(number)[row.state - 1])
            predictions["training_marginal"].append(marginal)
        truth = rows.next_state.to_numpy(dtype=int) - 1
        for name, vectors in predictions.items():
            probabilities = np.asarray(vectors)
            hits = probabilities.argmax(axis=1) == truth
            losses = -np.log(np.clip(probabilities[np.arange(len(rows)), truth], log_loss_epsilon, 1))
            scores = pd.DataFrame({"match_id": rows.match_id.to_numpy(), "accuracy": hits, "log_loss": losses})
            per_match = scores.groupby("match_id").agg(accuracy=("accuracy", "mean"), log_loss=("log_loss", "mean"), transitions=("accuracy", "size")).reset_index()
            per_match["model"], per_match["horizon_seconds"] = name, horizon
            match_rows.append(per_match)
            metric_rows.append({"model": name, "horizon_seconds": horizon, "transitions": len(rows),
                                "matches": rows.match_id.nunique(), "accuracy": hits.mean(), "log_loss": losses.mean(),
                                "match_macro_accuracy": per_match.accuracy.mean(), "match_macro_log_loss": per_match.log_loss.mean()})
            if name == "time_binned_markov":
                saved = rows[identity_columns + ["game_time", "next_game_time", "state", "next_state", "horizon_seconds", "next_relative_net_worth"]].reset_index(drop=True)
                for index in range(number):
                    saved[f"probability_state_{index + 1}"] = probabilities[:, index]
                saved["predicted_state"] = probabilities.argmax(axis=1) + 1
                target_stages = bin_index(saved.next_game_time, model["time_bin_starts"])
                values = np.asarray(model["representatives_by_time_bin"])[target_stages]
                saved["expected_relative_net_worth"] = np.sum(probabilities * values, axis=1)
                if settings["entity_name"] == "team":
                    saved["next_net_worth_advantage"] = 2 * saved.next_relative_net_worth
                    saved["expected_net_worth_advantage"] = 2 * saved.expected_relative_net_worth
                forecasts.append(saved)
    metrics = pd.DataFrame(metric_rows, columns=["model", "horizon_seconds", "transitions", "matches",
                                               "accuracy", "log_loss", "match_macro_accuracy", "match_macro_log_loss"])
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    per_match = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame(columns=["match_id", "model", "horizon_seconds", "accuracy", "log_loss", "transitions"])
    per_match.to_csv(output_dir / "validation_by_match.csv", index=False)
    forecast = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(columns=identity_columns + ["horizon_seconds"])
    forecast.to_parquet(output_dir / "heldout_predictions.parquet", index=False, compression="zstd", compression_level=19)
    return metrics


def draw_matrix(ax, matrix, labels, title):
    shown = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Next state", ylabel="Current state", title=title)
    ax.tick_params(axis="x", rotation=35)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "\u2014", ha="center", va="center",
                    color="white" if value > 0.55 else "black", fontsize=9)
    return shown


def make_figures(panel, model, matrices, transitions, state_summary, settings):
    output_dir = Path(settings["output_dir"])
    entity_col, entity_name = settings["entity_col"], settings["entity_name"]
    net_worth_col = settings["net_worth_col"]
    n_examples = settings["n_examples"]
    example_horizon_seconds = settings["example_horizon_seconds"]
    labels, starts, step = model["state_labels"], model["time_bin_starts"], model["step_seconds"]
    colors = plt.get_cmap("RdYlBu")(np.linspace(0.08, 0.92, len(labels)))
    figure_dir = output_dir / "figures"
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    image = draw_matrix(ax, matrices[step][0], labels, f"Economic-state transitions over {step:g} seconds\nAll matches; empirical probabilities")
    fig.colorbar(image, ax=ax, label="Probability")
    fig.savefig(figure_dir / "01_transition_heatmap.png", dpi=170)
    plt.close(fig)
    for horizon, values in matrices.items():
        columns = min(2, len(starts))
        fig, axes = plt.subplots(math.ceil(len(starts) / columns), columns, figsize=(7 * columns, 5 * math.ceil(len(starts) / columns)), squeeze=False, layout="constrained")
        for stage, ax in enumerate(axes.flat):
            if stage >= len(starts):
                ax.set_visible(False)
                continue
            count = len(transitions.loc[(transitions.horizon_seconds == horizon) & (transitions.time_bin == stage)])
            draw_matrix(ax, values[stage + 1], labels, f"{stage_names(starts)[stage]} | {count:,} transitions")
        fig.suptitle(f"Empirical {horizon:g}-second transitions by starting game time; all matches")
        fig.savefig(figure_dir / f"02_time_bin_heatmaps_{horizon:g}s.png", dpi=170)
        plt.close(fig)
    eligible = transitions.loc[(transitions.horizon_seconds == example_horizon_seconds) & (transitions.split == "test")]
    example_scope = "Held-out"
    if eligible.empty and len(transitions):
        eligible = transitions.loc[transitions.horizon_seconds == max(transitions.horizon_seconds)]
        example_scope = "Descriptive (no eligible held-out examples)"
    examples = []
    for mid, candidates in eligible.groupby("match_id", sort=True):
        if len(examples) >= n_examples:
            break
        target = candidates.iloc[(candidates.game_time - 720).abs().argsort().iloc[0]].game_time
        candidates = candidates.loc[candidates.game_time == target]
        row = candidates.iloc[(candidates.relative_net_worth - candidates.relative_net_worth.median()).abs().argsort().iloc[0]]
        examples.append(row)
    example_records = []
    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3.2 * len(examples)), squeeze=False, layout="constrained")
        for ax, example in zip(axes.flat, examples):
            track = panel.loc[(panel.match_id == example.match_id) & (panel[entity_col] == example[entity_col])]
            ax.plot(track.game_time / 60, track[net_worth_col], color="#999999", linewidth=0.8)
            for state, label in enumerate(labels, 1):
                subset = track.loc[track.state == state]
                ax.scatter(subset.game_time / 60, subset[net_worth_col], s=10, color=colors[state - 1], label=label)
            ax.set(title=f"Match {int(example.match_id)}, {entity_name} {int(example[entity_col])}", xlabel="Game time (minutes)", ylabel=f"{entity_name.title()} net worth")
            ax.grid(alpha=0.2)
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside upper center", ncol=min(5, len(labels)))
        fig.savefig(figure_dir / "03_actual_net_worth_states.png", dpi=170)
        plt.close(fig)
        fig, axes = plt.subplots(len(examples), 2, figsize=(15, 3.8 * len(examples)), squeeze=False, layout="constrained")
        for index, example in enumerate(examples):
            times = np.arange(0, example.horizon_seconds + step / 2, step)
            clock = example.game_time + times
            probabilities = np.asarray([forecast_probabilities(model, int(example.state), example.game_time, horizon) for horizon in times])
            track = panel.loc[(panel.match_id == example.match_id) & (panel[entity_col] == example[entity_col])].set_index("game_time").loc[clock]
            left, right = axes[index]
            left.stackplot(clock / 60, probabilities.T, colors=colors, labels=labels, alpha=0.85)
            left.set(xlabel="Game time (minutes)", ylabel="Predicted probability", ylim=(0, 1),
                     title=f"{example_scope}: match {int(example.match_id)}, {entity_name} {int(example[entity_col])}")
            actual_axis = left.twinx()
            actual_axis.step(clock / 60, track.state, where="post", color="black", linewidth=1.4)
            actual_axis.set(yticks=range(1, len(labels) + 1), ylim=(0.5, len(labels) + 0.5), ylabel="Actual state (black line)")
            target_stage = bin_index(clock, starts)
            representative = np.sum(probabilities * np.asarray(model["representatives_by_time_bin"])[target_stage], axis=1)
            right.plot(clock / 60, track.relative_net_worth, color="black", label="Actual relative net worth")
            right.plot(clock / 60, representative, color="#bb4a35", linestyle="--", label="Predicted state-mean proxy")
            right.axhline(0, color="#aaaaaa", linewidth=0.7)
            right.set(xlabel="Game time (minutes)", ylabel="Relative net worth (souls)")
            right.legend(fontsize=8)
            for k, horizon in enumerate(times):
                example_records.append({"match_id": int(example.match_id), entity_col: int(example[entity_col]),
                                        "origin_time": example.game_time, "horizon_seconds": horizon,
                                        "actual_state": int(track.state.iloc[k]), "actual_relative_net_worth": track.relative_net_worth.iloc[k],
                                        "expected_relative_net_worth": representative[k],
                                        **{f"probability_state_{s + 1}": probabilities[k, s] for s in range(len(labels))}})
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside upper center", ncol=min(5, len(labels)))
        fig.savefig(figure_dir / "04_forecasts_vs_realized.png", dpi=170)
        plt.close(fig)
    example_predictions = pd.DataFrame(example_records)
    if entity_name == "team" and len(example_predictions):
        example_predictions["actual_net_worth_advantage"] = 2 * example_predictions.actual_relative_net_worth
        example_predictions["expected_net_worth_advantage"] = 2 * example_predictions.expected_relative_net_worth
    example_predictions.to_csv(output_dir / "example_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.bar(labels, state_summary.relative_net_worth_mean, color=colors, edgecolor="white")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set(title="Mean relative net worth within each economic state\nTraining matches only", ylabel="Relative net worth (souls)")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(figure_dir / "05_mean_relative_net_worth_by_state.png", dpi=170)
    plt.close(fig)


def run_analysis(panel, step, audit, settings):
    output_dir = Path(settings["output_dir"])
    entity_col, entity_name = settings["entity_col"], settings["entity_name"]
    n_states = settings["n_states"]
    game_time_bins_minutes = settings["game_time_bins_minutes"]
    direct_horizons_seconds = settings["direct_horizons_seconds"]
    validation_horizons_seconds = settings["validation_horizons_seconds"]
    test_match_fraction = settings["test_match_fraction"]
    random_seed = settings["random_seed"]
    smoothing_alpha = settings["smoothing_alpha"]
    max_state_age_seconds = settings["max_state_age_seconds"]
    example_horizon_seconds = settings["example_horizon_seconds"]
    log_loss_epsilon = settings["log_loss_epsilon"]
    script_name = settings["script_name"]
    panel, audit = panel.copy(), audit.copy()
    if not 0 < test_match_fraction < 1 or smoothing_alpha < 0:
        raise ValueError("Use a test fraction between 0 and 1 and nonnegative smoothing")
    starts = np.asarray(game_time_bins_minutes, dtype=float) * 60
    if not len(starts) or starts[0] != 0 or not np.all(np.diff(starts) > 0):
        raise ValueError("Time bins must start at zero and increase")
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("figures", "time_bins"):
        (output_dir / directory).mkdir(exist_ok=True)
    ids = np.sort(panel.match_id.unique().astype(int))
    test_count = min(len(ids) - 2, max(1, math.ceil(len(ids) * test_match_fraction))) if len(ids) >= 3 else 0
    test_ids = np.random.default_rng(random_seed).permutation(ids)[:test_count]
    panel["split"] = np.where(panel.match_id.isin(test_ids), "test", "train")
    train = panel.loc[panel.split == "train"]
    cuts = fit_boundaries(train.relative_net_worth, n_states)
    labels = state_labels(len(cuts) + 1)
    panel["state"] = assign_states(panel.relative_net_worth, cuts)
    panel["state_label"] = panel.state.map(dict(enumerate(labels, 1)))
    panel["time_bin"] = bin_index(panel.game_time, starts)
    panel["time_bin_label"] = panel.time_bin.map(dict(enumerate(stage_names(starts))))
    requested = sorted(set([step, *direct_horizons_seconds, *validation_horizons_seconds, example_horizon_seconds]))
    horizons = [h for h in requested if h > 0 and np.isclose(h / step, round(h / step))]
    transitions = make_transitions(panel, horizons, step, entity_col)
    direct = sorted(set([step, *[h for h in direct_horizons_seconds if h in horizons]]))
    base = transitions.loc[transitions.horizon_seconds == step]
    next_values = base[["match_id", entity_col, "game_time", "next_game_time", "next_state", "next_relative_net_worth"]]
    modeling = panel.merge(next_values, on=["match_id", entity_col, "game_time"], how="left", validate="one_to_one")
    modeling["valid_next_transition"] = modeling.next_state.notna()
    modeling["next_state"] = modeling.next_state.astype("Int64")
    modeling.to_parquet(output_dir / "modeling_dataset.parquet", index=False, compression="zstd", compression_level=19)
    transitions.to_parquet(output_dir / "transitions.parquet", index=False, compression="zstd", compression_level=19)
    train = panel.loc[panel.split == "train"]
    model = fit_model(train, base.loc[base.split == "train"], cuts, labels, step, starts, smoothing_alpha)
    model.update(test_match_ids=sorted(test_ids.astype(int).tolist()), random_seed=random_seed,
                 notes="Training-only thresholds and probabilities. Empty stage rows fall back to smoothed overall rows. Forecasts condition on the match continuing.")
    relative_definition = ("Team total net worth minus the mean of both team totals; half the advantage over the opposing team"
                           if entity_name == "team" else "Player net worth minus the mean net worth of all players at the same match time")
    model.update(entity_name=entity_name, entity_col=entity_col, net_worth_col=settings["net_worth_col"],
                 relative_net_worth_definition=relative_definition)
    if entity_name == "team":
        model["notes"] += " Paired teams are complementary and dependent, held out together; predictions are per-team marginals, not joint forecasts."
    (output_dir / "model.json").write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
    matrices, comparison = save_matrices(transitions, labels, starts, step, direct, output_dir)
    pd.DataFrame(empirical_probabilities(np.asarray(model["training_counts_overall"])), index=labels, columns=labels).rename_axis("current_state").to_csv(output_dir / "training_transition_matrix.csv")
    pd.DataFrame(model["probabilities_overall"], index=labels, columns=labels).rename_axis("current_state").to_csv(output_dir / "prediction_transition_matrix.csv")
    state_summary = train.groupby("state").relative_net_worth.agg(observations="size", relative_net_worth_mean="mean", median="median", observed_min="min", observed_max="max").reindex(range(1, len(labels) + 1)).reset_index()
    state_summary["label"] = labels
    state_summary["lower_bound"] = np.r_[-np.inf, cuts]
    state_summary["upper_bound"] = np.r_[cuts, np.inf]
    if entity_name == "team":
        state_summary["net_worth_advantage_mean"] = 2 * state_summary.relative_net_worth_mean
        state_summary["advantage_lower_bound"] = 2 * state_summary.lower_bound
        state_summary["advantage_upper_bound"] = 2 * state_summary.upper_bound
    state_summary.to_csv(output_dir / "state_definitions.csv", index=False)
    audit["split"] = audit.match_id.map(panel.groupby("match_id").split.first())
    audit.to_csv(output_dir / "input_audit.csv", index=False)
    metrics = evaluate(model, transitions, train, [h for h in [step, *validation_horizons_seconds] if h in horizons], settings)
    make_figures(panel, model, matrices, transitions, state_summary, settings)
    persistence = pd.Series(np.diag(matrices[step][0]), index=labels).sort_values(ascending=False)
    persistence_text = "State persistence: " + "; ".join(f"{label}: {value:.1%}" for label, value in persistence.items())
    stage_diagonals = pd.DataFrame([np.diag(matrix) for matrix in matrices[step][1:]], columns=labels)
    ranges = stage_diagonals.max() - stage_diagonals.min()
    if ranges.notna().any():
        variable_state = ranges.idxmax()
        stage_text = (f"Largest change across game stages: {variable_state} persistence ranges from "
                      f"{stage_diagonals[variable_state].min():.1%} to {stage_diagonals[variable_state].max():.1%} "
                      f"({100 * ranges.max():.1f} percentage points). "
                      "This is descriptive, not a significance test; inspect counts for sparse states.")
    else:
        stage_text = "Game-stage differences are unavailable: no eligible transitions."
    split_summary = panel.groupby("split").agg(matches=("match_id", "nunique"), observations=("state", "size"))
    trajectories = panel[["split", "match_id", entity_col]].drop_duplicates().groupby("split").size()
    split_summary[f"{entity_name}_match_trajectories"] = trajectories
    if entity_name == "player":
        split_summary["players"] = panel.groupby("split").player_id.nunique()
    split_summary["base_transitions"] = base.groupby("split").size().reindex(split_summary.index, fill_value=0)
    if entity_name == "team":
        paired_counts = base.drop_duplicates(["match_id", "game_time"]).groupby("split").size()
        split_summary["paired_match_time_transitions"] = paired_counts.reindex(split_summary.index, fill_value=0)
    split_summary.to_csv(output_dir / "split_summary.csv")
    native = sorted(panel.native_step_seconds.unique().tolist())
    entity_count = f"{panel.groupby(['match_id', entity_col]).ngroups} {entity_name}-match trajectories"
    if entity_name == "player":
        entity_count = f"Players: {panel.player_id.nunique()} known unique accounts; {entity_count}"
    else:
        entity_count = f"Teams: {entity_count}; team IDs identify sides within each match, not persistent team identities"
    observations_text = f"Observations: {len(panel):,}; consecutive {step:g}-second transitions: {len(base):,}"
    if entity_name == "team":
        paired_count = len(base.drop_duplicates(["match_id", "game_time"]))
        observations_text += f" team perspectives from {paired_count:,} paired match-time transitions"
    lines = [f"Exploratory Markov model of {entity_name} relative net worth", "",
             f"Matches: {len(ids)} ({len(ids) - test_count} train, {test_count} held out; seed {random_seed})",
             entity_count,
             observations_text,
             f"Native sampling intervals: {native}; common analysis step: {step:g} seconds, without interpolation",
             f"Retained fresh replay wealth at complete-roster timestamps; maximum source age {max_state_age_seconds:g}s.",
             "Rank 1 is richest; percentile ranges from 0 (poorest) to 1 (richest), with average ranks for ties.",
             f"Requested states: {n_states}; effective states: {len(labels)}. Tied/endpoint quantile cuts are removed.",
             "State boundaries use pooled TRAINING observations only; intervals are [lower, upper).",
             state_summary[["state", "label", "lower_bound", "upper_bound", "relative_net_worth_mean"]].to_string(index=False), "",
             f"All-match empirical transition matrix ({step:g}s; undefined rows are NaN):",
             pd.DataFrame(matrices[step][0], index=labels, columns=labels).to_string(float_format=lambda x: f"{x:.4f}"), "",
             persistence_text, stage_text,
             "Game-stage differences: weighted total variation compares each stage's outgoing probabilities with overall probabilities.",
             comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"), "",
             "Counts by whole-match split:", split_summary.to_string(), "",
             "Held-out performance (overlapping windows; accuracy and log loss are descriptive):",
             metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(metrics) else "Not evaluated: too few matches or eligible future observations.", "",
             f"Validation is preliminary: matches, {entity_name}s, and overlapping transitions are dependent. Match-macro metrics weight each held-out match equally; no independent-row standard errors are reported.",
             f"Relative net worth: {relative_definition}.",
             "The short-horizon persistence baseline is essential: high accuracy can mostly reflect remaining in the same state.",
             f"Predictive matrices use additive smoothing alpha={smoothing_alpha}; empty stage rows use the overall training row. Log loss clips probabilities at {log_loss_epsilon:g}, including the deterministic persistence baseline.",
             "Descriptive matrices include all matches; prediction/validation matrices, cutpoints and state representatives use training matches only.",
             "All transitions require uninterrupted valid observations on the common grid, including direct longer-horizon transitions.",
             "Transitions are grouped by their starting game-time bin; recursive forecasts switch matrices as game time advances.",
             "Predictions assume the match continues. Evaluation excludes unavailable future observations, so results are conditional on future coverage and match survival.",
             "Expected relative wealth is a probability-weighted training state-mean proxy at the target game stage, not a calibrated continuous-wealth prediction or interval.",
             f"Pooled quintiles weight {entity_name}-time observations; longer games contribute more observations. This is a selected replay sample, not a random population sample.",
             "A successful fit does not establish the Markov property or independence; hero/player effects and deeper history are omitted.",
             "Fixed pooled thresholds can cause stage differences as economic dispersion grows; state labels describe relative wealth, not within-time quintiles.",
             f"Skipped incompatible requested horizons: {[h for h in requested if h not in horizons]}",
             f"Rerun: .\\.venv\\Scripts\\python.exe {script_name}.py (edit the settings at the top of the script).",
             f"Forecast: from {script_name} import predict_state_distribution; predict_state_distribution('Behind', 12 * 60, 5 * 60)",
             "Forecast arguments are seconds; horizons must be multiples of the fitted step. State IDs start at 1, or use a saved label.",
             f"Read features: pandas.read_parquet('{(output_dir / 'modeling_dataset.parquet').as_posix()}').",
             "modeling_dataset.parquet contains observations and next-step outcomes; transitions.parquet contains all requested horizons. Both use lossless Zstandard compression.",
             "transition_matrix*.csv and time_bins/ are all-match empirical estimates. model.json contains training-only prediction matrices and state means.",
             "validation_metrics.csv and validation_by_match.csv compare held-out forecasts with baselines; heldout_predictions.parquet stores the probability vectors.",
             f"Outputs: {output_dir.resolve()}"]
    if entity_name == "team":
        lines.extend(["Both teams at each match time form a dependent complementary pair: relative values sum to zero, and net_worth_advantage is twice relative_net_worth.",
                      "Both teams are always held out together. Counts include both team perspectives and must not be read as independent evidence.",
                      "Forecasts describe one team's marginal state distribution, not a joint distribution that enforces compatible outcomes for both teams."])
    report = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(report, encoding="utf-8")
    concise = lines[:11] + ["", lines[12], lines[13], "", persistence_text, stage_text]
    print("\n".join(concise))
    print(f"\nRelative net worth: {relative_definition}.")
    if entity_name == "team":
        print("Both team perspectives are dependent and held out together; forecasts are per-team marginals.")
    print("\nHeld-out time-binned Markov versus persistence:")
    if len(metrics):
        print(metrics.loc[metrics.model.isin(["time_binned_markov", "persistence"])].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nPreliminary, correlated observations. Full results: {output_dir / 'summary.txt'}")
    return model
