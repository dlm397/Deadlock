"""Stage 3: fine-tune the Stage 2 causal backbone on match-prefix win prediction.

Explicit train, predict, evaluate, and verify commands; no work runs on import.
The output logit is for Team1 winning. Apply sigmoid only for probabilities.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from app.config import DEFAULT_CONFIG
from app.console_log import log
import pretrain_static as static
import pretrain_dynamics as dynamics


ROOT = Path(__file__).resolve().parent
ARCHITECTURE = {key: value for key, value in dynamics.ARCHITECTURE.items() if key != "dynamics_head"}
ARCHITECTURE.update(win_head=[64, 32, 1], readout="temporal representation at last unpadded frame")
CONVENTIONS = {
    "label": "1 if matches.winning_team_id == 1, otherwise 0; TeamWin outcomes only",
    "probability": "sigmoid(logit) = P(Team1 wins); P(Team0 wins) = 1 - P(Team1 wins)",
    "team_order": [0, 1],
    "eligible_frames": "exclude final observations and times >= min(duration_s, terminal_game_time_s when available)",
    "inputs": "Stage 2 current-state features and static identities only; outcome and terminal metadata are never features",
    "sampling": "one uniformly sampled eligible cutoff per match per epoch; matches shuffled without replacement",
    "validation": "one fixed seeded cutoff per match for checkpoint selection; fixed elapsed times after selection",
}


class WinModel(dynamics.DynamicsModel):
    def __init__(self, num_heroes, dropout=0.1):
        super().__init__(num_heroes, dropout)
        del self.dynamics_head
        self.win_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, hero_indices, scaled_ranks, player_features, global_features, times, padding_mask):
        frames = self.encode_frames(hero_indices, scaled_ranks, player_features, global_features, times, padding_mask)
        cutoffs = (~padding_mask).sum(dim=1) - 1
        last = frames[torch.arange(len(frames), device=frames.device), cutoffs]
        return self.win_head(last).squeeze(-1)


def load_stage2_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("kind") not in ("stage2_complete", "stage2_transfer")
            or checkpoint.get("architecture") != dynamics.ARCHITECTURE):
        raise ValueError("Expected a compatible Stage 2 best_model.pt or dynamics_encoder.pt")
    for key in ("static_preprocessing", "preprocessing", "split_assignments", "active_match_ids"):
        if not checkpoint.get(key):
            raise ValueError(f"Stage 2 checkpoint is missing {key}")
    return checkpoint


def initialize_model(checkpoint, dropout=0.1):
    model = WinModel(len(checkpoint["static_preprocessing"]["hero_id_to_index"]) + 1, dropout)
    if checkpoint["kind"] == "stage2_transfer":
        weights = checkpoint["backbone_state_dict"]
    else:
        weights = {key: value for key, value in checkpoint["model_state_dict"].items()
                   if not key.startswith("dynamics_head.")}
    expected = {key for key in model.state_dict() if not key.startswith("win_head.")}
    if set(weights) != expected:
        raise ValueError("Stage 2 backbone keys do not match the win model")
    model.load_state_dict({**model.state_dict(), **weights}, strict=True)
    return model


def load_win_matches(database, match_ids=None):
    matches, _ = dynamics.load_matches(database, match_ids)
    with dynamics.open_database(database) as connection:
        rows = connection.execute("""
            SELECT m.match_id, m.match_outcome, m.winning_team_id, m.duration_s,
                   a.terminal_game_time_s
            FROM matches m JOIN processing_attempts a ON m.successful_attempt_id = a.attempt_id
            WHERE m.match_id IN (SELECT unnest(?::UBIGINT[]))
        """, [[match["match_id"] for match in matches]]).fetchall()
    outcomes = {row[0]: row[1:] for row in rows}
    usable, skipped = [], []
    for match in matches:
        outcome, winner, duration, terminal = outcomes[match["match_id"]]
        if outcome != "TeamWin" or winner not in (0, 1):
            skipped.append({"match_id": match["match_id"], "reason": "missing or non-TeamWin outcome"})
            continue
        boundaries = [float(duration)] + ([] if terminal is None else [float(terminal)])
        if not all(math.isfinite(value) and value > 0 for value in boundaries):
            raise ValueError(f"Invalid terminal boundary for match {match['match_id']}")
        match.update(label=float(winner), end_time_s=min(boundaries))
        usable.append(match)
    return usable, skipped


def prepare_groups(matches, checkpoint, split_path=None):
    """Keep the saved assignments authoritative; never create or rewrite a split."""
    saved = checkpoint["split_assignments"]
    if any(split not in static.SPLITS for split in saved.values()):
        raise ValueError("Invalid Stage 2 split assignments")
    if split_path is not None and Path(split_path).exists():
        with Path(split_path).open(encoding="utf-8-sig", newline="") as stream:
            seen = set()
            for row in csv.DictReader(stream):
                mid = str(int(row["match_id"]))
                if mid in seen or row["split"] not in static.SPLITS:
                    raise ValueError("Duplicate or invalid shared split assignment")
                seen.add(mid)
                if mid in saved and saved[mid] != row["split"]:
                    raise ValueError(f"Split conflict with Stage 2 for match {mid}")
    for split, ids in checkpoint["active_match_ids"].items():
        if split not in static.SPLITS or any(saved.get(str(mid)) != split for mid in ids):
            raise ValueError("Stage 2 active IDs disagree with its saved splits")
    groups = {split: [] for split in static.SPLITS}
    provenance = checkpoint.get("data_source", {}).get("match_provenance", {})
    active = {mid for ids in checkpoint["active_match_ids"].values() for mid in ids}
    for match in matches:
        mid = match["match_id"]
        if mid not in active or str(mid) not in saved:
            raise ValueError(f"Match {mid} is outside the Stage 2 collection; supply a matching checkpoint")
        if str(mid) in provenance and provenance[str(mid)] != match["provenance"]:
            raise ValueError(f"Successful replay source changed since Stage 2 for match {mid}")
        groups[saved[str(mid)]].append(match)
    return groups


def attach_eligible_times(database, matches, frame_stride=1):
    """Read timestamps/flags only. Downsample before filtering to keep the Stage 2 grid."""
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    report = {"matches": len(matches), "eligible_frames": 0, "excluded_matches": [], "max_prefix_frames": 0}
    with dynamics.open_database(database) as connection:
        for match in matches:
            rows = connection.execute("""
                SELECT game_time_s, is_final_observation FROM global_snapshots
                WHERE match_id = ? ORDER BY game_time_s
            """, [match["match_id"]]).fetchall()
            times = np.array([row[0] for row in rows], dtype=np.float64)
            final = np.array([row[1] for row in rows], dtype=bool)
            if (not len(times) or not np.isfinite(times).all() or (times < 0).any()
                    or (np.diff(times) <= 0).any()):
                raise ValueError(f"Invalid frame times for match {match['match_id']}")
            # A final marker must not occur inside an otherwise usable prefix.
            final_boundary = float(times[final].min()) if final.any() else float("inf")
            valid = (~final) & (times < min(match["end_time_s"], final_boundary))
            eligible = times[::frame_stride][valid[::frame_stride]]
            match.update(eligible_times=eligible.tolist(), last_observed_time_s=float(times[-1]))
            report["eligible_frames"] += len(eligible)
            report["max_prefix_frames"] = max(report["max_prefix_frames"], len(eligible))
            if not len(eligible):
                report["excluded_matches"].append(match["match_id"])
    return report


class PrefixDataset(Dataset):
    def __init__(self, database, matches, static_preprocessing, preprocessing, seed=42, fixed=False, cutoffs=None):
        self.database, self.matches = database, matches
        self.static_preprocessing, self.preprocessing = static_preprocessing, preprocessing
        self.seed, self.fixed, self.epoch, self.cutoffs = seed, fixed, 0, cutoffs

    def __len__(self):
        return len(self.matches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def cutoff_for(self, match):
        if self.cutoffs is not None:
            return self.cutoffs[match["match_id"]]
        epoch = 0 if self.fixed else self.epoch
        rng = random.Random(f"{self.seed}:{epoch}:{match['match_id']}")
        return rng.choice(match["eligible_times"])

    def __getitem__(self, index):
        match = self.matches[index]
        cutoff = self.cutoff_for(match)
        if cutoff not in match["eligible_times"]:
            raise ValueError("Prefix cutoff must be an eligible recorded frame")
        with dynamics.open_database(self.database) as connection:
            sequence = dynamics.load_sequence(connection, match, self.preprocessing["frame_stride"], cutoff)
        expected = np.asarray([time for time in match["eligible_times"] if time <= cutoff])
        if not np.array_equal(sequence["times"], expected):
            raise ValueError(f"Player/global frame mismatch for match {match['match_id']}")
        encoded = dynamics.encode_sequence(sequence, match, self.static_preprocessing, self.preprocessing)
        encoded.update(label=torch.tensor(match["label"], dtype=torch.float32), cutoff_time_s=float(cutoff))
        return encoded


def collate_prefixes(items):
    batch = dynamics.collate_sequences(items)
    batch.update(labels=torch.stack([item["label"] for item in items]),
                 cutoff_time_s=torch.tensor([item["cutoff_time_s"] for item in items], dtype=torch.float64))
    return batch


def classification_metrics(labels, logits, bins=10):
    labels, logits = np.asarray(labels, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if (labels.shape != logits.shape or labels.ndim != 1 or not np.isfinite(logits).all()
            or not np.isin(labels, [0, 1]).all() or bins < 1):
        raise ValueError("Expected finite one-dimensional logits and binary labels")
    if not len(labels):
        return {"count": 0, "positives": 0, "bce": None, "brier": None, "auc": None,
                "ece": None, "calibration": []}
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    bce = float(np.mean(np.logaddexp(0, logits) - labels * logits))
    positive, negative = int(labels.sum()), int(len(labels) - labels.sum())
    auc = None
    if positive and negative:
        order = np.argsort(logits, kind="stable")
        ranks = np.empty(len(labels), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and logits[order[end]] == logits[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        auc = float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))
    calibration, ece = [], 0.0
    indexes = np.minimum((probabilities * bins).astype(int), bins - 1)
    for index in range(bins):
        selected = indexes == index
        count = int(selected.sum())
        predicted = float(probabilities[selected].mean()) if count else None
        observed = float(labels[selected].mean()) if count else None
        calibration.append({"lower": index / bins, "upper": (index + 1) / bins, "count": count,
                            "mean_probability": predicted, "observed_frequency": observed})
        if count:
            ece += count / len(labels) * abs(predicted - observed)
    return {"count": len(labels), "positives": positive, "bce": bce,
            "brier": float(np.mean((probabilities - labels)**2)), "auc": auc,
            "ece": float(ece), "calibration": calibration}


@torch.inference_mode()
def evaluate_prefixes(model, loader, device="cpu"):
    model.eval()
    rows = []
    for batch in loader:
        logits = dynamics.forward_batch(model, batch, device).cpu().tolist()
        for mid, cutoff, label, value in zip(batch["match_id"], batch["cutoff_time_s"].tolist(),
                                            batch["labels"].tolist(), logits):
            rows.append({"match_id": mid, "cutoff_time_s": cutoff, "label": label, "logit": value,
                         "team1_win_probability": torch.sigmoid(torch.tensor(value, dtype=torch.float64)).item()})
    return classification_metrics([row["label"] for row in rows], [row["logit"] for row in rows]), rows


def prior_metrics(rows, probability):
    clipped = min(max(probability, 1e-7), 1 - 1e-7)
    logit = math.log(clipped / (1 - clipped))
    return classification_metrics([row["label"] for row in rows], [logit] * len(rows))


def prefix_loader(dataset, batch_size, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_prefixes,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def fixed_time_evaluation(model, database, matches, checkpoint, times, device="cpu", batch_size=1):
    results, predictions = {}, []
    for requested in times:
        selected, cutoffs = [], {}
        for match in matches:
            # A finished or incompletely recorded match cannot stand in for an in-progress one.
            if requested >= match["end_time_s"] or requested > match["last_observed_time_s"]:
                continue
            eligible = [time for time in match["eligible_times"] if time <= requested]
            if eligible:
                selected.append(match)
                cutoffs[match["match_id"]] = eligible[-1]
        dataset = PrefixDataset(database, selected, checkpoint["static_preprocessing"],
                                checkpoint["preprocessing"], fixed=True, cutoffs=cutoffs)
        metrics, rows = evaluate_prefixes(model, prefix_loader(dataset, batch_size), device)
        for row in rows:
            row["requested_time_s"] = requested
        predictions.extend(rows)
        results[str(requested)] = {"requested_time_s": requested, "model": metrics,
                                   "training_prior": prior_metrics(rows, checkpoint["training_win_prior"])}
    return results, predictions


def write_rows(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(history, fixed_metrics, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if history:
        figure, axis = plt.subplots(figsize=(7, 4))
        for key, label in (("train_bce", "Training: new prefixes"), ("validation_bce", "Validation: fixed prefixes"),
                           ("prior_bce", "Training-prior baseline")):
            axis.plot([row["epoch"] for row in history], [row[key] for row in history], label=label)
        axis.set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Stage 3: win fine-tuning")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / "loss_curve.png", dpi=160)
        plt.close(figure)
    if fixed_metrics:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        for entry in fixed_metrics.values():
            nonempty = [row for row in entry["model"]["calibration"] if row["count"]]
            if nonempty:
                axis.plot([row["mean_probability"] for row in nonempty],
                          [row["observed_frequency"] for row in nonempty], "o-",
                          label=f"{entry['requested_time_s'] / 60:g} min (n={entry['model']['count']})")
        axis.set(xlabel="Mean predicted P(Team1 wins)", ylabel="Observed Team1 win fraction",
                 xlim=(0, 1), ylim=(0, 1), title="Fixed-time calibration (10 equal-width bins)")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / "calibration_curve.png", dpi=160)
        plt.close(figure)


def load_stage3(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "stage3_complete" or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected a compatible complete Stage 3 checkpoint")
    model = WinModel(len(checkpoint["static_preprocessing"]["hero_id_to_index"]) + 1,
                     checkpoint["training_config"]["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def verify_reload(path, reference_path=None):
    path = Path(path)
    reference = torch.load(reference_path or path.parent / "reload_reference.pt", weights_only=True, map_location="cpu")
    model, _ = load_stage3(path)
    with torch.inference_mode():
        logits = dynamics.forward_batch(model, reference["batch"])
    torch.testing.assert_close(logits, reference["logits"], rtol=0, atol=0)
    torch.testing.assert_close(logits.sigmoid(), reference["probabilities"], rtol=0, atol=0)
    return {"passed": True, "device": "cpu", "max_absolute_error": 0.0}


def validate_settings(args):
    if not 1 <= args.epochs <= 100 or not 1 <= args.patience <= 10:
        raise ValueError("Require epochs 1..100 and patience 1..10")
    if args.batch_size < 1 or not 1 <= args.micro_batch_size <= args.batch_size:
        raise ValueError("Require positive batch size and micro-batch-size <= batch-size")
    for key in ("learning_rate", "backbone_learning_rate", "gradient_clip"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        raise ValueError("Require nonnegative finite weight decay and dropout in [0, 1)")
    if not math.isfinite(args.lr_factor) or not 0 < args.lr_factor < 1 or args.lr_patience < 0:
        raise ValueError("Require lr-factor in (0, 1) and nonnegative lr-patience")
    validate_times(args.evaluation_times)


def validate_times(times):
    if not times or len(set(times)) != len(times) or any(not math.isfinite(time) or time < 0 for time in times):
        raise ValueError("Evaluation times must be unique, finite, nonnegative gameplay seconds")


def create_lr_scheduler(optimizer, args):
    """Optional validation-driven decay; the baseline keeps constant learning rates."""
    if args.lr_scheduler == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        threshold=0.0, threshold_mode="abs", cooldown=0,
        min_lr=[group["lr"] * 0.01 for group in optimizer.param_groups], eps=0.0,
    )


def train(args):
    validate_settings(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    started = datetime.now(timezone.utc)
    source = load_stage2_checkpoint(args.dynamics_checkpoint)
    database = dynamics.resolve_database(args)
    selected_ids = args.match_ids or [mid for ids in source["active_match_ids"].values() for mid in ids]
    matches, skipped = load_win_matches(database, selected_ids)
    groups = prepare_groups(matches, source, dynamics.resolve_split_path(args))
    preprocessing = source["preprocessing"]
    audit = {"skipped_outcomes": skipped, "splits": {}}
    for split in ("train", "validation"):
        audit["splits"][split] = attach_eligible_times(database, groups[split], preprocessing["frame_stride"])
        groups[split] = [match for match in groups[split] if match["eligible_times"]]
        if not groups[split]:
            raise ValueError(f"No eligible prefixes in {split}")
        audit["splits"][split]["usable_matches"] = len(groups[split])
    audit["splits"]["test"] = {"matches": len(groups["test"]), "observations_inspected": False}
    training_prior = sum(match["label"] for match in groups["train"]) / len(groups["train"])
    datasets = {split: PrefixDataset(database, groups[split], source["static_preprocessing"], preprocessing,
                                     args.validation_seed if split == "validation" else args.seed,
                                     fixed=split == "validation") for split in ("train", "validation")}
    loaders = {split: prefix_loader(dataset, args.batch_size if split == "train" else args.micro_batch_size,
                                    shuffle=split == "train", seed=args.seed) for split, dataset in datasets.items()}
    run_dir = args.output_dir / f"run_{started:%Y%m%dT%H%M%S%fZ}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(database=str(database), optimizer="AdamW", deterministic_algorithms=True,
                  torch_version=str(torch.__version__), frame_stride=preprocessing["frame_stride"],
                  loss="BCEWithLogitsLoss, equal weight per sampled match")
    config["lr_scheduler_settings"] = ({"mode": "min", "factor": args.lr_factor,
        "patience": args.lr_patience, "threshold": 0.0, "threshold_mode": "abs", "cooldown": 0,
        "minimum_fraction_of_initial_lr": 0.01, "eps": 0.0,
        "metric": "fixed-prefix validation BCE", "step_timing": "after each validation pass"}
        if args.lr_scheduler == "plateau" else None)
    metadata = {
        "format_version": 1, "architecture": ARCHITECTURE, "conventions": CONVENTIONS,
        "static_preprocessing": source["static_preprocessing"], "preprocessing": preprocessing,
        "split_assignments": source["split_assignments"],
        "active_match_ids": {split: [match["match_id"] for match in group] for split, group in groups.items()},
        "data_source": source.get("data_source", {}), "training_config": config,
        "training_win_prior": training_prior,
        "stage2_source": {"path": str(args.dynamics_checkpoint.resolve()), "kind": source["kind"],
                          "sha256": hashlib.sha256(args.dynamics_checkpoint.read_bytes()).hexdigest(),
                          "best_epoch": source.get("best_epoch")},
    }
    fixed_prefixes = [{"match_id": match["match_id"], "cutoff_time_s": datasets["validation"].cutoff_for(match)}
                      for match in groups["validation"]]
    metadata["validation_prefixes"] = fixed_prefixes
    static.write_json(run_dir / "metadata.json", metadata)
    static.write_json(run_dir / "data_audit.json", audit)
    static.write_splits(run_dir / "split_matches.csv", source["split_assignments"])
    write_rows(run_dir / "validation_prefixes.csv", fixed_prefixes, ["match_id", "cutoff_time_s"])
    model = initialize_model(source, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW([
        {"params": [value for key, value in model.named_parameters() if not key.startswith("win_head.")],
         "lr": args.backbone_learning_rate},
        {"params": model.win_head.parameters(), "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = create_lr_scheduler(optimizer, args)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, stale, history = float("inf"), 0, []
    best_path = run_dir / "best_model.pt"
    log(f"Run: {run_dir}; matches: " + str({key: len(value) for key, value in groups.items()}))
    log(f"Effective batch {args.batch_size}, device micro-batch {args.micro_batch_size}; "
        f"Stage 2 frame stride {preprocessing['frame_stride']}; fixed validation prefixes")
    with (run_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["epoch", "train_bce", "validation_bce", "validation_brier", "prior_bce",
                                         "backbone_learning_rate", "head_learning_rate",
                                         "next_backbone_learning_rate", "next_head_learning_rate"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            rates_used = [group["lr"] for group in optimizer.param_groups]
            model.train()
            datasets["train"].set_epoch(epoch)
            loss_sum, count = 0.0, 0
            for batch_index, batch in enumerate(loaders["train"], start=1):
                optimizer.zero_grad(set_to_none=True)
                for part in dynamics.micro_batches(batch, args.micro_batch_size):
                    logits = dynamics.forward_batch(model, part, args.device)
                    loss = criterion(logits, part["labels"].to(args.device))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training BCE")
                    number = len(part["labels"])
                    (loss * number / len(batch["labels"])).backward()
                    loss_sum += loss.item() * number
                    count += number
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                if batch_index % 20 == 0:
                    log(f"Epoch {epoch}: {batch_index}/{len(loaders['train'])} batches; train BCE {loss_sum / count:.6f}")
            validation, rows = evaluate_prefixes(model, loaders["validation"], args.device)
            baseline = prior_metrics(rows, training_prior)
            if scheduler is not None:
                scheduler.step(validation["bce"])
            next_rates = [group["lr"] for group in optimizer.param_groups]
            row = {"epoch": epoch, "train_bce": loss_sum / count, "validation_bce": validation["bce"],
                   "validation_brier": validation["brier"], "prior_bce": baseline["bce"],
                   "backbone_learning_rate": rates_used[0], "head_learning_rate": rates_used[1],
                   "next_backbone_learning_rate": next_rates[0], "next_head_learning_rate": next_rates[1]}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            log(f"Epoch {epoch}: train BCE {row['train_bce']:.6f}, validation BCE {validation['bce']:.6f}, "
                f"prior BCE {baseline['bce']:.6f}")
            if next_rates != rates_used:
                log(f"Validation plateau: next backbone LR {next_rates[0]:g}, win-head LR {next_rates[1]:g}")
            if validation["bce"] < best_loss:
                best_loss, best_epoch, stale = validation["bce"], epoch, 0
                best_validation, best_baseline, best_rows = validation, baseline, rows
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({**metadata, "kind": "stage3_complete", "best_epoch": best_epoch,
                            "best_validation_bce": best_loss, "learning_rates_at_best": rates_used,
                            "model_state_dict": best_state}, best_path)
            else:
                stale += 1
                if stale >= args.patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    fixed_metrics, fixed_rows = fixed_time_evaluation(model, database, groups["validation"], metadata,
                                                     args.evaluation_times, args.device, args.micro_batch_size)
    fields = ["match_id", "cutoff_time_s", "label", "logit", "team1_win_probability"]
    write_rows(run_dir / "validation_predictions.csv", best_rows, fields)
    write_rows(run_dir / "validation_fixed_time_predictions.csv", fixed_rows, fields + ["requested_time_s"])
    # A bounded CPU prefix provides an independent exact serialization check.
    reference = collate_prefixes([datasets["validation"][0]])
    for name in ("players", "globals", "times", "targets", "target_mask", "padding_mask"):
        reference[name] = reference[name][:, :32].clone()
    reference["cutoff_time_s"] = reference["times"][:, -1].double()
    model.cpu()
    with torch.inference_mode():
        logits = dynamics.forward_batch(model, reference)
    torch.save({"batch": reference, "logits": logits, "probabilities": logits.sigmoid()}, run_dir / "reload_reference.pt")
    verification = verify_reload(best_path)
    plot_results(history, fixed_metrics, run_dir)
    metrics = {"best_epoch": best_epoch, "epochs_completed": len(history),
               "stop_reason": "early_stopping" if stale >= args.patience else "epoch_limit",
               "validation": best_validation, "training_prior": best_baseline,
               "validation_fixed_times": fixed_metrics, "reload_verification": verification,
               "test_evaluated": False, "started_at_utc": started.isoformat(),
               "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    static.write_json(run_dir / "metrics.json", metrics)
    log(f"Saved best epoch {best_epoch}; independent reload reproduces win probabilities exactly.")
    return run_dir


@torch.inference_mode()
def predict(args):
    validate_times([args.through_time])
    model, checkpoint = load_stage3(args.checkpoint, args.device)
    database = dynamics.resolve_database(args)
    matches, _ = load_win_matches(database, [args.match_id])
    if not matches:
        raise ValueError("Recorded match has no valid winning-team label")
    attach_eligible_times(database, matches, checkpoint["preprocessing"]["frame_stride"])
    match = matches[0]
    if args.through_time >= match["end_time_s"] or args.through_time > match["last_observed_time_s"]:
        raise ValueError("Requested time is terminal or beyond recorded replay coverage")
    eligible = [time for time in match["eligible_times"] if time <= args.through_time]
    if not eligible:
        raise ValueError("No eligible observed frame at or before the requested time")
    dataset = PrefixDataset(database, matches, checkpoint["static_preprocessing"], checkpoint["preprocessing"],
                            fixed=True, cutoffs={args.match_id: eligible[-1]})
    logit = dynamics.forward_batch(model, collate_prefixes([dataset[0]]), args.device)[0].item()
    probability = torch.sigmoid(torch.tensor(logit, dtype=torch.float64)).item()
    return {"match_id": args.match_id, "requested_time_s": args.through_time,
            "through_game_time_s": eligible[-1], "logit": logit,
            "team1_win_probability": probability, "team0_win_probability": 1 - probability}


def evaluate_checkpoint(args):
    validate_times(args.evaluation_times)
    model, checkpoint = load_stage3(args.checkpoint, args.device)
    database = dynamics.resolve_database(args)
    ids = checkpoint["active_match_ids"][args.split]
    if not ids:
        raise ValueError(f"Checkpoint has no active {args.split} matches")
    matches, skipped = load_win_matches(database, ids)
    groups = prepare_groups(matches, checkpoint)
    selected = groups[args.split]
    audit = attach_eligible_times(database, selected, checkpoint["preprocessing"]["frame_stride"])
    results, rows = fixed_time_evaluation(model, database, selected, checkpoint, args.evaluation_times,
                                         args.device, args.batch_size)
    directory = args.output_dir or args.checkpoint.parent / f"evaluation_{args.split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    directory.mkdir(parents=True, exist_ok=False)
    static.write_json(directory / "metrics.json", {"split": args.split, "fixed_times": results,
                      "data_audit": audit, "skipped": skipped,
                      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()})
    write_rows(directory / "predictions.csv", rows,
               ["match_id", "cutoff_time_s", "label", "logit", "team1_win_probability", "requested_time_s"])
    plot_results([], results, directory)
    return directory


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Fine-tune on random prefixes; test matches stay held out")
    prediction = commands.add_parser("predict", help="Predict a win probability from a recorded nonterminal prefix")
    evaluation = commands.add_parser("evaluate", help="Evaluate fixed elapsed times on a saved split")
    verification = commands.add_parser("verify", help="Verify exact CPU checkpoint reload")
    for command in (training, prediction, evaluation):
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--database", type=Path, help="Full-match DuckDB override; opened read-only")
        command.add_argument("--device", default="cpu")
    for command in (prediction, evaluation, verification):
        command.add_argument("--checkpoint", type=Path, required=True)
    for command in (training, evaluation):
        command.add_argument("--evaluation-times", type=float, nargs="+", default=[300, 600, 900, 1200, 1800])
    training.add_argument("--dynamics-checkpoint", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, default=ROOT / "data/supplemental/finetune_win")
    training.add_argument("--splits", type=Path, help="Optional shared split CSV consistency check; never rewritten")
    training.add_argument("--match-ids", type=int, nargs="+", help="Subset of the Stage 2 active match IDs")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--validation-seed", type=int, default=42,
                          help="Keep validation cutoffs fixed while varying the training seed")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--micro-batch-size", type=int, default=1, help="Accumulate to preserve effective batch size")
    training.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    training.add_argument("--learning-rate", type=float, default=1e-4, help="New win-head learning rate")
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--lr-scheduler", choices=("none", "plateau"), default="none")
    training.add_argument("--lr-factor", type=float, default=0.5, help="Plateau multiplier for both learning rates")
    training.add_argument("--lr-patience", type=int, default=2,
                          help="Plateau tolerance: 2 reduces rates after 3 non-improving validation epochs")
    training.add_argument("--gradient-clip", type=float, default=1.0)
    prediction.add_argument("--match-id", type=int, required=True)
    prediction.add_argument("--through-time", type=float, required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation.add_argument("--batch-size", type=int, default=1)
    evaluation.add_argument("--output-dir", type=Path)
    verification.add_argument("--reference", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        print(json.dumps(predict(args), allow_nan=False))
    elif args.command == "evaluate":
        print(evaluate_checkpoint(args))
    else:
        print(json.dumps(verify_reload(args.checkpoint, args.reference), allow_nan=False))


if __name__ == "__main__":
    main()
