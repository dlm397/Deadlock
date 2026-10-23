"""Stage 1 of research/model/outline.tex: supervised static player pre-training.

Run `python pretrain_static.py --help` for train, predict, and verify commands.
Only hero identity and initial_rank_badge enter the player encoder.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import tomllib

import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.console_log import log


ROOT = Path(__file__).resolve().parent
PLAYER_COLUMNS = ["match_id", "player_slot", "hero_id", "team_id", "initial_rank_badge"]
SPLITS = ("train", "validation", "test")
ARCHITECTURE = {
    "hero_embedding_dim": 16,
    "player_mlp": [17, 32, 32],
    "win_head": [64, 32, 1],
    "activation": "ReLU between linear layers only",
    "pooling": "mean",
    "players_per_team": 6,
    "team_order": [0, 1],
}


class StaticWinModel(nn.Module):
    """Inputs: hero indices and scaled ranks, both shaped (matches, 2, 6)."""

    def __init__(self, num_heroes):
        super().__init__()
        # Unseen heroes use a fixed zero vector, never a random untrained vector.
        self.hero_embedding = nn.Embedding(num_heroes, 16, padding_idx=0)
        self.player_mlp = nn.Sequential(nn.Linear(17, 32), nn.ReLU(), nn.Linear(32, 32))
        self.win_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, hero_indices, scaled_ranks):
        if hero_indices.ndim != 3 or hero_indices.shape[1:] != (2, 6):
            raise ValueError("Expected hero indices shaped (batch, 2, 6)")
        if scaled_ranks.shape != hero_indices.shape:
            raise ValueError("Ranks must have the same shape as hero indices")
        player_inputs = torch.cat((self.hero_embedding(hero_indices), scaled_ranks.unsqueeze(-1)), dim=-1)
        team_vectors = self.player_mlp(player_inputs).mean(dim=2)
        return self.win_head(team_vectors.flatten(start_dim=1)).squeeze(-1)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rank_value(value):
    """The normalizer maps unknown badge 0 to null; accept both at inference."""
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("initial_rank_badge must be a nonnegative integer or null")
    return value or None


def ordered_players(players):
    """Validate the complete roster and group team 0 then team 1, six each."""
    if len(players) != 12:
        raise ValueError("Expected exactly 12 players")
    slots = [p["player_slot"] for p in players]
    if any(type(s) is not int or s < 0 for s in slots) or len(set(slots)) != 12:
        raise ValueError("Expected 12 distinct nonnegative integer player slots")
    for player in players:
        if type(player["team_id"]) is not int or player["team_id"] not in (0, 1):
            raise ValueError("team_id must be 0 or 1")
        if type(player["hero_id"]) is not int or player["hero_id"] <= 0:
            raise ValueError("hero_id must be a positive integer")
        rank_value(player.get("initial_rank_badge"))
    if sum(p["team_id"] == 0 for p in players) != 6:
        raise ValueError("Expected six players on each team")
    return sorted(players, key=lambda p: (p["team_id"], p["player_slot"]))


def read_players(data_dir, match_id):
    # Explicit column selection prevents terminal statistics from becoming inputs.
    players = pq.read_table(data_dir / "end_match" / f"{match_id}.parquet", columns=PLAYER_COLUMNS).to_pylist()
    if any(p["match_id"] != match_id for p in players):
        raise ValueError(f"Match {match_id}: player match IDs disagree with filename")
    return ordered_players(players)


def load_matches(data_dir):
    matches, skipped = [], []
    for path in sorted((data_dir / "end_match").glob("*.parquet")):
        match_id = int(path.stem)
        header = data_dir / "matches" / path.name
        if not header.exists():
            skipped.append({"match_id": match_id, "reason": "missing match metadata"})
            continue
        rows = pq.read_table(header, columns=["match_id", "match_outcome", "winning_team_id"]).to_pylist()
        if len(rows) != 1 or rows[0]["match_id"] != match_id:
            raise ValueError(f"Match {match_id}: expected one matching metadata row")
        outcome = rows[0]
        if outcome["match_outcome"] != "TeamWin" or outcome["winning_team_id"] not in (0, 1):
            skipped.append({"match_id": match_id, "reason": "no verified TeamWin label"})
            continue
        try:
            players = read_players(data_dir, match_id)
        except ValueError as error:
            raise ValueError(f"Match {match_id}: {error}") from error
        matches.append({"match_id": match_id, "players": players, "label": float(outcome["winning_team_id"] == 1)})
    if not matches:
        raise ValueError(f"No labeled matches found under {data_dir}")
    if len({m["match_id"] for m in matches}) != len(matches):
        raise ValueError("Duplicate match IDs")
    return matches, skipped


def write_splits(path, assignments, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["match_id", "split"])
        writer.writerows((mid, assignments[mid]) for mid in sorted(assignments, key=int))


def get_splits(match_ids, path, seed, validation_fraction, test_fraction):
    """Existing assignments are authoritative, including IDs absent in this stage."""
    if path.exists():
        assignments = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                mid = str(int(row["match_id"]))
                if mid in assignments or row["split"] not in SPLITS:
                    raise ValueError(f"Invalid or duplicate split assignment for match {mid}")
                assignments[mid] = row["split"]
        missing = set(map(str, match_ids)) - assignments.keys()
        if missing:
            raise ValueError(f"{path} lacks {len(missing)} match IDs. Extend it explicitly; existing splits must be preserved.")
    else:
        ids = sorted(match_ids)
        random.Random(seed).shuffle(ids)
        n_validation = max(1, int(len(ids) * validation_fraction))
        n_test = max(1, int(len(ids) * test_fraction))
        if n_validation + n_test >= len(ids):
            raise ValueError("Need enough matches for nonempty training, validation, and test splits")
        assignments = {str(mid): ("validation" if i < n_validation else
                                 "test" if i < n_validation + n_test else "train") for i, mid in enumerate(ids)}
        # Exclusive creation prevents accidentally overwriting another run's splits.
        write_splits(path, assignments, mode="x")
    if any(not any(assignments[str(mid)] == split for mid in match_ids) for split in SPLITS):
        raise ValueError("The current dataset must contain matches in all three splits")
    return assignments


def fit_preprocessing(training_matches):
    players = [p for match in training_matches for p in match["players"]]
    heroes = sorted({p["hero_id"] for p in players})
    known_ranks = [rank_value(p.get("initial_rank_badge")) for p in players]
    known_ranks = [rank for rank in known_ranks if rank is not None]
    return {
        "hero_id_to_index": {str(hero): index for index, hero in enumerate(heroes, start=1)},
        "unknown_hero_index": 0,
        "unknown_hero_policy": "fixed zero embedding for IDs not seen in training",
        "rank": {
            "field": "initial_rank_badge",
            "method": "training median imputation, training min-max scaling, clip to [0, 1]",
            "missing_values": [None, 0],
            "median": float(statistics.median(known_ranks)) if known_ranks else 0.0,
            "min": float(min(known_ranks)) if known_ranks else 0.0,
            "max": float(max(known_ranks)) if known_ranks else 0.0,
            "known_training_players": len(known_ranks),
            "constant_or_all_missing_policy": "zero rank channel for every split and inference",
        },
    }


def encode_rosters(rosters, preprocessing):
    hero_indices, scaled_ranks = [], []
    scaling = preprocessing["rank"]
    span = scaling["max"] - scaling["min"]
    for roster in rosters:
        heroes, ranks = [], []
        for player in ordered_players(roster):
            heroes.append(preprocessing["hero_id_to_index"].get(str(player["hero_id"]), 0))
            rank = rank_value(player.get("initial_rank_badge"))
            value = scaling["median"] if rank is None else rank
            ranks.append(min(1.0, max(0.0, (value - scaling["min"]) / span)) if span else 0.0)
        hero_indices.append(heroes)
        scaled_ranks.append(ranks)
    return (torch.tensor(hero_indices, dtype=torch.long).reshape(-1, 2, 6),
            torch.tensor(scaled_ranks, dtype=torch.float32).reshape(-1, 2, 6))


def rank_report(matches):
    ranks = [rank_value(p.get("initial_rank_badge")) for m in matches for p in m["players"]]
    missing = sum(rank is None for rank in ranks)
    return {"matches": len(matches), "players": len(ranks), "missing_players": missing,
            "missing_fraction": missing / len(ranks),
            "all_ranks_missing_matches": sum(all(rank_value(p.get("initial_rank_badge")) is None
                                                   for p in m["players"]) for m in matches)}


def load_stage1(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["kind"] != "stage1_complete" or checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("Expected a compatible complete Stage 1 checkpoint")
    model = StaticWinModel(len(checkpoint["preprocessing"]["hero_id_to_index"]) + 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def win_probabilities(model, rosters, preprocessing):
    heroes, ranks = encode_rosters(rosters, preprocessing)
    device = next(model.parameters()).device
    model.eval()
    return model(heroes.to(device), ranks.to(device)).sigmoid().cpu()


@torch.inference_mode()
def validation_bce(model, loader, device):
    model.eval()
    loss_sum = 0.0
    for heroes, ranks, labels in loader:
        logits = model(heroes.to(device), ranks.to(device))
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="sum").item()
    return loss_sum / len(loader.dataset)


def verify_reload(checkpoint_path, reference_path=None):
    """Compare independent CPU reload with predictions recorded from the best model."""
    reference_path = reference_path or checkpoint_path.parent / "reload_reference.pt"
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    model, _ = load_stage1(checkpoint_path)
    with torch.inference_mode():
        logits = model(reference["hero_indices"], reference["scaled_ranks"])
    torch.testing.assert_close(logits, reference["logits"], rtol=0, atol=0)
    torch.testing.assert_close(logits.sigmoid(), reference["probabilities"], rtol=0, atol=0)
    return {"device": "cpu", "matches": len(logits), "max_absolute_error": 0.0, "passed": True}


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot([r["epoch"] for r in history], [r["train_bce"] for r in history], label="Training")
    axis.plot([r["epoch"] for r in history], [r["validation_bce"] for r in history], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Stage 1: static embedding pre-training")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(args):
    if not 1 <= args.epochs <= 100 or not 1 <= args.patience <= 10 or args.batch_size < 1:
        raise ValueError("Require epochs 1..100, patience 1..10, and positive batch size")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1
            and args.validation_fraction + args.test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    data_dir = resolve_data_dir(args)
    log(f"Reading static inputs from {data_dir}")
    matches, skipped = load_matches(data_dir)
    split_path = (args.splits or data_dir / "splits.csv").resolve()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    reused_splits = split_path.exists()
    assignments = get_splits([m["match_id"] for m in matches], split_path, split_seed,
                             args.validation_fraction, args.test_fraction)
    groups = {split: [m for m in matches if assignments[str(m["match_id"])] == split] for split in SPLITS}
    preprocessing = fit_preprocessing(groups["train"])
    datasets = {}
    for split in ("train", "validation"):
        heroes, ranks = encode_rosters([m["players"] for m in groups[split]], preprocessing)
        datasets[split] = TensorDataset(heroes, ranks, torch.tensor([m["label"] for m in groups[split]]))
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False)
    run_dir = args.output_dir / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(data_dir=str(data_dir), splits=str(split_path), effective_split_seed=split_seed,
                  reused_splits=reused_splits, optimizer="AdamW", weight_decay=0.0,
                  loss="BCEWithLogitsLoss", deterministic_algorithms=True, torch_version=str(torch.__version__))
    metadata = {
        "format_version": 1, "architecture": ARCHITECTURE, "preprocessing": preprocessing,
        "conventions": {"team_order": [0, 1], "positive_label": "winning_team_id == 1",
                        "probability": "P(canonical team_id 1 wins)",
                        "inputs": ["hero_id", "initial_rank_badge"],
                        "roster_keys_only": ["match_id", "player_slot", "team_id"],
                        "label_source": "matches.winning_team_id with match_outcome == TeamWin"},
        "split_assignments": assignments,
        "active_match_ids": {split: [m["match_id"] for m in groups[split]] for split in SPLITS},
        "training_config": config,
    }
    audit = {"rank_availability": {split: rank_report(groups[split]) for split in SPLITS}, "skipped": skipped}
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "data_audit.json", audit)
    write_splits(run_dir / "split_matches.csv", assignments)
    model = StaticWinModel(len(preprocessing["hero_id_to_index"]) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, stale_epochs, history = float("inf"), 0, []
    log(f"Run: {run_dir}; splits: " + str({key: len(value) for key, value in groups.items()}))
    log(f"Training rank availability: {audit['rank_availability']['train']}")
    best_path = run_dir / "best_model.pt"
    with (run_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["epoch", "train_bce", "validation_bce"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for heroes, ranks, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(heroes.to(device), ranks.to(device))
                loss = criterion(logits, labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
            validation_loss = validation_bce(model, validation_loader, device)
            if not math.isfinite(validation_loss):
                raise ValueError("Nonfinite validation loss")
            row = {"epoch": epoch, "train_bce": loss_sum / len(datasets["train"]), "validation_bce": validation_loss}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            log(f"Epoch {epoch}: train BCE {row['train_bce']:.6f}, validation BCE {validation_loss:.6f}")
            if validation_loss < best_loss:
                best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({**metadata, "kind": "stage1_complete", "best_epoch": best_epoch,
                            "best_validation_bce": best_loss, "model_state_dict": best_state}, best_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break
    # Restore the best state, not the final epoch, before exporting and checking.
    model.cpu().load_state_dict(best_state)
    model.eval()
    torch.save({**metadata, "kind": "stage1_transfer", "best_epoch": best_epoch,
                "best_validation_bce": best_loss,
                "hero_embedding_state_dict": model.hero_embedding.state_dict(),
                "player_mlp_state_dict": model.player_mlp.state_dict()}, run_dir / "static_encoder.pt")
    heroes, ranks, _ = datasets["validation"].tensors
    with torch.inference_mode():
        logits = model(heroes, ranks)
    torch.save({"hero_indices": heroes, "scaled_ranks": ranks, "logits": logits,
                "probabilities": logits.sigmoid()}, run_dir / "reload_reference.pt")
    verification = verify_reload(best_path)
    write_json(run_dir / "metrics.json", {"best_epoch": best_epoch, "best_validation_bce": best_loss,
                                          "epochs_completed": len(history), "reload_verification": verification})
    plot_history(history, run_dir / "loss_curve.png")
    log(f"Saved best epoch {best_epoch}; independent reload reproduces validation predictions exactly.")
    return run_dir


def resolve_data_dir(args):
    if args.data_dir is not None:
        return args.data_dir.resolve()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    return (args.config.parent / config["output_dir"]).resolve()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train Stage 1 (only when explicitly invoked)")
    prediction = commands.add_parser("predict", help="Reload and report team 1 win probability")
    verification = commands.add_parser("verify", help="Verify saved evaluation predictions after reload")
    for command in (training, prediction):
        command.add_argument("--config", type=Path, default=ROOT / "config.toml")
        command.add_argument("--data-dir", type=Path, help="Override config.toml output_dir")
        command.add_argument("--device", default="cpu")
    training.add_argument("--output-dir", type=Path, default=ROOT / "research" / "outputs" / "pretrain_static")
    training.add_argument("--splits", type=Path, help="Existing match_id,split CSV; default: <data-dir>/splits.csv")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--split-seed", type=int, help="Defaults to seed; ignored when reusing splits")
    training.add_argument("--validation-fraction", type=float, default=0.1)
    training.add_argument("--test-fraction", type=float, default=0.1)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    prediction.add_argument("--checkpoint", type=Path, required=True)
    source = prediction.add_mutually_exclusive_group(required=True)
    source.add_argument("--match-id", type=int, help="Read just the static fields of a collected match")
    source.add_argument("--players", type=Path, help="JSON array of 12 players, no outcome required")
    verification.add_argument("--checkpoint", type=Path, required=True)
    verification.add_argument("--reference", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        model, checkpoint = load_stage1(args.checkpoint, args.device)
        players = (json.loads(args.players.read_text(encoding="utf-8")) if args.players
                   else read_players(resolve_data_dir(args), args.match_id))
        probability = win_probabilities(model, [players], checkpoint["preprocessing"]).item()
        print(json.dumps({"team_1_win_probability": probability, "team_0_win_probability": 1 - probability}))
    else:
        print(json.dumps(verify_reload(args.checkpoint, args.reference)))


if __name__ == "__main__":
    main()
