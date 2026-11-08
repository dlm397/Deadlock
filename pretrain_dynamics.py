"""Stage 2: causal next-frame change in Team1-minus-Team0 soul wealth.

Reads full_match.duckdb and transfers the Stage 1 hero/player encoder.
Run --help for train, predict, and verify. No collection or training on import.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import os
import tempfile

import duckdb
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from app.config import DEFAULT_CONFIG, data_paths, model_output_dir, resolve_saved_path
from app.console_log import log
import pretrain_static as static


ROOT = Path(__file__).resolve().parent
PLAYER_FEATURES = (
    "net_worth", "unspent_souls", "kills", "deaths", "assists", "last_hits",
    "denies", "level", "health", "max_health", "x", "y", "z", "hero_damage",
    "objective_damage", "hero_healing", "self_healing", "alive",
)
ARCHITECTURE = {
    "static": static.ARCHITECTURE, "model_dim": 64,
    "player_features": list(PLAYER_FEATURES),
    "player_channels": 2 * len(PLAYER_FEATURES) + 2,
    "player_channel_order": "standardized values, per-field masks, team_id, observation mask",
    "global_features": ["elapsed_time_s"],
    "spatial_layers": 1, "temporal_layers": 2, "attention_heads": 4,
    "feedforward_dim": 128, "pooling": "six-player mean, team order [0, 1]",
    "temporal_encoding": "sinusoidal actual elapsed seconds, base 10000",
    "dynamics_head": [64, 32, 1], "transformer_activation": "ReLU",
}
TARGET = {
    "name": "soul_lead_change", "source": "player_snapshots.net_worth",
    "formula": "(sum_team1(t+1)-sum_team0(t+1)) - (sum_team1(t)-sum_team0(t))",
    "units": "souls", "persistence": "zero change in team soul lead",
    "validity": "all twelve net-worth observations valid at both adjacent selected frames",
}


def open_database(database):
    if not Path(database).is_file():
        raise FileNotFoundError(f"No full-match database at {database}; ingest retained replays first")
    return duckdb.connect(str(database), read_only=True)


def load_matches(database, match_ids=None):
    """Read only static identities and successful-source provenance, never outcomes."""
    with open_database(database) as connection:
        if connection.execute("SELECT version FROM schema_info").fetchall() != [(1,)]:
            raise ValueError("Expected full-match schema version 1")
        requested = list(dict.fromkeys(match_ids or []))
        where = "WHERE m.match_id IN (SELECT unnest(?::UBIGINT[]))" if requested else ""
        headers = connection.execute(f"""
            SELECT m.match_id, CAST(m.successful_attempt_id AS VARCHAR), a.sampling_interval_s,
                   a.parser_revision, a.metadata_sha256
            FROM matches m JOIN processing_attempts a ON m.successful_attempt_id = a.attempt_id
            {where} ORDER BY m.match_id
        """, [requested] if requested else []).fetchall()
        if requested and set(requested) != {row[0] for row in headers}:
            raise ValueError("Some requested match IDs have no successful full-match data")
        roster_rows = connection.execute(f"""
            SELECT p.match_id, p.player_slot, p.hero_id, p.team_id, p.initial_rank_badge
            FROM match_players p JOIN matches m USING(match_id) {where}
            ORDER BY p.match_id, p.team_id, p.player_slot
        """, [requested] if requested else []).fetchall()
    rosters = {}
    for row in roster_rows:
        rosters.setdefault(row[0], []).append(dict(zip(static.PLAYER_COLUMNS, row)))
    matches = []
    for mid, attempt, interval, parser, metadata in headers:
        players = static.ordered_players(rosters.get(mid, []))
        matches.append({"match_id": mid, "players": players,
                        "provenance": {"successful_attempt_id": attempt, "sampling_interval_s": interval,
                                       "parser_revision": parser, "metadata_sha256": metadata}})
    if not matches:
        raise ValueError("No successful full-match sequences found")
    return matches, []


def load_sequence(connection, match, frame_stride=1, through_time=None):
    """Preserve missing frames; future rows are never used to fill current values."""
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    cutoff = "AND game_time_s <= ?" if through_time is not None else ""
    parameters = [match["match_id"]] + ([through_time] if through_time is not None else [])
    columns = "player_slot, game_time_s, observed_time_s, " + ", ".join(PLAYER_FEATURES)
    data = connection.execute(f"SELECT {columns} FROM player_snapshots WHERE match_id = ? {cutoff} "
                              "ORDER BY game_time_s, player_slot", parameters).fetchnumpy()
    count = len(data["player_slot"])
    if not count or count % 12:
        raise ValueError(f"Match {match['match_id']}: expected twelve rows at every frame")
    slots = np.asarray(data["player_slot"]).reshape(-1, 12)
    sorted_slots = sorted(p["player_slot"] for p in match["players"])
    times = np.asarray(data["game_time_s"], dtype=np.float64).reshape(-1, 12)
    if (not np.all(slots == sorted_slots) or not np.isfinite(times).all()
            or (times < 0).any() or not np.all(times == times[:, :1])
            or (np.diff(times[:, 0]) <= 0).any()):
        raise ValueError(f"Match {match['match_id']}: inconsistent roster or frame timestamps")
    # Database NULLs become NaN only in these model-local arrays.
    def numeric(name):
        return np.ma.asarray(data[name], dtype=np.float64).filled(np.nan).reshape(-1, 12)

    observed_times = numeric("observed_time_s")
    observed = np.isfinite(observed_times)
    if np.any(observed & ((observed_times < 0) | (observed_times > times))):
        raise ValueError(f"Match {match['match_id']}: future or negative observation time")
    values = np.stack([numeric(name) for name in PLAYER_FEATURES], axis=-1)
    values[~np.isfinite(values)] = np.nan
    values[~observed] = np.nan
    order = [sorted_slots.index(p["player_slot"]) for p in match["players"]]
    # A stride is explicit downsampling, not a fixed-length crop. Never force the
    # trajectory's final row into a prefix: that would make prefix sampling differ.
    times = times[::frame_stride, 0].copy()
    values = values[::frame_stride][:, order].copy()
    observed = observed[::frame_stride][:, order].copy()
    wealth = values[:, :, PLAYER_FEATURES.index("net_worth")]
    valid_lead = np.isfinite(wealth).all(axis=1)
    lead = wealth[:, 6:].sum(axis=1) - wealth[:, :6].sum(axis=1)
    targets = np.zeros(len(times), dtype=np.float64)
    target_mask = np.zeros(len(times), dtype=bool)
    target_mask[:-1] = valid_lead[:-1] & valid_lead[1:]
    targets[:-1] = np.where(target_mask[:-1], np.diff(lead), 0.0)
    return {"times": times, "player_values": values, "observed": observed,
            "targets": targets, "target_mask": target_mask}


def update_moments(moments, values):
    """Merge per-column population moments without retaining the collection."""
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    mean = np.where(valid, values, 0).sum(axis=0) / np.maximum(count, 1)
    m2 = np.where(valid, values - mean, 0).__pow__(2).sum(axis=0)
    total = moments["count"] + count
    delta = mean - moments["mean"]
    moments["m2"] += m2 + delta**2 * moments["count"] * count / np.maximum(total, 1)
    moments["mean"] += delta * count / np.maximum(total, 1)
    moments["count"] = total


def fit_preprocessing(database, matches, frame_stride=1):
    moments = {name: {"count": np.zeros(width), "mean": np.zeros(width), "m2": np.zeros(width)}
               for name, width in (("player", len(PLAYER_FEATURES)), ("global", 1), ("target", 1))}
    usable, skipped, lengths = [], [], []
    with open_database(database) as connection:
        for index, match in enumerate(matches):
            sequence = load_sequence(connection, match, frame_stride)
            if not sequence["target_mask"].any():
                skipped.append(match["match_id"])
                continue
            usable.append(match["match_id"])
            lengths.append(len(sequence["times"]))
            update_moments(moments["player"], sequence["player_values"].reshape(-1, len(PLAYER_FEATURES)))
            update_moments(moments["global"], sequence["times"][:, None])
            update_moments(moments["target"], sequence["targets"][sequence["target_mask"]][:, None])
            if (index + 1) % 100 == 0:
                log(f"Fitting training preprocessing: {index + 1}/{len(matches)} matches")
    if not usable:
        raise ValueError("No valid adjacent soul-change targets in the training split")
    result = {"fit_scope": "training matches only", "missing_inputs": "mean imputation plus explicit masks",
              "frame_stride": frame_stride, "usable_training_ids": usable, "skipped_training_ids": skipped,
              "max_training_frames": max(lengths)}
    for name, value in moments.items():
        std = np.sqrt(value["m2"] / np.maximum(value["count"], 1))
        scale = np.where(std > 1e-12, std, 1.0)
        result[name] = {"mean": value["mean"].tolist(), "scale": scale.tolist(),
                        "count": value["count"].astype(int).tolist()}
    return result


def encode_sequence(sequence, match, static_preprocessing, preprocessing):
    heroes, ranks = static.encode_rosters([match["players"]], static_preprocessing)
    values = sequence["player_values"]
    valid = np.isfinite(values)
    scaling = preprocessing["player"]
    scaled = np.where(valid, (values - np.array(scaling["mean"])) / np.array(scaling["scale"]), 0)
    teams = np.broadcast_to(np.array([p["team_id"] for p in match["players"]])[None, :, None],
                            (len(values), 12, 1))
    features = np.concatenate((scaled, valid, teams, sequence["observed"][..., None]), axis=-1)
    globals_ = (sequence["times"][:, None] - preprocessing["global"]["mean"]) / preprocessing["global"]["scale"]
    targets = (sequence["targets"] - preprocessing["target"]["mean"][0]) / preprocessing["target"]["scale"][0]
    return {"match_id": match["match_id"], "heroes": heroes[0], "ranks": ranks[0],
            "players": torch.tensor(features, dtype=torch.float32),
            "globals": torch.tensor(globals_, dtype=torch.float32),
            "times": torch.tensor(sequence["times"], dtype=torch.float32),
            "targets": torch.tensor(np.where(sequence["target_mask"], targets, 0), dtype=torch.float32),
            "target_mask": torch.tensor(sequence["target_mask"], dtype=torch.bool)}


class SequenceDataset(Dataset):
    def __init__(self, database, matches, static_preprocessing, preprocessing):
        self.database, self.matches = database, matches
        self.static_preprocessing, self.preprocessing = static_preprocessing, preprocessing

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        match = self.matches[index]
        with open_database(self.database) as connection:
            sequence = load_sequence(connection, match, self.preprocessing["frame_stride"])
        return encode_sequence(sequence, match, self.static_preprocessing, self.preprocessing)


def collate_sequences(items):
    lengths = torch.tensor([len(item["times"]) for item in items])
    batch = {name: nn.utils.rnn.pad_sequence([item[name] for item in items], batch_first=True)
             for name in ("players", "globals", "times", "targets", "target_mask")}
    batch.update(heroes=torch.stack([item["heroes"] for item in items]),
                 ranks=torch.stack([item["ranks"] for item in items]),
                 match_id=[item["match_id"] for item in items],
                 padding_mask=torch.arange(int(lengths.max()))[None, :] >= lengths[:, None])
    return batch


class DynamicsModel(nn.Module):
    def __init__(self, num_heroes, dropout=0.1):
        super().__init__()
        encoder = static.StaticWinModel(num_heroes)
        self.hero_embedding = encoder.hero_embedding
        self.player_mlp = encoder.player_mlp
        self.player_projection = nn.Linear(32 + ARCHITECTURE["player_channels"], 64)
        spatial_layer = nn.TransformerEncoderLayer(64, 4, 128, dropout, batch_first=True)
        temporal_layer = nn.TransformerEncoderLayer(64, 4, 128, dropout, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, 1, enable_nested_tensor=False)
        self.frame_projection = nn.Linear(128 + 1, 64)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, 2, enable_nested_tensor=False)
        # Encoder clones otherwise start with identical layer weights.
        for layer in self.temporal_transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.dynamics_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode_frames(self, hero_indices, scaled_ranks, player_features, global_features, times, padding_mask):
        """Shared causal backbone used by dynamics pre-training and win fine-tuning."""
        batch, frames, players, width = player_features.shape
        if (players != 12 or width != ARCHITECTURE["player_channels"]
                or hero_indices.shape != (batch, 2, 6) or scaled_ranks.shape != hero_indices.shape
                or global_features.shape != (batch, frames, 1) or times.shape != (batch, frames)
                or padding_mask.shape != times.shape or padding_mask.dtype != torch.bool):
            raise ValueError("Invalid Stage 2 tensor shapes or padding mask")
        if padding_mask.all(dim=1).any() or (padding_mask[:, :-1] & ~padding_mask[:, 1:]).any():
            raise ValueError("Each sequence needs observations followed only by right padding")
        static_inputs = torch.cat((self.hero_embedding(hero_indices), scaled_ranks.unsqueeze(-1)), dim=-1)
        static_players = self.player_mlp(static_inputs).reshape(batch, 1, 12, 32).expand(-1, frames, -1, -1)
        # There are always twelve roster tokens. Missing states retain their static
        # identity and explicit observation masks; only absent frames are padding.
        tokens = self.player_projection(torch.cat((static_players, player_features), dim=-1))
        spatial = self.spatial_transformer(tokens.reshape(batch * frames, 12, 64))
        teams = spatial.reshape(batch, frames, 2, 6, 64).mean(dim=3).flatten(start_dim=2)
        tokens = self.frame_projection(torch.cat((teams, global_features), dim=-1))
        frequencies = torch.exp(torch.arange(0, 64, 2, device=times.device, dtype=times.dtype)
                                * (-math.log(10000.0) / 64))
        angles = times.unsqueeze(-1) * frequencies
        position = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=2)
        tokens = tokens + position
        causal_mask = torch.ones(frames, frames, dtype=torch.bool, device=tokens.device).triu(1)
        # Right padding is always later than every valid query, so the causal
        # mask already excludes its keys. Avoid merging a separate padding mask
        # into a dense (batch, heads, time, time) tensor on the GPU.
        temporal = self.temporal_transformer(tokens, mask=causal_mask, is_causal=True)
        return temporal.masked_fill(padding_mask.unsqueeze(-1), 0)

    def forward(self, hero_indices, scaled_ranks, player_features, global_features, times, padding_mask):
        temporal = self.encode_frames(hero_indices, scaled_ranks, player_features, global_features, times, padding_mask)
        return self.dynamics_head(temporal).squeeze(-1).masked_fill(padding_mask, 0)


def load_static_checkpoint(path):
    path = resolve_saved_path(path, base=Path.cwd())
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("kind") not in ("stage1_transfer", "stage1_complete")
            or checkpoint.get("architecture") != static.ARCHITECTURE):
        raise ValueError("Expected a compatible Stage 1 static_encoder.pt or best_model.pt")
    if not checkpoint.get("split_assignments"):
        raise ValueError("Stage 1 checkpoint has no split assignments; cannot ensure held-out separation")
    return checkpoint


def initialize_model(checkpoint, dropout=0.1):
    model = DynamicsModel(len(checkpoint["preprocessing"]["hero_id_to_index"]) + 1, dropout)
    if checkpoint["kind"] == "stage1_transfer":
        model.hero_embedding.load_state_dict(checkpoint["hero_embedding_state_dict"])
        model.player_mlp.load_state_dict(checkpoint["player_mlp_state_dict"])
    else:
        for name in ("hero_embedding", "player_mlp"):
            weights = {key.removeprefix(name + "."): value for key, value in checkpoint["model_state_dict"].items()
                       if key.startswith(name + ".")}
            getattr(model, name).load_state_dict(weights)
    return model


def prepare_splits(match_ids, path, checkpoint, seed, validation_fraction=.1, test_fraction=.1, extend=False):
    """Stage 1 assignments remain authoritative even when the shared CSV is absent."""
    saved = checkpoint["split_assignments"]
    if any(group not in static.SPLITS for group in saved.values()):
        raise ValueError("Invalid split assignments in Stage 1 checkpoint")
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                mid = str(int(row["match_id"]))
                if mid in saved and row["split"] != saved[mid]:
                    raise ValueError(f"Split conflict with Stage 1 for match {mid}; refusing cross-stage leakage")
        # Missing saved IDs must be restored with their old assignments, not randomized.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            present = {str(int(row["match_id"])) for row in csv.DictReader(stream)}
        missing_saved = set(saved) - present
        if missing_saved:
            raise ValueError(f"Split manifest omits {len(missing_saved)} Stage 1 assignments; "
                             "use its split_matches.csv or restore the complete shared manifest")
        return static.get_splits(match_ids, path, seed, validation_fraction, test_fraction, extend=extend)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        static.write_splits(temporary, saved)
        assignments = static.get_splits(match_ids, temporary, seed, validation_fraction, test_fraction, extend=extend)
        static.write_splits(path, assignments, mode="x")
        return assignments
    finally:
        temporary.unlink(missing_ok=True)


def forward_batch(model, batch, device="cpu"):
    return model(*(batch[key].to(device) for key in
                   ("heroes", "ranks", "players", "globals", "times", "padding_mask")))


def micro_batches(batch, size):
    """Trim padding within each device batch; keep the effective match batch intact."""
    for start in range(0, len(batch["heroes"]), size):
        part = {key: value[start:start + size] for key, value in batch.items()}
        frames = int((~part["padding_mask"]).sum(dim=1).max())
        for key in ("players", "globals", "times", "targets", "target_mask", "padding_mask"):
            part[key] = part[key][:, :frames]
        yield part


def match_mse(predictions, targets, mask):
    """Equal weight per match, regardless of sequence length or valid target count."""
    counts = mask.sum(dim=1)
    active = counts > 0
    if not active.any():
        raise ValueError("Batch contains no valid next-frame targets")
    errors = torch.where(mask, predictions - targets, 0).square()
    return (errors.sum(dim=1)[active] / counts[active]).mean()


def persistence_prediction(preprocessing):
    # Zero raw change is generally NOT zero in standardized target coordinates.
    return -preprocessing["target"]["mean"][0] / preprocessing["target"]["scale"][0]


@torch.inference_mode()
def evaluate(model, loader, preprocessing, device="cpu"):
    model.eval()
    model_sum = baseline_sum = 0.0
    matches = valid_targets = 0
    for batch in loader:
        predictions = forward_batch(model, batch, device)
        targets, mask = batch["targets"].to(device), batch["target_mask"].to(device)
        mask = mask & ~batch["padding_mask"].to(device)
        active = int(mask.any(dim=1).sum())
        if not active:
            continue
        baseline = torch.full_like(predictions, persistence_prediction(preprocessing))
        model_sum += match_mse(predictions, targets, mask).item() * active
        baseline_sum += match_mse(baseline, targets, mask).item() * active
        matches += active
        valid_targets += int(mask.sum())
    if not matches:
        raise ValueError("Validation has no valid adjacent soul-change targets")
    scale = preprocessing["target"]["scale"][0]
    return {"mse": model_sum / matches, "persistence_mse": baseline_sum / matches,
            "rmse_souls": math.sqrt(model_sum / matches) * scale,
            "persistence_rmse_souls": math.sqrt(baseline_sum / matches) * scale,
            "matches": matches, "valid_targets": valid_targets}


def load_stage2(path, device="cpu"):
    path = resolve_saved_path(path, base=Path.cwd())
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "stage2_complete" or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected a compatible complete Stage 2 checkpoint")
    model = DynamicsModel(len(checkpoint["static_preprocessing"]["hero_id_to_index"]) + 1,
                          checkpoint["training_config"]["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def verify_reload(path, reference_path=None):
    path = resolve_saved_path(path, base=Path.cwd())
    if reference_path is not None:
        reference_path = resolve_saved_path(reference_path, base=Path.cwd())
    reference = torch.load(reference_path or path.parent / "reload_reference.pt", weights_only=True, map_location="cpu")
    model, _ = load_stage2(path)
    with torch.inference_mode():
        prediction = forward_batch(model, reference["batch"])
    torch.testing.assert_close(prediction, reference["prediction"], rtol=0, atol=0)
    return {"passed": True, "device": "cpu", "matches": len(reference["batch"]["heroes"]),
            "max_absolute_error": 0.0}


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = [row["epoch"] for row in history]
    for field, label in (("train_mse", "Training"), ("validation_mse", "Validation"),
                         ("persistence_mse", "Validation persistence")):
        axis.plot(epochs, [row[field] for row in history], label=label)
    axis.set(xlabel="Epoch", ylabel="Standardized MSE (equal weight per match)", title="Stage 2: soul-lead dynamics")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def audit_sequences(database, groups, preprocessing):
    audit = {"target": TARGET, "splits": {}}
    with open_database(database) as connection:
        # Test identities are retained; their observations and errors stay held out.
        audit["splits"]["test"] = {"matches": len(groups["test"]), "observations_inspected": False}
        for split in ("train", "validation"):
            report = {"matches": len(groups[split]), "frames": 0, "valid_targets": 0,
                      "missing_player_observations": 0, "matches_without_targets": [], "max_frames": 0}
            for match in groups[split]:
                sequence = load_sequence(connection, match, preprocessing["frame_stride"])
                count = int(sequence["target_mask"].sum())
                report["frames"] += len(sequence["times"])
                report["valid_targets"] += count
                report["max_frames"] = max(report["max_frames"], len(sequence["times"]))
                report["missing_player_observations"] += int((~sequence["observed"]).sum())
                if not count:
                    report["matches_without_targets"].append(match["match_id"])
            if not report["valid_targets"]:
                raise ValueError(f"{split} has no valid adjacent soul-change targets")
            audit["splits"][split] = report
    return audit


def train(args):
    args.output_dir = model_output_dir(args, "pretrain_dynamics")
    args.static_checkpoint = resolve_saved_path(args.static_checkpoint, args.config, base=Path.cwd())
    if not 1 <= args.epochs <= 100 or not 1 <= args.patience <= 10 or args.batch_size < 1:
        raise ValueError("Require epochs 1..100, patience 1..10, and positive batch size")
    if args.frame_stride < 1 or not 0 <= args.dropout < 1:
        raise ValueError("Require a positive frame stride and dropout in [0, 1)")
    micro_batch_size = args.micro_batch_size if args.micro_batch_size is not None else args.batch_size
    if not 1 <= micro_batch_size <= args.batch_size:
        raise ValueError("micro-batch-size must be between 1 and batch-size")
    for name in ("learning_rate", "transferred_learning_rate", "gradient_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1
            and args.validation_fraction + args.test_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    checkpoint = load_static_checkpoint(args.static_checkpoint)
    database = resolve_database(args)
    log(f"Reading full-match identities from {database}")
    matches, skipped = load_matches(database, args.match_ids)
    split_path = resolve_split_path(args)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    assignments = prepare_splits([match["match_id"] for match in matches], split_path, checkpoint,
                                 split_seed, args.validation_fraction, args.test_fraction, args.extend_splits)
    groups = {split: [match for match in matches if assignments[str(match["match_id"])] == split]
              for split in static.SPLITS}
    preprocessing = fit_preprocessing(database, groups["train"], args.frame_stride)
    audit = audit_sequences(database, groups, preprocessing)
    audit.update(skipped=skipped, rank_availability={split: static.rank_report(group) for split, group in groups.items()})
    loaders = {}
    for split in ("train", "validation"):
        excluded = set(audit["splits"][split]["matches_without_targets"])
        usable = [match for match in groups[split] if match["match_id"] not in excluded]
        dataset = SequenceDataset(database, usable, checkpoint["preprocessing"], preprocessing)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size if split == "train" else micro_batch_size,
                                    shuffle=split == "train",
                                    collate_fn=collate_sequences, num_workers=0,
                                    generator=torch.Generator().manual_seed(args.seed))
    run_dir = args.output_dir / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(database=str(database), splits=str(split_path), effective_split_seed=split_seed,
                  micro_batch_size=micro_batch_size,
                  optimizer="AdamW", weight_decay=0.0, loss="valid-target mean within match, then match mean",
                  deterministic_algorithms=True, torch_version=str(torch.__version__))
    metadata = {
        "format_version": 1, "architecture": ARCHITECTURE, "target": TARGET,
        "static_preprocessing": checkpoint["preprocessing"], "preprocessing": preprocessing,
        "stage1_source": {"path": str(args.static_checkpoint.resolve()), "kind": checkpoint["kind"],
                          "sha256": hashlib.sha256(args.static_checkpoint.read_bytes()).hexdigest()},
        "data_source": {"database": str(database), "schema_version": 1,
                        "tables": ["match_players", "player_snapshots"],
                        "match_provenance": {str(match["match_id"]): match["provenance"] for match in matches}},
        "split_assignments": assignments,
        "active_match_ids": {split: [match["match_id"] for match in group] for split, group in groups.items()},
        "training_config": config,
    }
    static.write_json(run_dir / "metadata.json", metadata)
    static.write_json(run_dir / "data_audit.json", audit)
    static.write_splits(run_dir / "split_matches.csv", assignments)
    model = initialize_model(checkpoint, args.dropout).to(device)
    transferred = list(model.hero_embedding.parameters()) + list(model.player_mlp.parameters())
    transferred_ids = {id(parameter) for parameter in transferred}
    new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in transferred_ids]
    optimizer = torch.optim.AdamW([
        {"params": transferred, "lr": args.transferred_learning_rate},
        {"params": new_parameters, "lr": args.learning_rate},
    ], weight_decay=0.0)
    log(f"Run: {run_dir}; matches: " + str({key: len(value) for key, value in groups.items()}))
    log(f"Longest training sequence: {preprocessing['max_training_frames']} frames; "
        f"batch {args.batch_size}, device micro-batch {micro_batch_size}, frame stride {args.frame_stride}")
    best_loss, stale_epochs, history = float("inf"), 0, []
    fields = ["epoch", "train_mse", "validation_mse", "persistence_mse", "validation_rmse_souls",
              "persistence_rmse_souls"]
    best_path = run_dir / "best_model.pt"
    with (run_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = count = 0
            for batch_index, batch in enumerate(loaders["train"], start=1):
                optimizer.zero_grad(set_to_none=True)
                batch_count = int(batch["target_mask"].any(dim=1).sum())
                for part in micro_batches(batch, micro_batch_size):
                    prediction = forward_batch(model, part, device)
                    mask = part["target_mask"].to(device) & ~part["padding_mask"].to(device)
                    active = int(mask.any(dim=1).sum())
                    loss = match_mse(prediction, part["targets"].to(device), mask)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    # Accumulate the same per-match mean before clipping and one
                    # optimizer update, including a smaller final effective batch.
                    (loss * active / batch_count).backward()
                    loss_sum += loss.item() * active
                    count += active
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                if batch_index % 20 == 0:
                    log(f"Epoch {epoch}: {batch_index}/{len(loaders['train'])} batches; "
                        f"running train MSE {loss_sum / count:.6f}")
            validation = evaluate(model, loaders["validation"], preprocessing, device)
            if not math.isfinite(validation["mse"]):
                raise ValueError("Nonfinite validation loss")
            row = {"epoch": epoch, "train_mse": loss_sum / count, "validation_mse": validation["mse"],
                   "persistence_mse": validation["persistence_mse"],
                   "validation_rmse_souls": validation["rmse_souls"],
                   "persistence_rmse_souls": validation["persistence_rmse_souls"]}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            log(f"Epoch {epoch}: train MSE {row['train_mse']:.6f}, validation MSE {validation['mse']:.6f}, "
                f"persistence MSE {validation['persistence_mse']:.6f}")
            if validation["mse"] < best_loss:
                best_loss, best_epoch, stale_epochs = validation["mse"], epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_validation = validation
                torch.save({**metadata, "kind": "stage2_complete", "best_epoch": best_epoch,
                            "best_validation_mse": best_loss, "model_state_dict": best_state}, best_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break
    model.cpu().load_state_dict(best_state)
    model.eval()
    backbone = {key: value for key, value in best_state.items() if not key.startswith("dynamics_head.")}
    torch.save({**metadata, "kind": "stage2_transfer", "best_epoch": best_epoch,
                "best_validation_mse": best_loss, "backbone_state_dict": backbone}, run_dir / "dynamics_encoder.pt")
    # A fixed, small prefix is sufficient to check serialization without storing
    # another full batch of multi-thousand-frame sequences in the run artifacts.
    reference = collate_sequences([loaders["validation"].dataset[0]])
    for name in ("players", "globals", "times", "targets", "target_mask", "padding_mask"):
        reference[name] = reference[name][:, :32].clone()
    with torch.inference_mode():
        prediction = forward_batch(model, reference)
    torch.save({"batch": reference, "prediction": prediction}, run_dir / "reload_reference.pt")
    verification = verify_reload(best_path)
    static.write_json(run_dir / "metrics.json", {"best_epoch": best_epoch, "epochs_completed": len(history),
                      "validation": {"soul_lead_change": best_validation}, "reload_verification": verification})
    plot_history(history, run_dir / "loss_curve.png")
    log(f"Saved best epoch {best_epoch}; independent reload reproduces dynamics predictions exactly.")
    return run_dir


def resolve_database(args):
    if args.database is not None:
        return resolve_saved_path(args.database, args.config, base=Path.cwd())
    return data_paths(args.config)["full_match"] / "full_match.duckdb"


def resolve_split_path(args):
    # Always share Stage 1 assignments, including when selecting another database.
    return (resolve_saved_path(args.splits, args.config, base=Path.cwd()) if args.splits
            else data_paths(args.config)["supplemental"] / "splits.csv")


@torch.inference_mode()
def predict(args):
    args.checkpoint = resolve_saved_path(args.checkpoint, args.config, base=Path.cwd())
    if not math.isfinite(args.through_time) or args.through_time < 0:
        raise ValueError("through-time must be a nonnegative finite gameplay time")
    model, checkpoint = load_stage2(args.checkpoint, args.device)
    database = resolve_database(args)
    matches, _ = load_matches(database, [args.match_id])
    with open_database(database) as connection:
        sequence = load_sequence(connection, matches[0], checkpoint["preprocessing"]["frame_stride"], args.through_time)
    encoded = encode_sequence(sequence, matches[0], checkpoint["static_preprocessing"], checkpoint["preprocessing"])
    batch = collate_sequences([encoded])
    standardized = forward_batch(model, batch, args.device)[0, -1].item()
    scaling = checkpoint["preprocessing"]["target"]
    return {"match_id": args.match_id, "through_game_time_s": float(sequence["times"][-1]),
            "predicted_soul_lead_change": standardized * scaling["scale"][0] + scaling["mean"][0],
            "convention": "positive means Team1 gains net worth relative to Team0 over the next selected frame interval"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train Stage 2 only when explicitly invoked")
    prediction = commands.add_parser("predict", help="Predict next-frame change from a recorded prefix")
    verification = commands.add_parser("verify", help="Verify exact CPU checkpoint reload")
    for command in (training, prediction):
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--database", type=Path, help="Full-match DuckDB override (opened read-only)")
        command.add_argument("--device", default="cpu")
    training.add_argument("--static-checkpoint", type=Path, required=True, help="Stage 1 static_encoder.pt or best_model.pt")
    training.add_argument("--output-dir", type=Path, help="Default: configured supplemental/pretrain_dynamics")
    training.add_argument("--splits", type=Path, help="Shared match_id,split CSV; must agree with Stage 1")
    training.add_argument("--extend-splits", action="store_true", help="Assign previously unseen IDs without reassigning Stage 1 matches")
    training.add_argument("--match-ids", type=int, nargs="+", help="Explicit subset for a bounded training experiment")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--split-seed", type=int)
    training.add_argument("--validation-fraction", type=float, default=.1)
    training.add_argument("--test-fraction", type=float, default=.1)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--micro-batch-size", type=int,
                          help="Sequences per device step; accumulate gradients to preserve batch-size (default batch-size)")
    training.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for new components")
    training.add_argument("--transferred-learning-rate", type=float, default=1e-4)
    training.add_argument("--dropout", type=float, default=.1)
    training.add_argument("--gradient-clip", type=float, default=1.0)
    training.add_argument("--frame-stride", type=int, default=1,
                          help="Keep every Nth recorded frame (default all); changes the target interval, saved in checkpoint")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--match-id", type=int, required=True)
    prediction.add_argument("--through-time", type=float, required=True, help="Use only observations at or before this gameplay second")
    verification.add_argument("--checkpoint", type=Path, required=True)
    verification.add_argument("--reference", type=Path)
    return parser


def main():
    import json
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        print(json.dumps(predict(args), allow_nan=False))
    else:
        print(json.dumps(verify_reload(args.checkpoint, args.reference)))


if __name__ == "__main__":
    main()
