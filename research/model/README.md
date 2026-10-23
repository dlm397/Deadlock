# Stage 1: static embedding pre-training

`pretrain_static.py` implements only Stage 1 of [outline.tex](outline.tex).
The encoder is a learned 16-dimensional hero embedding concatenated with one
scaled pre-match rank, followed by a shared `17 → 32 → 32` MLP. Six player
vectors are mean-pooled per team, concatenated in canonical team order `[0, 1]`,
and passed through the temporary `64 → 32 → 1` win head. Both MLPs have ReLU
between their linear layers only. The model returns a raw logit; sigmoid gives
**P(canonical team 1 wins)**, and the target is `winning_team_id == 1`.

## Data and preprocessing

The script uses the pipeline's `output_dir` in `config.toml`, or `--data-dir`.
It joins `end_match/<match_id>.parquet` to `matches/<match_id>.parquet`.
It reads only player `hero_id`, `initial_rank_badge`, and roster keys
(`match_id`, `player_slot`, `team_id`). The match table supplies the verified
`TeamWin` label. No `won`, kills, items, souls, other terminal statistics, or
replay features enter the model. CSV exports are ignored to avoid duplicates.
Metadata-only matches are usable; a successful replay parse is not required.
Missing match metadata and non-TeamWin/missing-winner outcomes are skipped and
recorded in `data_audit.json`. Malformed IDs or rosters fail with an error.
Every accepted match must have 12 distinct player slots and six players per team.

The source mapping in `app/normalize.py` obtains `initial_rank_badge` from
`player_rank_data.initial_display_rank`, not a post-match rank. On inspection
on 2026-10-23, the local dataset contained 2,533 matches and 30,396 players;
16,643 ranks (54.8%) were null, and 1,352 matches had no known ranks. All files
had the rank column. The pipeline maps the unknown badge sentinel `0` to null;
the model accepts either as missing. These counts are a snapshot; every run
records current availability separately for training, validation, and test.

Matches are split **before fitting preprocessing**. Default seeded proportions
are 80% training, 10% validation, and 10% test, with at least one match per split.
`<data-dir>/splits.csv` is created once and reused unchanged. Use `--splits` to
point to an existing CSV with columns `match_id,split` and split values
`train`, `validation`, `test`, including a previous run's `split_matches.csv`.
Duplicate IDs, invalid assignments, or empty active splits cause an error.
A manifest may contain extra IDs for other stages. If the dataset gains new
IDs, explicitly extend the manifest while preserving existing assignments;
the script will not silently repartition the data. Changing the seed/fractions
does not override an existing manifest. Later stages should reuse this manifest.

Rank is treated as the numeric packed badge supplied by the pipeline, not as
calibrated Elo or a newly inferred ordinal ranking. Missing values receive the
**training-only median**, then all values are min-max scaled with training-only
bounds and clipped to `[0, 1]`. If training has no known ranks or only one distinct
rank, the rank channel is zero everywhere, including inference. No missingness
feature is added, so the input width stays 17. Imputation is model preprocessing;
source Parquet values are never changed.

The hero vocabulary also uses training matches only. Index 0 is a fixed zero
embedding for unseen hero IDs. Unknown heroes still pass through the shared
MLP with their rank. Both checkpoints save the vocabulary and all preprocessing;
reuse them unchanged in subsequent stages and at inference.

## Commands

Run from the repository root with the intended Python environment activated:

```powershell
python -m pip install -r research/model/requirements.txt
python pretrain_static.py train
# Optional settings; defaults are seed 42, CPU, batch 64, LR 0.001.
python pretrain_static.py train --seed 42 --device cuda --epochs 100 --patience 10 --batch-size 64 --learning-rate 0.001
```

Training uses `BCEWithLogitsLoss` and AdamW with explicit `weight_decay=0.0`, no
dropout, shuffled training batches, and deterministic PyTorch operations.
Epochs are capped at 100 and patience at 10 (both can be reduced). Improvement
means strictly lower mean validation BCE. The best weights are restored before
export; test matches do not select the model and are not evaluated by this script.
`--split-seed` can separate the initial split seed from the training seed.
Reproducibility assumes the same data, environment, and device; CPU is the default.

Each invocation creates
`research/outputs/pretrain_static/run_<UTC timestamp>_seed<seed>/`, following
the repository's archived research run naming pattern. `--output-dir` changes
the parent directory. Outputs are:

- `best_model.pt`: complete Stage 1 `model_state_dict`, plus metadata and best epoch/BCE.
- `static_encoder.pt`: separate hero embedding and shared player MLP state dicts,
  plus the same metadata; no win-head weights.
- `metadata.json`: architecture, hero mapping, rank preprocessing, team/label
  conventions, full split assignments, active match IDs, and training configuration.
- `split_matches.csv`: reusable copy of the complete split manifest.
- `data_audit.json`: rank availability and excluded matches.
- `training_history.csv`, `loss_curve.png`: training and validation BCE per epoch.
- `metrics.json`, `reload_reference.pt`: best epoch/BCE and an exact CPU reload
  check against the restored model's validation logits and probabilities.

All `.pt` files contain tensors and plain metadata, never pickled model objects;
loading explicitly uses `weights_only=True`. An optimizer state is not saved:
these are inference/transfer checkpoints, not interrupted-training resume files.

```powershell
# Substitute the actual run directory printed by train.
$run = "research/outputs/pretrain_static/run_<timestamp>_seed42"
python pretrain_static.py verify --checkpoint "$run/best_model.pt"
python pretrain_static.py predict --checkpoint "$run/best_model.pt" --match-id 109509684
# Or predict a new roster without any outcome fields:
python pretrain_static.py predict --checkpoint "$run/best_model.pt" --players players.json
```

`players.json` is an array of exactly 12 objects, each with `player_slot`,
`team_id` (0 or 1), `hero_id`, and optional `initial_rank_badge` (integer or null).
It must include six players per team. No `match_id` or winning team is needed.

Reload and predict in Python:

```python
import json
from pretrain_static import load_stage1, win_probabilities

model, checkpoint = load_stage1(f"{run}/best_model.pt")  # already in eval mode
with open("players.json") as stream:
    players = json.load(stream)
probability = win_probabilities(model, [players], checkpoint["preprocessing"])[0].item()
```

Transfer into compatible later-stage components (using the saved mapping and
preprocessing rather than fitting new ones):

```python
import torch

checkpoint = torch.load(f"{run}/static_encoder.pt", map_location="cpu", weights_only=True)
# The later model must use the saved vocabulary size and the same 16 / 17→32→32 architecture.
later_model.hero_embedding.load_state_dict(checkpoint["hero_embedding_state_dict"])
later_model.player_mlp.load_state_dict(checkpoint["player_mlp_state_dict"])
```

Verification uses tiny temporary synthetic Parquet fixtures, never the real
research dataset or its split manifest:

```powershell
python -m unittest discover -s research/model -p "test_pretrain_static.py" -v
```

Checks cover train-only preprocessing, missing ranks and unknown heroes, split
reuse, input/label separation, roster validation, within-team permutation
invariance, raw logits, ten-epoch patience, best-weight restoration, transferable
weights, plotting, and exact evaluation-mode prediction reproduction after reload.
