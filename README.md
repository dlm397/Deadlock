# Deadlock

The goal of this project is practicing using statistical tools. But, to that end, I first needed data, so I used the Codex tool by OpenAI, to compose a simple data pipe, that downloaded, parsed, and stored game. Now, I have enough games to start doing analysis.

## Current Research

Consider the contents of `research/` for what I am currently doing, and what I have done. As of last updating, I am playing with a double pre-train plus fine-tuning win prediction architecture, see [model outline](research/model/buildTex/outline.pdf), I do not expect this to be fruitful, it is implementation practice concurrent with reading Simon J. D. Prince’s [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/).

Stage 1 static embedding pre-training is implemented in `pretrain_static.py` by Codex using my outline. As I make adjustments, this will be edited.

## Current Objectives
- Resolve implementation of pretraining 1
- Evaluate run 1 of pretraining 1, and tune hyperparameters

## Project layout

```text
Deadlock/
  pipeline.py                    # shared dataset launcher
  config/pipelines.toml          # config for paths, and both datasets
  app/                           # all pipeline implementation, SQL, replay parser
  .validation/                   # pipeline tests, fixtures, and other checks
  research/                      # analysis and write-ups
  pretrain_static.py             # pretrain 1 codex script
  data/
    full_match/                  # fresh Source 2 replay-backed dataset
      full_match.duckdb          # all structured data, events, status and provenance
      replays/<match_id>.dem.zst  # unchanged raw files (.dem.bz2 / .dem also supported)
    end_match/                   # public Parquet data-lake dataset
      end_match.duckdb
      tables/match_player.parquet
      tables/matches.parquet
      shards/                    # retained immutable source Parquets
      manifests/                 # pinned manifest snapshots
      state.json
      summary.json
      runs/                      # per-invocation summaries
      samples/                   # original exploratory samples, preserved
    supplemental/
      version/                   # versioned item/ability and event ID tables
      splits.csv                 # shared training/validation/test assignments
      pretrain_static/           # Stage 1 model runs, checkpoints, and plots
      migrations/                # verified layout migration records (artifact from a data format refactoring)
```

`app/pipeline.py` dispatches to `app/full_match.py` or `app/end_match.py`. `app/config.py` resolves and validates the shared configuration.

## Running

Notes for running the code.

```powershell
python -m pip install -r app/requirements.txt
npm ci --prefix app

# Local Source 2 replay ingestion (default: one attempt, 20-second snapshots)
python pipeline.py full-match ingest --replay data/full_match/replays/109507240.dem.zst
python pipeline.py full-match ingest --limit 10
python pipeline.py full-match resume --limit 10
python pipeline.py full-match retry --limit 5
python pipeline.py full-match reprocess --match-ids 109507240 --interval 40

# Public Parquet lake ingestion
python pipeline.py end-match collect
python pipeline.py end-match collect --matches 300 --max-download-mib 32 --max-files 1
python pipeline.py end-match collect --match-ids 123456 234567
python pipeline.py end-match reprocess
python pipeline.py end-match retry
```

The dataset name is required. Full-match defaults to local `ingest`; end-match defaults to `collect`. Both accept `--config`, `--match-ids` or `--match-file`, `--retry-failures`, and `--rebuild`. Only end-match supports `--csv`. Failed work exits nonzero.


## Stage 1 training from DuckDB

`pretrain_static.py` opens the configured `end_match/end_match.duckdb` read-only
and reads the selected `match_player` view.

```powershell
# Initiate training
python pretrain_static.py train
# Assign newly colect ID's, keep old ids, then training
python pretrain_static.py train --extend-splits
# Set rng seed and compute method
python pretrain_static.py train --seed 42 --device cpu

# Use the actual run directory printed by training.
$run = "data/supplemental/pretrain_static/run_<timestamp>_seed42"
python pretrain_static.py verify --checkpoint "$run/best_model.pt"
python pretrain_static.py predict --checkpoint "$run/best_model.pt" --match-id 122978436
python pretrain_static.py predict --checkpoint "$run/best_model.pt" --players players.json
```

`--database path/to/file.duckdb` selects another database; alternatively,
`--data-dir` selects a directory containing `end_match.duckdb`.

## Eligibility, coverage, and failures

Eligibility requires `duration_s > 900`, exactly 12 distinct slots in the complete roster, `game_mode = 1` (Normal), and `match_mode` in `{1, 4}` (Unranked/Ranked). `game_mode = 4` is Street Brawl and is excluded. Missing/conflicting evidence is unverified eligibility and produces no eligible outputs. A capped bulk response cannot establish the full roster count.

## Sources
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) by Simon J. D. Prince (MIT Press, 2023)
- [Deadlock API data lake and MCP access](https://deadlock-api.com/data-dumps), used to discover matches with recorded replay salts.
- [Deadlock data-lake manifest](https://data.deadlock-api.com/v1/manifest.json), used by the end-match Parquet pipeline.
- [Deadlock API metadata implementation](https://github.com/deadlock-api/deadlock-api/blob/master/api/src/routes/v1/matches/metadata.rs) and [API mode definitions](https://github.com/deadlock-api/deadlock-api/blob/master/api/src/routes/v1/matches/types.rs).
- [Valve metadata protocol](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/citadel_gcmessages_common.proto), [replay header](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/demo.proto), and [user messages](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/citadel_usermessages.proto).
- [deadem parser/engine](https://github.com/Igor-Losev/deadem), pinned to `4.0.1`; the parser includes the original project's small decoder fixes for newer Source 2 fields.
- [Extracted Valve currency enum](https://github.com/SteamTracking/GameTracking-Deadlock/blob/master/DumpSource2/schemas/client/ECurrencyType.h), [player state schema](https://github.com/SteamTracking/GameTracking-Deadlock/blob/master/DumpSource2/schemas/client/PlayerDataGlobal_t.h), and [Deadworks game-clock implementation](https://github.com/Deadworks-net/deadworks/blob/76f73b504d7b811b678d092a1e762a8e08b5d2d9/managed/DeadworksManaged.Api/GameRules.cs).
