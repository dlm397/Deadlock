# Deadlock research, simplified

An independent implementation of replay collection, complete end-of-match lake collection, and player/team Markov analysis. It uses ordinary functions, processes one match at a time, and prints a failure before continuing to the next match. The original project is unchanged.

## Project layout

```text
refactored/
  main.py                  # Collection settings and entry point
  README.md
  requirements.txt         # Python dependencies
  package.json             # Replay parser dependencies
  package-lock.json
  datapipe/
    collect.py             # Detailed collection
    lake.py                # End-of-match collection
    sources.py             # API requests and reference definitions
    replay.py, replay.mjs  # Replay decoding
    normalize.py           # Research table construction
    dataset.py             # Table readers, storage, and match counts
  analysis/
    markov.py              # Player/team model settings, forecasts, and plots
    __main__.py            # python -m analysis
```

Dependency manifests and `.gitignore` remain at the project root. Python packages also contain `__init__.py`. Installed dependencies, generated analysis outputs, and temporary verification files use `node_modules/`, `data/`, and `validation/` respectively. Run the commands below from `refactored/`; collection can also be launched from elsewhere using the full path to `main.py`.

## Setup and settings

Requires Python 3.11+ and Node.js 18+ with npm. From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci --ignore-scripts --no-audit --no-fund
.\.venv\Scripts\python.exe main.py
```

You can instead reuse the original environment with `..\.venv\Scripts\python.exe`. Node dependencies are installed locally; nothing imports the original implementation.

Edit **main.py** for collection:

| Setting | Meaning |
|---|---|
| `OUTPUT_DIR` | `"../data"` uses existing data without copying it; `"./data"` creates a separate dataset. Paths are relative to this directory. |
| `COLLECTION` | `"detailed"` downloads metadata and parses replays; `"lake"` collects all exposed match columns without demos. |
| `N_MATCHES` | Additional successful matches to collect automatically. The candidate limit can leave fewer. |
| `MATCH_IDS` | Explicit IDs, or `None` for automatic selection. Explicit IDs bypass selection filters. |
| `SAMPLE_INTERVAL` | Replay grid in game seconds; default **20**. |
| `SAVE_RAW_DEMOS`, `SAVE_RAW_JSON` | Retain downloaded demos and API JSON. Replay extracts and historical definitions remain available. |
| `PARSE_REPLAYS` | Set `False`, together with `SAVE_RAW_DEMOS=False`, for metadata-only detailed collection. |
| `REPROCESS` | Rebuild saved matches; otherwise completion markers control skipping. Changing code/settings does not trigger rebuilding automatically. |
| `OFFLINE` | Detailed collection only: rebuild archived extracts without network access. Use with `REPROCESS=True`; changing the grid requires reparsing with `OFFLINE=False`. |
| Filters | Duration/player count and Normal mode for detailed selection. Lake also supports mode, UTC date, and match-ID bounds. |

`main.py` chooses the collector. `python -m datapipe.collect` can also run detailed collection directly. Normal automatic collection requires 12 players and at least 15 minutes. With offline collection and no explicit IDs, every archived match is considered regardless of `N_MATCHES`. Excluded matches print `SKIPPED` with the filter reasons. Network/parser failures print a short `FAILED` message, save details to the match's `sources/error.log` (lake: `error.log`), and continue. Full parser output stays in `sources/parser.log`. A temporary HTTP 503 is a server failure; rerun those IDs when the service recovers. Startup/selection failures stop cleanly with details in `tables/collection_error.log`.

Cached extracts must be complete and match the requested sampling interval. Incomplete extracts are reparsed online; completed matches still require `REPROCESS=True` to rebuild. Parser failures never replace previous successful extracts or count as completed matches.

## Read and analyze

Export a downloaded, processed match's player states for inspection in Excel by editing the settings at the top of `export_match.py`, then running that file directly in VS Code:

```python
MATCH_ID = 109521468
TABLE = "player_state"  # Set "all" to export every saved table.
DATA_DIR = OUTPUT_DIR  # Uses main.py's data directory.
```

Optional command-line overrides are also supported:

```powershell
.\.venv\Scripts\python.exe export_match.py 109521468
.\.venv\Scripts\python.exe export_match.py 109521468 --table all
```

The first command writes `data/csv/109521468/player_state.csv`; `--table all` writes one CSV for each saved table in that folder. Use `--table events` or another table name to select a single table, and `--data-dir` to override `OUTPUT_DIR` from `main.py`. Existing exports are overwritten on rerun. Blank cells represent missing values; nested fields remain JSON text. This exports existing tables without downloading or parsing replays.

Run Python from this directory so imports select the new reader:

```python
from datapipe.dataset import match_ids, read_match, read_reference, iter_matches

mid = match_ids()[0]
states = read_match(mid, "player_state")
version = read_match(mid, "matches").iloc[0].game_version
heroes = read_reference("heroes", version=version)
for mid, teams in iter_matches("team_state"):
    pass  # Build a small feature record, then release this match's frame.

lake_ids = match_ids(lake=True)
players = read_match(lake_ids[0], "match_player")
```

All reader functions accept `output_dir`. `python -m datapipe.dataset` prints detailed and end-of-match counts. Lake tables retain duplicate rows and multiple source versions; inspect `created_at` and remove exact duplicates deliberately for modeling. Nested arrays/maps are JSON strings, decoded with `json.loads`.

Edit **analysis/markov.py** and run `python -m analysis` for the player or team Markov model (`LEVEL`). It follows the data directory from main.py and writes to `refactored/data/processed/markov_player_souls/` or `markov_team_souls/`; change `RESULTS_DIR` to choose another destination. Settings include state count, game-time bins, sampling interval, prediction horizons, and held-out fraction. `describe_match()` produces the five descriptive views in two figures. Use the module commands above to run files inside the packages, so Python resolves their imports from the project root.

```python
from analysis import run, predict_state_distribution, describe_match
model = run(level="team")
forecast = predict_state_distribution("Behind", 12 * 60, 5 * 60, model)
describe_match(mid)
```

Markov outputs include compressed modeling/transition tables, empirical matrices and counts by horizon/stage, held-out predictions and scores, figures, model parameters, and summary.txt. Whole matches are held out **before** estimating boundaries and forecast parameters. Descriptive matrices use all matches; predictive matrices use training matches only. Longer transitions require uninterrupted observations.

## Storage and measurements

```text
OUTPUT_DIR/
  raw/<match_id>/
    summaries.zip          # Matches, players, versions, teams, damage, custom/gold stats
    states.zip             # Player/team panels and native metadata stats/paths
    purchases.zip          # Transactions and original item records
    events.zip             # Events, objectives, Midboss, legacy events, event counts
    sources/               # Compressed metadata, replay extracts, demo, completion record
  tables/                  # Historical heroes/items/abilities and their source definitions
  lake/<match_id>/         # match_player.parquet, match_salts.parquet, status.json.gz
  lake/tables/             # Source schemas and observed manifest
```

The detailed bundles retain all **18 table names**, using Zstandard level 19 inside uncompressed ZIP containers. Lake Parquet uses the same compression. JSON/extracts use gzip; downloaded demos retain their original compression. IDs remain exact nullable integers. The existing dataset.py can read the new detailed bundles; this reader can read both versions.

Use `net_worth`, `unspent_souls`, and `spent_souls`; the ambiguous `souls` placeholder stays null. Spirit power is metadata's `tech_power`, typically observed much less often than the replay grid. Keep its source-time and age columns. Metadata is carried forward only from the past, and the latest sparse snapshot is not silently treated as a final scoreboard.

Team totals require complete observations. Positions are derived from replay cells or quantized metadata paths, with quality/provenance labels. A removed item is not necessarily a sale; purchase balances are earlier sampled balances. Rejuvenator claim teams can be known while claim times remain missing. Definitions are joined by the replay's exact client version; current assets never substitute for unavailable historical definitions.

Collection is a convenience sample. Repeated players, opposing teams, and overlapping transitions are dependent. Forecasts assume the match continues; expected wealth is a state-mean proxy. Three matches validate implementation agreement, not predictive usefulness or the Markov assumption.

## Differences and verification

Removed API wrapper classes, prefetch/parallel-range machinery, retry/backoff orchestration, code fingerprints, catalog/checkpoint management, exhaustive schema/coverage audits, migrations, and the broad test suite. Independent remote demo SQL kill-count auditing was removed because it duplicates local event information and can stall on service failures. No per-read checksum scan is performed; bundle CRC and Parquet decoding still apply, and saved hashes remain available.

Player and team modeling share one preparation/estimation path; team wealth is summed from the same fresh player observations. The persistence and overall/stage model comparisons remain; the training-marginal baseline and duplicate report files were removed. Detailed state rows are ordered by player slot and time. `field_provenance` is smaller, metadata carry-forward quality is named `metadata_asof`, and explicit `net_worth_source_game_time` improves purchase/team freshness when a replay field is unavailable. Remote-audit-only columns are absent. Core economic/event measurements are unchanged.

Compared original and new normalization on **107673695, 108313916, 104533570** (versions 6698, 6701, 6686): all table row counts and substantive winners, rosters, K/D/A, economic states, purchases and event counts agreed. Saved-table comparisons passed, including reading the new bundles with the original reader. A fresh new-parser run on 108313916 produced identical decompressed player-state, event, and entity extracts. Both original Markov scripts were run on these matches; observations, states, matrices, forecasts and held-out scores agreed with the new models.

Two local lake matches round-tripped all fields, exact integer/null types and duplicate multiplicities. A live fetch exposed **157** player columns, adding six fields to the archived 151-column schema. Its records matched the archived values; the service currently returns 12 rather than the archived 24 records for 108313916. The collector retains what the source actually returns. Integrity checks cover temporal gaps, match-separated validation, source alignment, historical definitions, null/exact IDs, and continuing after a match failure.

The necessary complexity remains in replay clock/player alignment, measurement normalization, and chunked lake retrieval: the public SQL endpoint limits response size, so records are chunked and checked to prevent silent truncation. Deadem/engine are pinned to 4.0.1 with a small adapter for the replay-declared `fixed8` encoding, `CUtlBinaryBlock`, and `CGlobalSymbol`. Binary blocks remain lossless byte arrays. These newer encodings caused the October 1 missing-entity errors; the adapter fixes decoding without ignoring entities or fabricating state. Decoder extensions are recorded in the replay summary and `game_versions`.

After the decoder fix, all three reported failures (109509684, 109507240, 109510742; version 6730) parsed and saved successfully. Their 924, 1,356, and 1,968 player-state rows passed roster, source-time, team-total, and fresh net-worth checks. Replay hero-kill counts matched metadata deaths (70, 83, 135). The older 108313916 replay still produced byte-identical decoded extracts. Separate failure-path checks verified concise messages, continued collection, and preservation of previous successful extracts.

At the initial simplification, source/configuration files decreased from 42 to 14 and Python/JS from 6,655 lines (4,007 excluding tests) to 1,740. Dependencies remain the same research stack; NumPy and the parser engine, previously transitive, are now explicit. Counts exclude installed dependencies and generated data.

Temporary verification outputs remain in `validation/` because automatic approval review blocked cleanup. They are ignored by Git, unused by the implementation, and can be removed manually. Existing datasets were accessed in place; the temporary outputs were generated for comparison. The initial verification confirmed all 42 original tracked file hashes and the paths, sizes and modification times of all 10,882 existing data files were unchanged.
