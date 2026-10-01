# Deadlock research dataset

Edit [main.py](main.py), run [collect.py](collect.py), and read saved tables through [dataset.py](dataset.py). Collection parses, normalizes, validates, and saves **one match at a time**, while prefetching the next compressed replay. Independent remote auditing runs alongside local replay work. A failed match is logged and skipped. [analysis.py](analysis.py) optionally creates five descriptive plots for one match.

**Setup and collection**

Requires Python 3.11+ and Node.js 18+ with npm on PATH. Run from the project directory. The existing virtual environment can be reused; a new installation on Windows uses:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe collect.py
```

Node dependencies are installed automatically with `npm ci` when first needed. The replay bridge uses [deadem 4.0.1](https://github.com/Igor-Losev/deadem). On macOS/Linux use `python3` and `.venv/bin/python`. There are no command-line arguments; running `main.py` also starts collection.

| Setting in `main.py` | Behavior |
|---|---|
| `N_MATCHES` | Target number of additional successful matches under automatic selection; unavailable candidates can leave fewer. |
| `MATCH_IDS` | Explicit list, such as `[107532139, 107532504]`; overrides the count and automatic selection filters. |
| `SAMPLE_INTERVAL` | Replay sampling grid in game seconds; defaults to 10. |
| `OUTPUT_DIR` | Dataset directory; defaults to `./data`. |
| `REPROCESS` | Rebuild completed matches even when the code and relevant settings are unchanged. |
| `OFFLINE` | Use archived metadata and replay extracts without downloading or parsing demos. |
| `SAVE_RAW_DEMOS`, `SAVE_RAW_JSON` | Retain original sources; keep both `True` for reproducible future extraction. |
| `PARSE_REPLAYS` | Parse replay state; keep `True` for the richer dataset. |
| `QUERY_DEMO_API` | Optional independent replay kill-count check; public API limits can delay or prevent it. |
| `PREFETCH` | Default `True`: prepare at most one next match while the current match is processed. `False` disables lookahead. |
| `DOWNLOAD_WORKERS` | Default `4`: parallel byte-range streams for large replays when supported. Use `1` for one stream; maximum `8`. |

Completed matches with the same code/settings fingerprint are skipped; automatic selection excludes them. Completion checks are cached within each run to avoid repeatedly hashing the same bundles. `REPROCESS=True` overrides skipping. Rebuilding one match replaces only its own bundles. Automatic selection queries the [public data lake](https://deadlock-api.com/data-dumps) for rows containing both metadata and replay salts plus a cluster ID; its hourly snapshot can lag the newest matches. Recent-match fallback checks replay availability before selecting a match. Normal, 12-player games lasting at least 15 minutes are required: this is a convenience sample, not a random population sample. Missing replays are skipped, and an exhausted candidate search reports how many requested matches were saved.

GET requests and interrupted replay downloads retry up to three times for temporary connection errors and HTTP 429/500/502/503/504 responses, with waits of 2, 4, and 60 seconds. `Retry-After` is respected; waits longer than 60 seconds defer that request to a later run. Failed matches are logged and collection continues. Missing replay URLs and eligibility-filter skips are expected; retries cannot create unavailable source data. POST job submissions are not automatically retried.

Both collectors accept the data lake's JSON and TOON SQL result formats. Read-only SQL queries also retry temporary HTTP failures, empty RPC responses, and the server's `NotReadyError` while its snapshot catalog loads. Malformed or truncated tables are rejected rather than treated as empty results; SQL errors retain the server's explanation. These retries do not resubmit replay audit jobs.

The speed settings keep all existing fields, schema audits, optional API query checks, source archives, and compression levels. Parallel downloads stream disjoint ranges to disk, verify response lengths and server validators, and fall back to one stream if needed. Each worker has its own HTTP session. Only one local parser and one match's normalized tables are active; completed work is saved before processing the next match. Successful API queries and compatible replay extracts are reused, and decoded reference definitions are cached for up to two versions per run. Stage timings are written to each match's status and replay status; `tables/last_run.json` includes elapsed time and speed settings. The `prefetch_download` timing records download/cache preparation before processing; the later `download` timing includes local checksum verification and can be very short because the replay is already cached. Logs distinguish cached reuse from network transfers. Timings can overlap and should not be summed as wall time.

To apply cleaning changes locally, set `OFFLINE = True`, `REPROCESS = True`, then run `collect.py`. With `MATCH_IDS = None`, offline mode visits every archived match and ignores `N_MATCHES`. To change the sampling interval, use `OFFLINE = False` and select the affected match IDs so the demos can be reparsed. Set `OFFLINE` back to `False` when collecting new matches.

**Lake collection without replays**

Edit the settings at the top of [collect_lake.py](collect_lake.py), then run:

```powershell
.\.venv\Scripts\python.exe collect_lake.py
```

This separate collector writes to `data/lake` and leaves replay collection and `data/raw` alone. It downloads **every column** in the lake's `match_player` and `match_salts` tables: currently 151 and 5 columns. That includes final match/player results, item and upgrade histories, death details, objective records, statistics time series, tracked stats, rank fields, and every other exposed field. Source names and missing values are retained. The column list is discovered on each run, so new upstream columns are included automatically. Steam profile histories and leaderboard tables are not match records and are not collected.

| Setting in `collect_lake.py` | Behavior |
|---|---|
| `N_MATCHES` | Additional successful matches to save; completed matches and failures do not consume the target. |
| `MATCH_IDS` | Explicit IDs; overrides the count and automatic selection filters. |
| `GAME_MODE`, `MATCH_MODE` | Source enum names, e.g. `"Normal"`, `"StreetBrawl"`, `"Ranked"`; `None` removes the filter. |
| `MIN_DURATION_S`, `EXPECTED_PLAYERS` | Defaults: at least 900 seconds and 12 distinct player slots. Use `0` and `None` to remove these restrictions. |
| `START_DATE`, `END_DATE` | Match start-time interval, inclusive start and exclusive end. ISO dates/times; naive values are UTC. |
| `MIN_MATCH_ID`, `MAX_MATCH_ID` | Inclusive ID bounds; discovery proceeds newest ID first. |
| `REPROCESS` | Replace existing lake matches after successful retrieval; otherwise valid saved matches are skipped. |
| `PAGE_SIZE`, `MAX_CANDIDATES` | Discovery query size and limit on new attempted matches. The automatic candidate limit defaults to `max(100, N_MATCHES * 10)`. |
| `REQUEST_DELAY` | Optional minimum interval between SQL queries, in seconds. HTTP retries retain the existing backoff policy. |
| `OUTPUT_DIR`, `COMPRESSION_LEVEL` | Default `./data/lake`, with lossless Zstandard level 19. |

Each match has `<match_id>/match_player.parquet`, `match_salts.parquet`, and `status.json.gz`. The status records all source schemas, column encodings, checksums, row counts, and provenance; request logs are compressed alongside them. `tables/` holds the observed manifest and schemas; `last_run.json` summarizes the run. Both tables are staged before replacing an existing saved match. Failures are recorded and the next match is attempted.

The SQL service caps response size. Full records are transferred in bounded chunks with checksums, byte-length checks, column checks, and source-inventory checks. The collector holds only one match at a time and retains **all source versions and exact duplicate rows**. Those rows must not be treated as independent players in statistical models. The live lake updates periodically; the observed manifest is archived for context, but the MCP queries are not pinned to that manifest. A table that changes during retrieval is rejected for a later retry.

```python
import json
from dataset import lake_match_ids, read_lake_match, iter_lake_matches

mid = lake_match_ids()[0]
players = read_lake_match(mid)  # Every stored version; nested values are JSON strings.
distinct_records = players.drop_duplicates()
economy = read_lake_match(mid, columns=["match_id", "player_slot", "net_worth", "created_at"])
history = json.loads(players.iloc[0]["stats.net_worth"])
times = json.loads(players.iloc[0]["stats.time_stamp_s"])

for mid, players in iter_lake_matches():
    # Construct study features one match at a time.
    pass
```

Lake reader `output_dir` is the parent dataset directory (`./data`), consistent with the existing readers. For a custom location, keep the `lake` subdirectory, e.g. collect to `./study/lake` and read with `output_dir="./study"`. Enums retain their source names (`"Team0"`, `"Normal"`); timestamps retain their source text; integer IDs retain exact nullable integer types. Inspect `created_at` when deciding which source revision to analyze after removing exact duplicates. Statistics arrays are the API's sparse observations, not the replay collector's regular sampling grid; their last entry is not assumed to occur at match end. Fields absent from the upstream lake cannot be recovered by this collector.

**Storage**

```text
data/
  raw/
    107532139/
      summaries.zip
      states.zip
      purchases.zip
      events.zip
      sources/             # Downloads, replay extracts, provenance, audits, status
    107532504/
      ...
  tables/
    heroes.parquet
    items.parquet
    abilities.parquet
    catalog.json           # Match status, versions, and row counts
    last_run.json
    run.log
    sources/               # Versioned reference definitions and selection records
  processed/               # Your study features, aggregates, and model outputs
    plots/                 # Created only when analysis.py is run
  lake/                    # Separate collector: collect_lake.py
    108313916/
      match_player.parquet
      match_salts.parquet
      status.json.gz
      requests.json.gz
    tables/                # Source manifest and complete column schemas
    last_run.json
    run.log
```

Each ZIP contains named Parquet tables compressed losslessly with Zstandard level 19. ZIP stores those already-compressed files without a second compression pass. Source JSON is gzip-compressed with its original bytes preserved; replay extracts use gzip NDJSON and demos retain their original downloaded compression. The loader handles bundles directly; no manual extraction or CSV duplicates are needed.

| Bundle | Tables |
|---|---|
| `summaries.zip` | `matches`, `players`, `game_versions`, `metadata_teams`, `metadata_damage`, `metadata_custom_stats`, `metadata_gold_sources` |
| `states.zip` | `player_state`, `team_state`, `metadata_player_stats`, `metadata_player_paths` |
| `purchases.zip` | `purchases`, `metadata_item_records` |
| `events.zip` | `events`, `metadata_objectives`, `metadata_midboss`, `legacy_events`, `replay_event_counts` |

A manifest is also included in `summaries.zip`. Detailed per-match warnings and field coverage are in `sources/status.json.gz`; failures appear in `sources/last_error.json.gz`.

**Reading data and building model inputs**

```python
from dataset import match_ids, read_match, read_reference

mid = match_ids()[0]
states = read_match(mid, "player_state", columns=[
    "player_id", "game_time", "net_worth", "unspent_souls",
    "spirit_power", "spirit_power_source_game_time", "spirit_power_state_age",
])
heroes = read_reference("heroes", version=6698)
```

Reference tables keep multiple `definitions_version` values; join item/hero IDs using the match's version. IDs remain exact and readers use nullable pandas dtypes; do not convert IDs to floating point or replace missing IDs with zero. Nested dictionaries and arrays are JSON strings: decode a nonmissing value with `json.loads(value)` when needed.

For modeling, iterate through matches and retain only small feature records. This example produces one observation per match using team 0's advantage at ten minutes:

```python
from pathlib import Path
import pandas as pd
from main import OUTPUT_DIR
from dataset import iter_matches, read_match

features = []
for mid, teams in iter_matches("team_state", columns=[
    "game_time", "team", "net_worth_advantage", "economy_complete",
]):
    at_ten = teams.loc[(teams["game_time"] == 600) & (teams["team"] == 0)]
    if len(at_ten) != 1 or not at_ten.iloc[0]["economy_complete"]:
        continue
    advantage = at_ten.iloc[0]["net_worth_advantage"]
    if pd.isna(advantage):
        continue
    match = read_match(mid, "matches", columns=["winning_team", "game_version"]).iloc[0]
    features.append({"match_id": mid, "advantage_600": advantage,
                     "team0_win": int(match["winning_team"] == 0),
                     "game_version": match["game_version"]})

output = Path(OUTPUT_DIR) / "processed"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(features).to_parquet(output / "win_features.parquet", index=False, compression="zstd")
```

Memory grows with one match plus your retained features, rather than all replay rows. Individual unusually large matches can still exceed available resources and fail. Split model evaluation by match, avoid future information in predictors, and account for repeated observations. The initial five matches support workflow development, not reliable statistical inference.

**Markov analysis of relative net worth**

Edit the settings at the top of [markov_souls.py](markov_souls.py), then run:

```powershell
.\.venv\Scripts\python.exe markov_souls.py
```

The script reads existing matches through `dataset.py`, constructs relative net worth and within-match ranks, and estimates overall and game-stage transition matrices. It uses a common sampling interval without interpolation (20 seconds for the current mix of 10- and 20-second data). State boundaries and prediction parameters use training matches only; complete matches are held out for validation. Collection and raw data are unchanged.

Results, compressed modeling tables, and figures are saved under [data/processed/markov_souls](data/processed/markov_souls/). Start with [summary.txt](data/processed/markov_souls/summary.txt). After running the script, forecast from a state using seconds:

```python
from markov_souls import predict_state_distribution

forecast = predict_state_distribution("Behind", 12 * 60, 5 * 60)
print(forecast)
```

Forecast horizons must be multiples of the fitted sampling interval. The returned relative-net-worth value is a probability-weighted state-mean proxy. Evaluation is exploratory: repeated players and overlapping transitions are dependent, and forecasts assume the match continues.

For the same analysis at team level, edit [markov_team_souls.py](markov_team_souls.py) and run:

```powershell
.\.venv\Scripts\python.exe markov_team_souls.py
```

This reads `team_state` through `dataset.py` and requires both teams' complete, fresh replay observations. `relative_net_worth` is the team total minus the mean of the two team totals; `net_worth_advantage` is the full lead over the opponent, twice that relative value. Both teams remain in the same training or test split: their opposite relative trajectories are dependent observations of the same match.

Outputs mirror the player analysis under [data/processed/markov_team_souls](data/processed/markov_team_souls/): compressed modeling tables, overall and time-bin matrices, validation scores, forecasts, figures, and [summary.txt](data/processed/markov_team_souls/summary.txt). Team forecasts also return `expected_net_worth_advantage`, a state-mean estimate of the future lead:

```python
from markov_team_souls import predict_state_distribution

forecast = predict_state_distribution("Behind", 12 * 60, 5 * 60)
```

Both scripts share their estimation, validation, and plotting code under `agent/analysis/`; settings and data preparation remain in the scripts.

**Measurement coverage**

`spirit_power` was missing because the source calls it `tech_power`. Both names are now retained. Spirit and weapon power come from sparse metadata snapshots, typically every 180 seconds; their dedicated source-time and age columns distinguish them from the finer replay grid. Values before the first observation remain missing. Existing extracts also now expose unspent/spent souls, available ability points, unlock tokens, denies, last hits, kill streak, and disconnect state.

Exact transaction balances, verified 3D velocity, interpreted ability levels, and some objective timelines remain unavailable. The ambiguous `souls` placeholder remains null; use the explicit `net_worth`, `unspent_souls`, and `spent_souls` measures. Missing values are not zeros. Item removal does not establish a sale, and overlapping metadata/replay events require source-aware counting. See [field coverage and evidence](data/tables/field_coverage.json) for the measured sample coverage and limitations.

Collection internals, parser dependencies, tests, and research notes live under `agent/`; normal collection and analysis do not require opening them. Change source interpretation there only when developing the collector. Keep study-specific restrictions and derived features in your own scripts and `data/processed/`.
