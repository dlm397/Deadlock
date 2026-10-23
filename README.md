# Deadlock

The goal of this project is practicing using statistical tools. But, to that end, I first needed data, so I used the Codex tool by OpenAI, to compose a simple data pipe, that downloaded, parsed, and stored game. Now, I have enough games to start doing analysis.

## Current Research

Consider the contents of `research/` for what I am currently doing, and what I have done. As of last updating, I am playing with a double pre-train plus fine-tuning win prediction architecture, see [model outline](research/model/buildTex/outline.pdf), I do not expect this to be fruitful, it is implementation practice concurrent with reading Simon J. D. Prince’s [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/).

Stage 1 static embedding pre-training is implemented in `pretrain_static.py` by Codex using my outline. As I make adjustments, this will be edited.

## Project layout

```text
Deadlock/
  .venv/
  .validation/         # local verification artifacts and code backups
  pipeline.py          # command to run the pipeline
  pretrain_static.py   # code to pretrain the static embedding
  config.toml          # settings to edit
  README.md
  app/                 # Python code, replay parser, dependencies
  other/               # legacy importer and CSV cleanup utility
  research/            # analysis tools and results
  data/                # retained replays and parsed outputs
```

The implementation is in `app/pipeline.py`; the top level `pipeline.py` is a launcher.

## Running

Edit `config.toml` for settings. For desired match ID's, set `match_ids = [123456, 234567]`, or set `match_ids_file` to a text file of IDs. Empty IDs and file enable discovery through the Deadlock API's public SQL/MCP endpoint. Discovery selects newest match IDs with recorded replay salts, metadata salts, and replay server identifiers. Every candidate, including an explicitly supplied ID, must pass the same eligibility check using its full individual match metadata.

```powershell
# Discovers and downloads match, using settings in comfig.toml
python pipeline.py collect
# Downloads the given match ID's
python pipeline.py collect --match-ids 123456 234567
# Downlods all the match ID's in matches.txt, --csv enables option for optional csv.
python pipeline.py collect --match-file matches.txt --csv 
# Rebuild outputs from retained reply files, using config.toml interval
python pipeline.py reprocess
# Rebuilds, overrides interval timing from config.toml with --interval flag
python pipeline.py reprocess --interval 60
# Retries matches recorded in the failure log, can be set for match ID's, same as collect
python pipeline.py retry
```

All commands accept `--config path/to/config.toml`, `--match-ids`, `--match-file`, `--interval`, `--retry-failures`, `--rebuild`, and `--csv`. With no command, collection is the default.

`reprocess` uses selected retained replay files, or all retained replays when no IDs are selected. It can fetch small metadata and versioned definitions again, but does not download the replay again. `retry` explicitly retries active failures; absent an explicit/configured selection, it uses the failure log. Ordinary collection and reprocessing skip failures unless `--retry-failures` is set. Collection skips complete outputs at the requested interval. Use `--rebuild` when changing extracted variables without changing the interval.

## Configuration

Paths are relative to the configuration file. Main settings:

| Setting | Default / purpose |
| --- | --- |
| `match_ids`, `match_ids_file` | Explicit IDs or an input file; empty enables discovery |
| `sampling_interval_s` | Defaults to `20`; time slicing of event stream in seconds |
| `output_dir` | Defaults to `data` |
| `retain_replays` | Defaults to `true`; false removes a replay only after successful output writes |
| `export_csv` | Defaults to `false`; optional CSV alongside each Parquet, for visual inspection |
| `retry_failures` | Defaults to `false`; allow attempts listed in the failure log |
| `rebuild_outputs` | Defaults to `false`; rebuild existing outputs |
| `discovery_count`,  | Number of matches to retreive with full data|
| `discovery_limit` | Number of allow match's to request for full data for exiting |
| `api_base` | API location; If future data sources want to be implemented |
| `disable_steam` | `false` permits Steam fallback, `true` disables it |

Changing `sampling_interval_s` rebuilds time slices from the retained replay on the next run. Files for that match are replaced, so use a different `output_dir` to keep multiple sampling resolutions.

Discovery pages through `match_salts` via `/v1/mcp`, excluding active failures unless retries are enabled, known ineligible matches unless retries/rebuilding are enabled, and complete outputs already at the requested interval unless rebuilding. These exclusions do not consume `discovery_limit`. Eligibility rejections and download/parse failures among the selected candidates do consume it. `discovery_count = 1` and `discovery_limit = 30` therefore stop after one successful collection or exhaustion of 30 candidates. Explicit IDs, `retry`, and `reprocess` do not use this discovery limit.

Recorded salts identify a replay URL but do not guarantee that Valve still serves the file. A failed discovery query stops collection with an error; it does not fall back to metadata-only recent matches. `disable_steam = false` does not force the API to refresh an existing record with a missing replay salt. Prior failed IDs still require `retry` or `--retry-failures` to be attempted again.

## Outputs and schemas

```text
data/
  matches/<match_id>.parquet          # End of match data common across players
  end_match/<match_id>.parquet        # End of match player data
  time_slices/<match_id>.parquet      # Player data by time slice
  events/<match_id>.parquet           # Processed event stream
  replays/<match_id>.dem.bz2          # or .dem.zst / .dem, according to content, raw retrieved files
  version/<client_version>_item.csv   # Lookup table for item ID's by version
  version/<client_version>_event.csv  # Look up table for event ID's by version
  collection_status.csv               # List of status of collected matches
  failures.csv                        # List of match ID's which failed
```

Tables contain scalar, typed columns. Identifiers remain exact integers; calendar datetimes use UTC; game times use seconds. Missing, unavailable, and unverified values are null, never a fabricated zero. A recorded zero remains zero, except documented unknown sentinels such as an unassigned rank badge. Join `matches` to the other tables on `match_id`, and player records on `(match_id, player_slot)`; `account_id` is optional. Canonical teams are metadata teams `0` and `1`. Output modes/outcomes use readable enum names; eligibility checks use their verified protocol meanings. `export_csv = true` writes a CSV alongside each Parquet file.

The column definitions and source mappings are kept near the top of `app/normalize.py` and `app/replay.mjs`; these are the small sections to edit when changing variables.

| Table | Row key | Contents |
| --- | --- | --- |
| `matches` | `match_id` | One row per match: start, duration, player count, modes, match outcome, winning team, map, and distinct software/build identifiers |
| `end_match` | `match_id, player_slot` | Identity/team; player win outcome; rank/lane; K/D/A, net worth, level, last hits, denies; final combat statistics where proven final |
| `time_slices` | `match_id, player_slot, game_time_s` | Replay-observed counters, net worth and unspent souls, health/alive state and XYZ position; actual observation time and final-observation flag |
| `events` | `match_id, event_id` plus participant | Deaths/kills/assists, item changes, ability progression, objectives, midboss/rejuvenator information where supported |

Explicit metadata final counters take priority. Combat values come from one metadata snapshot whose integer `time_stamp_s` equals `duration_s`. `combat_stats_time_s` is that timestamp and `combat_stats_at_end` reports whether that snapshot exists. If it does not exist, the combat values/time are null and the flag is false. Individual absent fields remain null even when the flag is true. There is no fallback to an earlier snapshot, so these fields share one observation time. Sparse metadata snapshots are not inserted into the replay timeline.

`net_worth` and `unspent_souls` are different variables. Replay `hero_damage_taken` remains null where no verified cumulative replay field is available. Position units are Source 2 world units. A client version, replay build number, replay patch version, and game-mode version remain separate; no unverified readable balance-patch label is assigned. `hero_build_id` is a player's selected build guide, not a software version, and is not exported.

## Version definitions

`data/version/` (or `version/` inside your configured `output_dir`) contains `<client_version>_item.csv` and `<client_version>_event.csv`. These are always CSV, independently of `export_csv`. Collection writes definitions for each known replay client version, and `migrate` also fills them for versions already present in `matches`. A metadata-only match can have an unknown client version; no version is guessed for it.

The item file has `client_version, item_id, name, class_name, type, item_slot_type, item_tier, cost`. It contains the exact-version API asset catalog, including abilities and weapon definitions as well as shop items. Join an event's `item_id`, `ability_id`, or `imbued_ability_id` to this file's `item_id`, using `matches.client_version` to choose the file. Names may be internal labels where the source has no friendly name. Missing shop fields stay blank, and catalog `cost` is a definition price rather than an observed payment. Unknown asset IDs should remain unmatched.

The event file has `client_version, event_type, event_name, description, participant_roles, notes`. It documents all event types supported by this pipeline and their limitations. Join using `event_type`; `event_id` identifies a particular occurrence and is shared by its participant rows. These definitions describe pipeline labels, so they may be identical across versions and do not assert that every type occurred or was available in that version. Objective IDs and raw upgrade codes are not item catalog IDs.

If exact-version assets are unavailable, an event codebook is still written for a known client version. No empty item catalog replaces an existing one, and current-version assets are never substituted for missing historical definitions.

## Game clock and sampling

Time zero is the recorded gameplay start. The clock uses the replay's match-clock update anchor and actual tick interval when available. The fallback is `(effective_network_tick - total_paused_ticks) * tick_interval - game_start_time`, with the pause-start tick used during an ongoing pause. Pregame and postgame are excluded. Required clock anchors must be available; otherwise parsing fails rather than inventing a clock.

Sampling targets are `0, Δ, 2Δ, ...`. Complete tick updates are processed before taking state. Each target uses the latest valid state at or before it; there is no future backfill or interpolation. All 12 roster slots remain represented even when their state is missing. Prior states older than 2.1 actual tick intervals are left missing; this allows normal tick-boundary alignment without carrying state across gaps. Grid rows stop at the last observation, rather than extending a partial replay to the final duration. `observed_time_s` records the source state time, not the target time.

The final observed gameplay state is appended at its actual time when off-grid, or marks the matching grid row. Metadata duration has whole-second precision, so an observed fractional gameplay ending up to one second beyond it is retained; the first verified game-end marker stops extraction before postgame cleanup. `is_final_observation` means the last observed gameplay state, not proof that the replay reaches match end. Event intervals are `(previous_timestamp, timestamp]`; counter differences describe interval changes, not exact event times. The first interval has no assumed zero baseline.

## Events and source limitations

Event participant rows share an `event_id`; count distinct event IDs to count events. Relevant IDs and participant roles are scalar columns; fields that do not apply stay null. Overlapping metadata/replay reports are reconciled before export. An item's removal does not by itself prove a sale, and definition prices are not actual payments/refunds. An objective's owning team differs from its attacking team. Midboss death and rejuvenator claim times must not be equated.

`objective_team_id` identifies the objective owner; it is not the attacking team. `claiming_team_id` and `related_event_id` connect a known rejuvenator claim to its midboss event, with a null claim time when only the metadata identifies the team. Death `duration_s` is the recorded death duration and `time_to_kill_s` is separate. `event_source` identifies metadata, replay, or their reconciliation. `upgrade_id` and `upgrade_info` retain protocol codes, not inferred ability levels; `imbued_ability_id` identifies an imbue target. A metadata ability record without a verified action remains `ability_progression`, and unavailable definitions can leave an `item_record` classification.

Verified replay `EPurchased`/`EUpgraded` notifications can label ability unlocks/upgrades. In the two demonstrations, these notifications describe item changes; ability progression is therefore retained with its timestamp, ability ID, and native upgrade codes. The packing of metadata upgrade codes has not been verified, so zero is not guessed to mean an unlock. Rejuvenator notifications likewise lack a verified action-code mapping; known claiming teams are retained with missing claim times.

Versioned item/ability definitions are requested only for the replay's identified client version; current definitions are not substituted for unknown historical ones. Missing definitions or protocol fields reduce available event detail. Full damage/currency streams, general casts, inventory snapshots, and cooldown histories are deliberately outside this initial scope. An empty event result does not prove the game had no events if the source lacks coverage.

## Eligibility, coverage, and failures

Eligibility requires `duration_s > 900`, exactly 12 distinct slots in the complete roster, `game_mode = 1` (Normal), and `match_mode` in `{1, 4}` (Unranked/Ranked). `game_mode = 4` is Street Brawl and is excluded. Missing/conflicting evidence is unverified eligibility and produces no eligible outputs. A capped bulk response cannot establish the full roster count.

`collection_status.csv` distinguishes `complete`, `end_match_only`, `ineligible`, `unverified_eligibility`, and `failed`. It records the sampling interval, final observed gameplay time, replay-end coverage (true/false/unknown), and a short reason. `complete` means the requested research tables were written, not necessarily full replay coverage or availability of every optional definition. Coverage is true when an end marker agrees with metadata duration within two seconds, false when observations end clearly earlier, and blank when uncertain. Eligibility and replay coverage are separate: partial replay observations can be retained for an eligible match. The `matches` and `end_match` outputs survive replay retrieval or parsing failure; this retains the status name `end_match_only`. A failed rebuild preserves earlier successful tables and their recorded interval, with the new failure recorded separately.

`failures.csv` contains active failures with `match_id, stage, reason, attempted_at_utc`. Ordinary runs skip these IDs before querying; `retry` permits another attempt, and successful recovery clears the active failure. Expected per-match failures are reported and collection continues. Unexpected programming errors remain visible with a traceback.

## Sources
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) by Simon J. D. Prince (MIT Press, 2023)
- [Deadlock API data lake and MCP access](https://deadlock-api.com/data-dumps), used to discover matches with recorded replay salts.
- [Deadlock API metadata implementation](https://github.com/deadlock-api/deadlock-api/blob/master/api/src/routes/v1/matches/metadata.rs) and [API mode definitions](https://github.com/deadlock-api/deadlock-api/blob/master/api/src/routes/v1/matches/types.rs).
- [Valve metadata protocol](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/citadel_gcmessages_common.proto), [replay header](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/demo.proto), and [user messages](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/citadel_usermessages.proto).
- [deadem parser/engine](https://github.com/Igor-Losev/deadem), pinned to `4.0.1`; the parser includes the original project's small decoder fixes for newer Source 2 fields.
- [Extracted Valve currency enum](https://github.com/SteamTracking/GameTracking-Deadlock/blob/master/DumpSource2/schemas/client/ECurrencyType.h), [player state schema](https://github.com/SteamTracking/GameTracking-Deadlock/blob/master/DumpSource2/schemas/client/PlayerDataGlobal_t.h), and [Deadworks game-clock implementation](https://github.com/Deadworks-net/deadworks/blob/76f73b504d7b811b678d092a1e762a8e08b5d2d9/managed/DeadworksManaged.Api/GameRules.cs).
