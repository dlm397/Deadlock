"""Launch the replay or data-lake pipeline with shared sectioned configuration."""

import argparse
import sys

import requests

from .config import DEFAULT_CONFIG, load_dataset_config
from .console_log import log


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = parser.add_subparsers(dest="dataset", required=True)
    full = datasets.add_parser("full-match", help="Ingest retained local Source 2 replays into DuckDB")
    end = datasets.add_parser("end-match", help="Ingest public data-lake Parquet shards")
    for command in (full, end):
        choices = ("ingest", "collect", "resume", "retry", "reprocess", "init") if command is full else ("collect", "retry", "reprocess")
        command.add_argument("command", nargs="?", default="ingest" if command is full else "collect", choices=choices)
        command.add_argument("--config", default=str(DEFAULT_CONFIG))
        selection = command.add_mutually_exclusive_group()
        selection.add_argument("--match-ids", type=int, nargs="+")
        selection.add_argument("--match-file", dest="match_ids_file")
        command.add_argument("--retry-failures", action=argparse.BooleanOptionalAction, default=None)
        command.add_argument("--rebuild", dest="rebuild_outputs", action=argparse.BooleanOptionalAction, default=None)
        if command is end:
            command.add_argument("--csv", dest="export_csv", action=argparse.BooleanOptionalAction, default=None)
    full.add_argument("--interval", dest="sampling_interval_s", type=float)
    full.add_argument("--replay", dest="replay_path", help="One local replay; ID from filename or --match-ids")
    full.add_argument("--replay-dir", help="Local replay folder; never downloaded or deleted")
    full.add_argument("--limit", dest="batch_size", type=int, help="Maximum attempts, including failures (default 1)")
    end.add_argument("--manifest", dest="manifest_url", help="Manifest URL or local pinned JSON")
    end.add_argument("--matches", dest="match_limit", type=int)
    end.add_argument("--max-download-mib", type=float)
    end.add_argument("--max-files", type=int)
    end.add_argument("--retain-shards", action=argparse.BooleanOptionalAction, default=None)
    migration = datasets.add_parser("migrate-layout", help="Move legacy local data into the new dataset directories")
    migration.add_argument("--config", default=str(DEFAULT_CONFIG))
    migration.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        if args.dataset == "migrate-layout":
            from .layout import migrate_layout
            migrate_layout(args.config, dry_run=args.dry_run)
            return
        overrides = {key: value for key, value in vars(args).items()
                     if key not in ("dataset", "command", "config")}
        config = load_dataset_config(args.dataset.replace("-", "_"), args.config, overrides)
        if args.dataset == "full-match":
            import duckdb
            from .full_match import run
            try:
                result = run(config, args.command)
            except duckdb.Error as error:
                raise RuntimeError(f"Full-match database error: {error}") from error
        else:
            import duckdb
            from .end_match import run
            try:
                result = run(config, args.command)
            except duckdb.Error as error:
                raise RuntimeError(f"Data-lake database error: {error}") from error
        if result.get("failed", 0) or result.get("errors"):
            raise RuntimeError("Some collection work failed; see the dataset's failure log/summary and use retry")
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as error:
        log(f"ERROR: Pipeline stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
