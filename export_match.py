"""Export a downloaded match's saved tables to CSV for inspection."""

import argparse
from pathlib import Path

from datapipe.dataset import GROUPS, read_match
from main import OUTPUT_DIR


# Edit these settings, then run this file directly (for example, Run in VS Code).
MATCH_ID = 109521468
TABLE = "player_state"  # Use "all" for every table, or e.g. "events" or "team_state".
DATA_DIR = OUTPUT_DIR  # Uses the data directory configured in main.py.

TABLES = tuple(table for group in GROUPS.values() for table in group)


def export_match(match_id, table="player_state", data_dir=OUTPUT_DIR):
    """Write one CSV per selected table under data_dir/csv/<match_id>/."""
    if table not in (*TABLES, "all"):
        raise ValueError(f"Unknown table: {table}")
    destination = Path(data_dir) / "csv" / str(int(match_id))
    paths = []
    for name in TABLES if table == "all" else (table,):
        frame = read_match(match_id, name, output_dir=data_dir)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{name}.csv"
        # UTF-8 BOM helps Excel recognize text; nested fields stay JSON strings.
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{name}: {len(frame):,} rows, {len(frame.columns)} columns -> {path}")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_id", type=int, nargs="?", default=MATCH_ID,
                        help="An already downloaded and processed match ID (default: MATCH_ID setting)")
    parser.add_argument("--table", choices=(*TABLES, "all"), default=TABLE,
                        help="Table to export, or all for a CSV per table (default: TABLE setting)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Data directory (default: DATA_DIR setting)")
    args = parser.parse_args()
    try:
        export_match(args.match_id, args.table, args.data_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
