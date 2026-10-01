from pathlib import Path

from agent.io import read_json
from agent.lake_storage import complete_lake_match
from agent.pipeline import complete
from collect_lake import OUTPUT_DIR as LAKE_DIR
from main import OUTPUT_DIR as DATA_DIR

PROJECT_DIR = Path(__file__).resolve().parent


def detailed_complete(folder):
    return complete(folder) and read_json(folder / "sources/status.json", {}).get("replay_parsed") is True


def count_complete(directory, check):
    if not directory.is_dir():
        return 0
    count = 0
    for folder in directory.iterdir():
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        try:
            count += bool(check(folder))
        except (OSError, ValueError, TypeError, AttributeError, EOFError):
            continue  # Incomplete or unreadable matches do not count.
    return count


def main():
    # Verify saved tables; detailed matches must also have a successful replay parse.
    detailed = count_complete(PROJECT_DIR / DATA_DIR / "raw", detailed_complete)
    lake = count_complete(PROJECT_DIR / LAKE_DIR, complete_lake_match)
    print(f"Full detailed matches: {detailed:,}")
    print(f"Full end-of-match matches: {lake:,}")


if __name__ == "__main__":
    main()
