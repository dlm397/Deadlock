from pathlib import Path

# Collection settings. Relative paths are anchored to this file.
OUTPUT_DIR = "./data"
N_MATCHES = 1
MATCH_IDS = None  # Explicit IDs override automatic selection filters.
SAMPLE_INTERVAL = 20
SAVE_RAW_DEMOS = True
SAVE_RAW_JSON = True
PARSE_REPLAYS = True
REPROCESS = False
OFFLINE = False  # Rebuild from archived metadata and replay extracts.
COLLECTION = "detailed"  # "detailed" or "lake"

# Automatic selection; explicit MATCH_IDS bypass these filters.
MIN_DURATION_S = 900
EXPECTED_PLAYERS = 12
GAME_MODE = "Normal"
MATCH_MODE = None  # Lake collection only.
START_DATE = None  # UTC, inclusive; lake collection only.
END_DATE = None  # UTC, exclusive.
MIN_MATCH_ID = None
MAX_MATCH_ID = None

OUTPUT_DIR = (Path(__file__).resolve().parent / OUTPUT_DIR).resolve()

if __name__ == "__main__":
    import main as settings
    if COLLECTION == "lake":
        from datapipe.lake import run
    else:
        from datapipe.collect import run
    from datapipe.sources import report_error
    try:
        run(settings)
    except Exception as error:
        report_error(error, OUTPUT_DIR / "tables/collection_error.log", "Collection stopped")
        raise SystemExit(1) from None
