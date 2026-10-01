# Lake settings. Run this file to collect full match records without replays.
N_MATCHES = 50 # 12 seconds per match, 200 matches ~ 40 minutes
OUTPUT_DIR = "./data/lake"
MATCH_IDS = None  # Explicit IDs override the count and selection filters.
GAME_MODE = "Normal"  # None includes all game modes, including StreetBrawl.
MATCH_MODE = None  # For example "Ranked", "Unranked", or None for all.
MIN_DURATION_S = 900
EXPECTED_PLAYERS = 12  # None allows any roster size.
START_DATE = None  # Inclusive UTC date/time, e.g. "2026-09-01".
END_DATE = None  # Exclusive UTC date/time, e.g. "2026-10-01".
MIN_MATCH_ID = None  # Inclusive lower bound.
MAX_MATCH_ID = None  # Inclusive upper bound.
REPROCESS = False
PAGE_SIZE = 100  # Match IDs per discovery query; data is fetched one match at a time.
MAX_CANDIDATES = None  # None allows up to max(100, N_MATCHES * 10) new candidates.
REQUEST_DELAY = 0.0  # Minimum seconds between SQL requests.
COMPRESSION_LEVEL = 19  # Lossless Zstandard, 1 through 22.


def main():
    from agent.lake import run
    return run(n_matches=N_MATCHES, output_dir=OUTPUT_DIR, match_ids=MATCH_IDS,
               game_mode=GAME_MODE, match_mode=MATCH_MODE, min_duration_s=MIN_DURATION_S,
               expected_players=EXPECTED_PLAYERS, start_date=START_DATE, end_date=END_DATE,
               min_match_id=MIN_MATCH_ID, max_match_id=MAX_MATCH_ID, reprocess=REPROCESS,
               page_size=PAGE_SIZE, max_candidates=MAX_CANDIDATES,
               request_delay=REQUEST_DELAY, compression_level=COMPRESSION_LEVEL)


if __name__ == "__main__":
    main()
