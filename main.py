# User settings. Run collect.py after editing.
N_MATCHES = 10
SAMPLE_INTERVAL = 20  # game seconds between replay observations
OUTPUT_DIR = "./data"
MATCH_IDS = None  # e.g. [107544778, 107535709]; overrides N_MATCHES
SAVE_RAW_DEMOS = True
SAVE_RAW_JSON = True
PARSE_REPLAYS = True
QUERY_DEMO_API = True  # Optional independent kill-count check; API rate limits apply.
REPROCESS = False  # Rebuild completed matches even when code/settings are unchanged.
OFFLINE = False  # Rebuild local extracts without downloading or parsing demos.

if __name__ == "__main__":
    from collect import main
    main()
