import main as settings
from agent.pipeline import run


def main():
    return run(settings.N_MATCHES, settings.SAMPLE_INTERVAL, settings.OUTPUT_DIR,
               settings.MATCH_IDS, settings.SAVE_RAW_DEMOS, settings.SAVE_RAW_JSON,
               settings.PARSE_REPLAYS, settings.QUERY_DEMO_API,
               settings.REPROCESS, settings.OFFLINE)


if __name__ == "__main__":
    main()
