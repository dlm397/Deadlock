"""Timestamped console output in the computer's local timezone."""
from datetime import datetime


def log(message, *, file=None, flush=True):
    timestamp = datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")
    for line in str(message).splitlines() or [""]:
        print(f"[{timestamp}] {line}", file=file, flush=flush)
