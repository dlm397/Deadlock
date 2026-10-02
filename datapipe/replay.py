import bz2
import gzip
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import zstandard

from .sources import read_json


def decompress(source, destination):
    with Path(source).open("rb") as stream, Path(destination).open("wb") as output:
        magic = stream.read(4)
        stream.seek(0)
        if magic == b"\x28\xb5\x2f\xfd":
            zstandard.ZstdDecompressor().copy_stream(stream, output)
        elif magic.startswith(b"BZh"):
            with bz2.BZ2File(stream) as decoded:
                shutil.copyfileobj(decoded, output, 1024 * 1024)
        elif magic == b"PBDE":
            shutil.copyfileobj(stream, output, 1024 * 1024)
        else:
            raise ValueError("Unknown replay compression")


def parse_replay(compressed, folder, interval=20):
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_suffix(".mjs")
    with tempfile.TemporaryDirectory(prefix=".parse-", dir=folder.parent) as temporary:
        expanded = Path(temporary) / "replay.dem"
        parsed = Path(temporary) / "replay"
        decompress(compressed, expanded)
        log_path = folder.parent / "parser.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(["node", "--max-old-space-size=6144", str(script), str(expanded.resolve()),
                                     str(parsed.resolve()), str(interval)], stdout=log, stderr=subprocess.STDOUT)
        summary = read_json(parsed / "summary.json", {})
        if result.returncode or not summary.get("success"):
            # Keep the complete child output on disk; expose only its useful diagnosis.
            output = log_path.read_text(encoding="utf-8", errors="replace")
            reason = summary.get("error") or next((line.strip() for line in output.splitlines()
                if re.match(r"^(?:\w*Error:|Error \[)", line.strip())), "See parser.log for the parser's error")
            reason = str(reason).splitlines()[0]
            version = f" (game version {summary['game_version']})" if summary.get("game_version") else ""
            raise RuntimeError(f"Replay parsing failed{version}: {reason[:220]}. See parser.log for details.")
        required = ("player_state.ndjson.gz", "events.ndjson.gz", "entity_samples.ndjson.gz", "schema.json")
        assert summary.get("player_state", 0) > 0 and all((parsed / name).is_file() for name in required), \
            "Parser returned incomplete replay extracts"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").unlink(missing_ok=True)
        (folder / "summary.json.gz").unlink(missing_ok=True)
        for source in sorted(parsed.iterdir(), key=lambda p: p.name == "summary.json"):
            if source.suffix == ".json":
                with source.open("rb") as inp, gzip.open(folder / (source.name + ".gz"), "wb", compresslevel=9) as out:
                    shutil.copyfileobj(inp, out)
                (folder / source.name).unlink(missing_ok=True)
            else:
                source.replace(folder / source.name)
    return folder
