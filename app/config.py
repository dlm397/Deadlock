"""Shared, sectioned configuration for the two independent datasets."""

import math
from pathlib import Path
import tomllib
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "pipelines.toml"


def resolve_saved_path(value, config=DEFAULT_CONFIG, *, base=None):
    """Resolve portable data/... references and paths saved before the repo moved.

    Only known project roots are rebased; explicit paths elsewhere stay explicit.
    Relative non-data paths use the repository unless a CLI supplies its cwd.
    Historical checkpoints need not be rewritten (their hashes stay valid).
    """
    path = Path(value).expanduser()
    relative = path if not path.is_absolute() else None
    if path.is_absolute():
        old_desktop = Path.home() / "OneDrive - Drexel University" / "Desktop" / "VS Code"
        for project in (ROOT, old_desktop / "Deadlock", old_desktop / "Deadlock2"):
            if path.is_relative_to(project):
                relative = path.relative_to(project)
                break
    if relative is not None:
        if relative.parts and relative.parts[0].lower() == "data":
            root = data_paths(config)["data_dir"]
            target = (root / Path(*relative.parts[1:])).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Saved data path escapes data_dir")
            return target
        if relative.parts[:2] == ("research", "model"):
            relative = Path("research/model_one") / Path(*relative.parts[2:])
        return ((ROOT if path.is_absolute() or base is None else Path(base)) / relative).resolve()
    return path.resolve()


def model_output_dir(args, stage):
    """Keep explicit CLI outputs, otherwise use this invocation's shared config."""
    if args.output_dir is not None:
        return resolve_saved_path(args.output_dir, args.config, base=Path.cwd())
    return data_paths(args.config)["supplemental"] / stage


def read_config(path=DEFAULT_CONFIG):
    path = resolve_saved_path(path, base=Path.cwd())
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    for section in ("paths", "full_match", "end_match"):
        if not isinstance(document.get(section), dict):
            raise ValueError(f"Missing [{section}] section in {path}; use config/pipelines.toml")
    return path, document


def data_paths(path=DEFAULT_CONFIG):
    path, document = read_config(path)
    root = (path.parent / document["paths"]["data_dir"]).resolve()
    paths = {"data_dir": root}
    for name, value in (("full_match", document["full_match"]["output_dir"]),
                        ("end_match", document["end_match"]["output_dir"]),
                        ("supplemental", document["paths"]["supplemental_dir"])):
        target = (root / value).resolve()
        if target == root or not target.is_relative_to(root):
            raise ValueError(f"{name} must be a subdirectory of data_dir")
        paths[name] = target
    names = ("full_match", "end_match", "supplemental")
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if paths[first].is_relative_to(paths[second]) or paths[second].is_relative_to(paths[first]):
                raise ValueError(f"Dataset/supplemental directories must not overlap: {first}, {second}")
    return paths


def load_dataset_config(dataset, path=DEFAULT_CONFIG, overrides=None):
    if dataset not in ("full_match", "end_match"):
        raise ValueError(f"Unknown dataset: {dataset}")
    path, document = read_config(path)
    paths = data_paths(path)
    config = dict(document[dataset])
    if "end_match_only" in config:
        raise ValueError("end_match_only was replaced by the full-match and end-match launch commands")
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    config.update(overrides)
    config.update(output_dir=paths[dataset], supplemental_dir=paths["supplemental"], data_dir=paths["data_dir"])
    boolean_names = ["retry_failures", "rebuild_outputs"]
    boolean_names += ["disable_steam"] if dataset == "full_match" else ["export_csv", "retain_shards"]
    for name in boolean_names:
        if type(config.get(name)) is not bool:
            raise ValueError(f"[{dataset}].{name} must be true or false")
    integer_names = ["batch_size"] if dataset == "full_match" else ["match_limit", "max_files"]
    for name in integer_names:
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"[{dataset}].{name} must be a positive integer")
    number = "sampling_interval_s" if dataset == "full_match" else "max_download_mib"
    value = config.get(number)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"[{dataset}].{number} must be positive and finite")
    ids = config.get("match_ids", [])
    input_file = config.get("match_ids_file")
    if input_file and "match_ids" not in overrides:
        base = Path.cwd() if "match_ids_file" in overrides else path.parent
        file = resolve_saved_path(input_file, path, base=base)
        text = " ".join(line.split("#", 1)[0] for line in file.read_text(encoding="utf-8-sig").splitlines())
        ids = [int(token) for token in text.replace(",", " ").split()]
    if not isinstance(ids, list) or any(type(mid) is not int or not 0 < mid < 2**64 for mid in ids):
        raise ValueError("match_ids must be positive exact uint64 integers")
    config["match_ids"] = list(dict.fromkeys(ids))
    if dataset == "full_match":
        config["sampling_interval_explicit"] = "sampling_interval_s" in overrides
        if config.get("export_csv") or config.get("retain_replays") is False:
            raise ValueError("Full-match writes only DuckDB and always retains raw replays")
        replay_dir = config.get("replay_dir")
        base = Path.cwd() if "replay_dir" in overrides else paths["data_dir"]
        config["replay_dir"] = (resolve_saved_path(replay_dir, path, base=base)
                                if replay_dir else paths["full_match"] / "replays")
        if config.get("replay_path"):
            config["replay_path"] = resolve_saved_path(config["replay_path"], path, base=Path.cwd())
    if dataset == "end_match":
        manifest = str(config["manifest_url"])
        if urlparse(manifest).scheme not in ("http", "https"):
            base = Path.cwd() if "manifest_url" in overrides else path.parent
            config["manifest_url"] = str(resolve_saved_path(manifest, path, base=base))
    return config
