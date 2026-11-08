"""Split conformal winner sets for the frozen Stage 3 model; no retraining.

Only ``summary.tex`` is written by ``calibrate``. It contains the report, inline
PGFPlots figures, calibration state, and predictions needed to reproduce them.
No image, checkpoint, cache, log, or split sidecars are created.

    python -B conformal.py self-test
    python -B conformal.py calibrate --device cuda
    python -B conformal.py predict --probability 0.7
    python -B conformal.py predict --match-id 111578174 --through-time 600

Programmatic use: wrapper, state = load_conformal(); wrapper.predict_batch(batch)
returns the original logit/probabilities and a boolean [batch, 2] membership mask
ordered [Team0, Team1]. Scalar probability prediction does not require a label.
"""

import sys

# Also suppress bytecode in locally imported training/data modules.
sys.dont_write_bytecode = True

import argparse
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from zoneinfo import ZoneInfo

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import finetune_win as win
from app.config import data_paths, resolve_saved_path


ROOT = Path(__file__).resolve().parent
CHECKPOINT_WITHIN_SUPPLEMENTAL = (
    "finetune_win/"
    "run_20261029T180904763762Z_seed42/best_model.pt"
)
DEFAULT_CHECKPOINT = data_paths()["supplemental"] / CHECKPOINT_WITHIN_SUPPLEMENTAL
REPORT = ROOT / "summary.tex"
SOURCE = "https://arxiv.org/abs/2107.07511"
ALPHA_GRID = (0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def probabilities_array(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected a finite one-dimensional array of P(Team1) in [0, 1]")
    return values


def labels_array(labels, count):
    values = np.asarray(labels)
    if values.shape != (count,) or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected one binary winning-team label per probability")
    return values.astype(np.int64)


class SplitConformal:
    """Non-randomized likelihood-score sets; ties are included, empty sets kept."""

    def __init__(self, scores, alpha=0.1):
        self.scores = probabilities_array(scores).copy()
        self.alpha = float(alpha)
        if not len(self.scores) or not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("Need nonempty calibration scores and 0 < alpha < 1")
        # Decimal avoids a spurious extra rank at floating-point integer boundaries.
        rank = Decimal(len(self.scores) + 1) * (Decimal(1) - Decimal(str(self.alpha)))
        self.rank = int(rank.to_integral_value(rounding=ROUND_CEILING))
        self.threshold = (1.0 if self.rank > len(self.scores)
                          else float(np.sort(self.scores)[self.rank - 1]))

    @classmethod
    def fit(cls, probabilities, labels, alpha=0.1):
        probabilities = probabilities_array(probabilities)
        labels = labels_array(labels, len(probabilities))
        scores = np.where(labels == 1, 1.0 - probabilities, probabilities)
        return cls(scores, alpha)

    def predict_sets(self, probabilities):
        probabilities = probabilities_array(probabilities)
        # Candidate scores: Team0 -> p1; Team1 -> 1-p1.
        scores = np.column_stack((probabilities, 1.0 - probabilities))
        return scores <= self.threshold


class ConformalWinModel:
    """Inference-only wrapper; the base model and its probabilities are unchanged."""

    def __init__(self, model, calibrator, device="cpu"):
        self.model = model.to(device).eval().requires_grad_(False)
        self.calibrator = calibrator
        self.device = device

    @torch.inference_mode()
    def predict_batch(self, batch):
        self.model.eval()
        logits = win.dynamics.forward_batch(self.model, batch, self.device)
        team1 = logits.double().sigmoid()
        probabilities = torch.stack((1 - team1, team1), dim=1)
        scores = torch.stack((team1, 1 - team1), dim=1)
        return {"logits": logits, "probabilities": probabilities,
                "prediction_sets": scores <= self.calibrator.threshold}


def partition_matches(match_ids, seed=2026, calibration_fraction=0.5):
    ids = sorted(int(mid) for mid in match_ids)
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError("Need at least two distinct heldout matches")
    if not math.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
        raise ValueError("calibration-fraction must be between 0 and 1")
    random.Random(seed).shuffle(ids)
    count = math.floor(len(ids) * calibration_fraction)
    if not 0 < count < len(ids):
        raise ValueError("Both calibration and evaluation partitions must be nonempty")
    return sorted(ids[:count]), sorted(ids[count:])


def wilson_interval(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    fraction = successes / count
    denominator = 1 + z * z / count
    middle = (fraction + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, middle - radius), min(1.0, middle + radius)]


def set_metrics(calibrator, probabilities, labels):
    probabilities = probabilities_array(probabilities)
    labels = labels_array(labels, len(probabilities))
    if not len(labels):
        raise ValueError("Evaluation must contain at least one match")
    membership = calibrator.predict_sets(probabilities)
    sizes = membership.sum(axis=1)
    covered = membership[np.arange(len(labels)), labels]
    singletons = sizes == 1
    singleton_count = int(singletons.sum())
    return {
        "alpha": calibrator.alpha, "nominal_coverage": 1 - calibrator.alpha,
        "threshold": calibrator.threshold, "quantile_rank": calibrator.rank,
        "count": len(labels), "covered": int(covered.sum()),
        "coverage": float(covered.mean()),
        "coverage_wilson_95": wilson_interval(int(covered.sum()), len(labels)),
        "mean_set_size": float(sizes.mean()),
        "set_size_counts": [int((sizes == size).sum()) for size in (0, 1, 2)],
        "singleton_accuracy": (float(covered[singletons].mean()) if singleton_count else None),
        "argmax_accuracy": float(((probabilities >= 0.5).astype(int) == labels).mean()),
    }


def checkpoint_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_report(text):
    # Machine-readable state is a LaTeX comment, so there is no third artifact.
    match = re.search(r"^% CONFORMAL_STATE (.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Report has no embedded conformal calibration state")
    state = json.loads(match.group(1))
    if state.get("format_version") != 1 or state.get("score") != "1-p_true":
        raise ValueError("Unsupported conformal report format")
    calibration_ids = [row["match_id"] for row in state["calibration_rows"]]
    evaluation_ids = [row["match_id"] for row in state["evaluation_rows"]]
    expected = partition_matches(calibration_ids + evaluation_ids, state["split_seed"],
                                 state["calibration_fraction"])
    if expected != (sorted(calibration_ids), sorted(evaluation_ids)):
        raise ValueError("Embedded calibration/evaluation assignments are inconsistent")
    calibrator = SplitConformal.fit(
        [row["team1_win_probability"] for row in state["calibration_rows"]],
        [row["label"] for row in state["calibration_rows"]], state["alpha"])
    if calibrator.threshold != state["metrics"]["threshold"]:
        raise ValueError("Saved threshold does not match calibration predictions")
    return calibrator, state


def read_report(path=REPORT):
    return parse_report(Path(path).read_text(encoding="utf-8"))


def load_conformal(report=REPORT, device="cpu", checkpoint=None, config=win.DEFAULT_CONFIG):
    calibrator, state = read_report(report)
    path = resolve_saved_path(checkpoint if checkpoint is not None else state["checkpoint"], config,
                              base=Path.cwd() if checkpoint is not None else ROOT)
    if checkpoint_digest(path) != state["checkpoint_sha256"]:
        raise ValueError("Model checkpoint differs from the one used for calibration")
    model, _ = win.load_stage3(path, device)
    return ConformalWinModel(model, calibrator, device), state


def require_heldout_ids(checkpoint):
    ids = checkpoint["active_match_ids"]["test"]
    used = set(checkpoint["active_match_ids"]["train"]) | set(checkpoint["active_match_ids"]["validation"])
    if used.intersection(ids) or any(checkpoint["split_assignments"].get(str(mid)) != "test" for mid in ids):
        raise ValueError("Calibration candidates overlap training/validation or have invalid splits")
    return ids


@torch.inference_mode()
def collect_predictions(model, checkpoint, database, ids, prefix_seed, device):
    matches, skipped = win.load_win_matches(database, ids)
    if skipped or {m["match_id"] for m in matches} != set(ids):
        raise ValueError("Some heldout matches are unavailable; refusing to silently change the cohort")
    matches = win.prepare_groups(matches, checkpoint)["test"]
    audit = win.attach_eligible_times(database, matches, checkpoint["preprocessing"]["frame_stride"])
    if audit["excluded_matches"]:
        raise ValueError("Some heldout matches have no eligible nonterminal prefix")
    dataset = win.PrefixDataset(database, sorted(matches, key=lambda m: m["match_id"]),
                                checkpoint["static_preprocessing"], checkpoint["preprocessing"],
                                seed=prefix_seed, fixed=True)
    rows = []
    for index, batch in enumerate(win.prefix_loader(dataset, batch_size=1), 1):
        logit = win.dynamics.forward_batch(model, batch, device).cpu().double()[0]
        rows.append({"match_id": int(batch["match_id"][0]),
                     "cutoff_time_s": float(batch["cutoff_time_s"][0]),
                     "label": int(batch["labels"][0]), "logit": logit.item(),
                     "team1_win_probability": logit.sigmoid().item()})
        if index % 10 == 0 or index == len(dataset):
            print(f"Inferred {index}/{len(dataset)} heldout prefixes", flush=True)
    return rows


def render_report(state):
    """Self-contained LaTeX/PGFPlots; no plotting library or image cache needed."""
    metrics = state["metrics"]
    count = metrics["count"]
    counts = metrics["set_size_counts"]
    nominal = 100 * metrics["nominal_coverage"]
    coverage = 100 * metrics["coverage"]
    low, high = [100 * value for value in metrics["coverage_wilson_95"]]
    singleton = ("Not applicable (no singleton sets)" if metrics["singleton_accuracy"] is None
                 else f"{100 * metrics['singleton_accuracy']:.1f}\\%")
    bars = " ".join(f"({size},{100 * number / count:.6f})" for size, number in enumerate(counts))
    curve = " ".join(f"({100 * row['nominal_coverage']:.6f},{100 * row['coverage']:.6f})"
                     for row in sorted(state["sensitivity"], key=lambda row: row["nominal_coverage"]))
    x_min = max(0, min(100 * row["nominal_coverage"] for row in state["sensitivity"]) - 3)
    y_min = max(0, min(100 * min(row["nominal_coverage"], row["coverage"])
                       for row in state["sensitivity"]) - 5)
    location = state["checkpoint"].replace("\\", "/")
    conclusion = (
        f"At the preselected {nominal:g}\\% target, the sets contained the winner in "
        f"{metrics['covered']} of {count} evaluation matches ({coverage:.1f}\\%). "
        f"They named one team in {counts[1]} matches and both teams in {counts[2]}; "
        f"{counts[0]} sets were empty. "
        "Returning both teams expresses uncertainty but does not choose a winner. "
        "This wrapper does not improve or recalibrate the original win probabilities."
    )
    text = r"""\documentclass[10pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,booktabs,tabularx,pgfplots}
\usepackage[hidelinks]{hyperref}
\pgfplotsset{compat=1.18}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\title{Conformal Winner Sets: A First Evaluation}
\author{}\date{@DATE@}
\begin{document}
\maketitle
\vspace{-2em}
\textbf{Setup.} I froze the selected Stage 3 model (epoch @EPOCH@) and split
its previously unused test matches into @NCAL@ calibration and @NEVAL@ evaluation
matches without looking at outcomes (split seed @SPLITSEED@). I sampled one
eligible nonterminal prefix per match (prefix seed @PREFIXSEED@), retaining the
model's preprocessing and match assignments. Training and model-selection
validation matches were excluded. No weights were trained or changed.

\textbf{Method.} I used split conformal classification with score
$s_i=1-p(y_i\mid x_i)$ and the $k=\lceil(n+1)(1-\alpha)\rceil$th ordered
calibration score $q$. A team is included when $1-p(y\mid x)\le q$; ties are
included and empty sets are retained. If $k>n$, both teams are included.
I chose $\alpha=@ALPHA@$ before evaluating outcomes. Under exchangeability of
calibration and future match/prefix pairs, coverage is marginal, not guaranteed
at every elapsed time, for each team, or simultaneously across a match.
See \href{@SOURCE@}{Angelopoulos and Bates} for the method.

\begin{center}\small
\begin{tabularx}{0.95\linewidth}{Xr}
\toprule
Statistic & Evaluation result \\
\midrule
Nominal / observed coverage & @NOMINAL@\% / @COVERAGE@\% \\
Observed coverage, descriptive 95\% Wilson interval & [@LOW@\%, @HIGH@\%] \\
Quantile rank / calibration count; threshold $q$ & @RANK@ / @NCAL@; @Q@ \\
Mean prediction-set size (maximum 2) & @SIZE@ \\
Empty / singleton / two-team sets & @COUNTS@ \\
Accuracy among singleton sets & @SINGLETON@ \\
Original model's argmax accuracy & @ACCURACY@\% \\
\bottomrule
\end{tabularx}\end{center}

\begin{center}
\begin{tikzpicture}
\begin{axis}[width=0.46\textwidth,height=4.6cm,ybar,bar width=18pt,
title={Prediction-set size},xlabel={Teams in set},ylabel={Evaluation matches (\%)},
xtick={0,1,2},ymin=0,ymax=105,enlarge x limits=0.3,
tick label style={font=\small},label style={font=\small},title style={font=\small},
nodes near coords,nodes near coords style={font=\scriptsize}]
\addplot[fill=teal!65,draw=teal!80!black] coordinates {@BARS@};
\end{axis}
\end{tikzpicture}\hfill
\begin{tikzpicture}
\begin{axis}[width=0.46\textwidth,height=4.6cm,
title={Coverage across fixed targets},xlabel={Nominal coverage (\%)},ylabel={Observed coverage (\%)},
xmin=@XMIN@,xmax=101,ymin=@YMIN@,ymax=102,grid=major,
tick label style={font=\small},label style={font=\small},title style={font=\small},
legend style={font=\scriptsize,at={(0.03,0.97)},anchor=north west}]
\addplot[gray,dashed] coordinates {(@XMIN@,@XMIN@) (100,100)};
\addlegendentry{Nominal}
\addplot[teal!80!black,mark=*] coordinates {@CURVE@};
\addlegendentry{Observed}
\addplot[red,only marks,mark=*,mark size=3pt] coordinates {(@NOMINAL@,@COVERAGE@)};
\addlegendentry{Primary target}
\end{axis}
\end{tikzpicture}
\end{center}
{\small The target sweep is descriptive, not a search for the best target on
evaluation data. All points reuse the same calibration and evaluation matches.}

\textbf{What I learned.} @CONCLUSION@
Accuracy among singleton sets is a selective diagnostic; the nominal coverage
target does not guarantee that accuracy on this subset.

\textbf{Limits and next steps.} The small evaluation sample gives a wide
uncertainty interval; observed coverage need not equal the nominal target in a
particular split. The Wilson interval assumes independent matches, and shared
players or patch changes could weaken that assumption and exchangeability.
These formerly heldout matches have now been used for calibration or evaluation;
I would collect fresh matches before further tuning. I would also check coverage
by elapsed time on a larger sample and compare efficiency across predeclared
scores. This experiment does not establish conditional or timeline-wide coverage.

\textbf{Reuse.} Only this report and \texttt{conformal.py} are new artifacts.
The report embeds the calibration state, exact split/prefix predictions, settings,
and checkpoint hash in a comment. Reload verifies that hash and reconstructs
the threshold. Numerical checks and original-model reload verification: @CHECKS@.
Runtime: @SECONDS@ seconds on \texttt{@DEVICE@}.
\begin{verbatim}
python -B conformal.py calibrate --device cuda
python -B conformal.py predict --probability 0.7
python -B conformal.py predict --match-id 111578174 --through-time 600
python -B conformal.py self-test
\end{verbatim}
{\footnotesize Checkpoint: \path{@CHECKPOINT@}. Team labels are
\texttt{Team0} and \texttt{Team1}; the sigmoid output is $P(\mathrm{Team1\ wins})$.
Compile this file with \texttt{pdflatex summary.tex} to view the embedded plots;
compilation creates PDF/auxiliary files only when explicitly run.}
\end{document}
"""
    substitutions = {
        "DATE": state["created_at"].split("T")[0], "NCAL": len(state["calibration_rows"]),
        "NEVAL": count, "SPLITSEED": state["split_seed"], "PREFIXSEED": state["prefix_seed"],
        "ALPHA": state["alpha"], "SOURCE": SOURCE, "NOMINAL": f"{nominal:g}",
        "COVERAGE": f"{coverage:.1f}", "LOW": f"{low:.1f}", "HIGH": f"{high:.1f}",
        "RANK": metrics["quantile_rank"], "Q": f"{metrics['threshold']:.6f}",
        "SIZE": f"{metrics['mean_set_size']:.3f}", "COUNTS": " / ".join(map(str, counts)),
        "SINGLETON": singleton, "ACCURACY": f"{100 * metrics['argmax_accuracy']:.1f}",
        "BARS": bars, "CURVE": curve, "CONCLUSION": conclusion,
        "CHECKS": "passed" if state["checks_passed"] else "failed",
        "SECONDS": f"{state['elapsed_seconds']:.1f}", "DEVICE": state["device"],
        "CHECKPOINT": location, "EPOCH": state["best_epoch"],
        "XMIN": f"{x_min:.2f}", "YMIN": f"{y_min:.2f}",
    }
    for name, value in substitutions.items():
        text = text.replace(f"@{name}@", str(value))
    if re.search(r"@[A-Z]+@", text):
        raise ValueError("Unsubstituted report field")
    return text + "\n% CONFORMAL_STATE " + json.dumps(state, separators=(",", ":"), allow_nan=False) + "\n"


def calibrate(args):
    args.checkpoint = (resolve_saved_path(args.checkpoint, args.config, base=Path.cwd()) if args.checkpoint
                       else data_paths(args.config)["supplemental"] / CHECKPOINT_WITHIN_SUPPLEMENTAL)
    started = time.perf_counter()
    self_test()
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, checkpoint = win.load_stage3(args.checkpoint, args.device)
    digest = checkpoint_digest(args.checkpoint)
    calibration_ids, evaluation_ids = partition_matches(require_heldout_ids(checkpoint), args.seed,
                                                        args.calibration_fraction)
    # Freeze the score, target, split and prefix policy before reading outcomes.
    SplitConformal([0.5], args.alpha)
    print(f"Frozen protocol: alpha={args.alpha}, calibration={len(calibration_ids)}, "
          f"evaluation={len(evaluation_ids)}, split seed={args.seed}, prefix seed={args.prefix_seed}", flush=True)
    database = win.dynamics.resolve_database(args)
    rows = collect_predictions(model, checkpoint, database, calibration_ids + evaluation_ids,
                               args.prefix_seed, args.device)
    by_id = {row["match_id"]: row for row in rows}
    calibration_rows = [by_id[mid] for mid in calibration_ids]
    evaluation_rows = [by_id[mid] for mid in evaluation_ids]
    calibrator = SplitConformal.fit([row["team1_win_probability"] for row in calibration_rows],
                                   [row["label"] for row in calibration_rows], args.alpha)
    probabilities = [row["team1_win_probability"] for row in evaluation_rows]
    labels = [row["label"] for row in evaluation_rows]
    metrics = set_metrics(calibrator, probabilities, labels)
    sensitivity = [set_metrics(SplitConformal(calibrator.scores, alpha), probabilities, labels)
                   for alpha in sorted(set(ALPHA_GRID + (args.alpha,)))]
    # Original reference checks the model without consuming more heldout examples.
    del model
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    win.verify_reload(args.checkpoint)
    try:
        checkpoint_name = args.checkpoint.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        checkpoint_name = str(args.checkpoint.resolve())
    state = {
        "format_version": 1, "score": "1-p_true", "alpha": args.alpha,
        "checkpoint": checkpoint_name, "checkpoint_sha256": digest,
        "database": str(database), "device": args.device,
        "split_seed": args.seed, "prefix_seed": args.prefix_seed,
        "calibration_fraction": args.calibration_fraction,
        "prefix_policy": "one uniform eligible nonterminal prefix per match; fixed seeded cutoff",
        "calibration_rows": calibration_rows, "evaluation_rows": evaluation_rows,
        "metrics": metrics, "sensitivity": sensitivity,
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "checks_passed": False,
        "best_epoch": checkpoint["best_epoch"],
        "torch_version": str(torch.__version__), "method_reference": SOURCE,
        "test_partition_consumed": True,
    }
    # Verify serialization and the wrapper before declaring success or writing.
    restored, _ = parse_report(render_report(state))
    np.testing.assert_array_equal(restored.predict_sets(probabilities), calibrator.predict_sets(probabilities))
    if checkpoint_digest(args.checkpoint) != digest:
        raise RuntimeError("Checkpoint changed during calibration")
    restored_model, _ = win.load_stage3(args.checkpoint)
    wrapper = ConformalWinModel(restored_model, restored)
    reference = torch.load(args.checkpoint.parent / "reload_reference.pt", weights_only=True, map_location="cpu")
    output = wrapper.predict_batch(reference["batch"])
    torch.testing.assert_close(output["logits"], reference["logits"], rtol=0, atol=0)
    np.testing.assert_array_equal(output["prediction_sets"].numpy(),
                                  restored.predict_sets(output["probabilities"][:, 1].numpy()))
    state["checks_passed"] = True
    state["elapsed_seconds"] = time.perf_counter() - started
    # This is the only output write in the program. No split/model file is updated.
    REPORT.write_text(render_report(state), encoding="utf-8")
    saved, _ = read_report(REPORT)
    np.testing.assert_array_equal(saved.predict_sets(probabilities), calibrator.predict_sets(probabilities))
    print(json.dumps({"report": str(REPORT), "metrics": metrics,
                      "reload_sets_identical": True, "model_logits_unchanged": True}, indent=2))
    return state


def predict(args):
    calibrator, state = read_report(args.report)
    if args.probability is not None:
        probability = float(probabilities_array([args.probability])[0])
        result = {"input": "user-supplied probability; must come from the calibrated checkpoint"}
    else:
        if args.through_time is None:
            raise ValueError("--through-time is required with --match-id")
        # Existing predictor validates prefix eligibility and applies saved scaling.
        args.checkpoint = resolve_saved_path(state["checkpoint"], args.config)
        if checkpoint_digest(args.checkpoint) != state["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after conformal calibration")
        if args.database is None:
            args.database = resolve_saved_path(state["database"], args.config)
        result = win.predict(args)
        probability = result["team1_win_probability"]
        result["coverage_scope"] = "marginal for calibration prefix policy; arbitrary elapsed times have no conditional guarantee"
    included = calibrator.predict_sets([probability])[0]
    result.update(team0_win_probability=1 - probability, team1_win_probability=probability,
                  prediction_set=[f"Team{team}" for team in (0, 1) if included[team]],
                  set_size=int(included.sum()), nominal_coverage=1 - calibrator.alpha,
                  threshold=calibrator.threshold)
    return result


def self_test():
    """Small numerical checks kept in this file to avoid extra test artifacts."""
    assert SplitConformal(np.linspace(0, 1, 67), 0.1).rank == 62
    assert SplitConformal(np.arange(9) / 10, 0.1).threshold == 0.8
    assert SplitConformal([0.1] * 5, 0.01).predict_sets([0, 0.5, 1]).all()
    ties = SplitConformal([0.5] * 9, 0.1)
    assert ties.predict_sets([0.5]).all()
    empty = SplitConformal([0.3] * 9, 0.1).predict_sets([0.5])
    assert not empty.any()
    assert set_metrics(SplitConformal([0.3] * 9, 0.1), [0.5], [1])["coverage"] == 0
    assert SplitConformal.fit([0.9, 0.2], [1, 0]).scores.tolist() == [1 - 0.9, 0.2]
    np.testing.assert_array_equal(ties.predict_sets([0, 1]), [[True, False], [False, True]])
    assert set_metrics(ties, [0.5], [0])["singleton_accuracy"] is None
    scores, probabilities = np.linspace(0, 1, 67), np.linspace(0, 1, 101)
    previous = np.zeros((101, 2), dtype=bool)
    for alpha in reversed(ALPHA_GRID):
        current = SplitConformal(scores, alpha).predict_sets(probabilities)
        assert np.all(~previous | current)
        previous = current
    a, b = partition_matches(range(134))
    assert len(a) == len(b) == 67 and not set(a).intersection(b)
    assert (a, b) == partition_matches(reversed(range(134)))
    for invalid in (lambda: SplitConformal([], 0.1), lambda: SplitConformal([0.5], 0),
                    lambda: SplitConformal([0.5], 1), lambda: SplitConformal([0.5], float("nan")),
                    lambda: SplitConformal([float("nan")]), lambda: ties.predict_sets([1.1]),
                    lambda: ties.predict_sets([[0.5]]), lambda: SplitConformal.fit([0.5], [2]),
                    lambda: partition_matches([1, 1])):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    return "Numerical conformal checks passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="Calibrate/evaluate frozen model; write only summary.tex")
    prediction = commands.add_parser("predict", help="Reload embedded calibration; output a winner set")
    commands.add_parser("self-test", help="Run in-memory numerical checks without writing files")
    for command in (calibration, prediction):
        command.add_argument("--device", default="cpu")
        command.add_argument("--database", type=Path)
        command.add_argument("--config", type=Path, default=win.DEFAULT_CONFIG)
    calibration.add_argument("--checkpoint", type=Path,
                             help="Default: second fine-tuning checkpoint under configured supplemental")
    calibration.add_argument("--alpha", type=float, default=0.1)
    calibration.add_argument("--seed", type=int, default=2026, help="Label-blind calibration/evaluation split seed")
    calibration.add_argument("--prefix-seed", type=int, default=2026)
    calibration.add_argument("--calibration-fraction", type=float, default=0.5)
    prediction.add_argument("--report", type=Path, default=REPORT)
    inputs = prediction.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--probability", type=float)
    inputs.add_argument("--match-id", type=int)
    prediction.add_argument("--through-time", type=float)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args)
    elif args.command == "predict":
        print(json.dumps(predict(args), indent=2, allow_nan=False))
    else:
        print(self_test())


if __name__ == "__main__":
    main()
