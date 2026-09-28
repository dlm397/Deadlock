# Optional descriptive plots for one saved match.

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from main import OUTPUT_DIR
from dataset import match_ids, read_match


EXAMPLE_MATCH_ID = None  # None selects the first saved match.
TEAM_COLORS = ("#b67916", "#315fbb", "#3b8563", "#994c94")


def _match(frame, match_id):
    if frame.empty or "match_id" not in frame:
        return pd.DataFrame(columns=frame.columns)
    return frame.loc[frame["match_id"].astype(str) == str(match_id)].copy()


def _column(frame, *names):
    return next((name for name in names if name in frame), None)


def _empty(ax, message):
    ax.text(0.5, 0.5, textwrap.fill(message, 75), transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, output, filename, note):
    fig.text(0.015, 0.015, textwrap.fill(note, 150), fontsize=8, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(output / filename, dpi=160, facecolor="white")
    plt.close(fig)


def _style(ax, ylabel=None):
    ax.set_xlabel("Game time (minutes)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def _state_note(states):
    if states.empty:
        return "No player-state observations were available."
    age_column = _column(states, "state_age", "state_age_s")
    source_column = _column(states, "state_source", "source")
    note = "Missing values remain gaps."
    if source_column:
        sources = sorted(str(value) for value in states[source_column].dropna().unique())
        note += " Sources: " + ", ".join(sources) + "."
    if age_column:
        ages = pd.to_numeric(states[age_column], errors="coerce").dropna()
        if not ages.empty and ages.max() > 0:
            note += f" States carry earlier observations forward (median age {ages.median():.0f}s; maximum {ages.max():.0f}s). The sampling interval is not the source frequency."
    return note


def make_plots(output_dir=OUTPUT_DIR, match_id=EXAMPLE_MATCH_ID):
    root = Path(output_dir)
    output = root / "processed/plots"
    output.mkdir(parents=True, exist_ok=True)
    if match_id is None:
        available = match_ids(root)
        match_id = available[0] if available else None
    tables = {name: read_match(match_id, name, output_dir=root) if match_id is not None else pd.DataFrame()
              for name in ("matches", "players", "player_state", "team_state", "purchases", "events")}
    matches = tables["matches"]
    if match_id is None and not matches.empty and "match_id" in matches:
        match_id = matches.iloc[0]["match_id"]
        if isinstance(match_id, float) and match_id.is_integer():
            match_id = int(match_id)
    if match_id is None:
        print("Analysis: no collected match is available. Writing annotated empty plots.")
    tables = {name: _match(frame, match_id) for name, frame in tables.items()}
    players, states, teams = (tables[name] for name in ("players", "player_state", "team_state"))
    for frame in tables.values():
        if "game_time" in frame:
            frame["minutes"] = pd.to_numeric(frame["game_time"], errors="coerce") / 60
    note = _state_note(states)
    title_suffix = f"Match {match_id}" if match_id is not None else "No collected match"

    def player_label(player_id):
        if not players.empty and "player_id" in players:
            rows = players.loc[players["player_id"] == player_id]
            if not rows.empty:
                row = rows.iloc[0]
                hero = row.get("hero_name")
                if not isinstance(hero, str) or not hero:
                    hero_id = row.get("hero_id")
                    hero = f"Hero {int(hero_id)}" if pd.notna(hero_id) else "Unknown hero"
                return f"{hero} / {player_id}"
        return str(player_id)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Player net worth · {title_suffix}")
    if {"minutes", "net_worth", "player_id"}.issubset(states.columns) and states["net_worth"].notna().any():
        colors = plt.get_cmap("tab20")
        for index, (player_id, rows) in enumerate(states.groupby("player_id", sort=False)):
            rows = rows.sort_values("minutes")
            ax.step(rows["minutes"], pd.to_numeric(rows["net_worth"], errors="coerce"), where="post", label=player_label(player_id), color=colors(index % 20), linewidth=1.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        _style(ax, "Net worth (souls)")
    else:
        _empty(ax, "Player net worth is unavailable in the saved state data.")
    _save(fig, output, "01_player_net_worth.png", note)

    team_value = _column(teams, "net_worth", "team_net_worth", "total_net_worth")
    totals = pd.DataFrame()
    if team_value and {"team", "minutes"}.issubset(teams.columns):
        valid = teams.dropna(subset=["team", "minutes"]).copy()
        valid[team_value] = pd.to_numeric(valid[team_value], errors="coerce")
        # A duplicate aggregate is a data problem, never silently sum it again.
        if not valid.duplicated(["minutes", "team"]).any():
            totals = valid.pivot(index="minutes", columns="team", values=team_value).sort_index()
        else:
            print("Analysis: duplicate team/time aggregates; team plots will show missing coverage.")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(f"Team net-worth totals · {title_suffix}")
    if not totals.empty and totals.notna().any().any():
        for index, team in enumerate(totals):
            ax.step(totals.index, totals[team], where="post", label=f"Team {team}", color=TEAM_COLORS[index % len(TEAM_COLORS)], linewidth=2)
        ax.legend()
        _style(ax, "Total net worth (souls)")
    else:
        _empty(ax, "Team net-worth totals are unavailable or do not have unique team/time rows.")
    _save(fig, output, "02_team_net_worth.png", note + " Team totals are the saved aggregates; unavailable player values must not be interpreted as zero.")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(f"Team net-worth differential · {title_suffix}")
    if len(totals.columns) == 2 and totals.notna().all(axis=1).any():
        team_a, team_b = totals.columns
        difference = totals[team_a] - totals[team_b]
        ax.step(totals.index, difference, where="post", linewidth=2, color=TEAM_COLORS[0])
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.fill_between(totals.index, difference, 0, where=difference >= 0, color=TEAM_COLORS[0], alpha=0.2, step="post")
        ax.fill_between(totals.index, difference, 0, where=difference < 0, color=TEAM_COLORS[1], alpha=0.2, step="post")
        _style(ax, f"Team {team_a} minus team {team_b} (souls)")
    else:
        _empty(ax, "A differential requires both teams' net worth at the same time; those observations are unavailable.")
    _save(fig, output, "03_team_net_worth_difference.png", note + " Difference is calculated from the two saved team totals, with no imputation.")

    purchases = tables["purchases"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Item transaction timing · {title_suffix}")
    if {"minutes", "player_id"}.issubset(purchases.columns) and not purchases.empty:
        transactions = purchases.dropna(subset=["minutes", "player_id"]).copy()
        order = list(players["player_id"].dropna().unique()) if "player_id" in players else []
        order.extend(player_id for player_id in transactions["player_id"].unique() if player_id not in order)
        positions = {player_id: index for index, player_id in enumerate(order)}
        transactions["position"] = transactions["player_id"].map(positions)
        if "transaction_type" not in transactions:
            transactions["transaction_type"] = "unknown transaction"
        transactions["transaction_type"] = transactions["transaction_type"].fillna("unknown transaction")
        for index, (kind, rows) in enumerate(transactions.groupby("transaction_type", sort=False)):
            marker = "x" if "sell" in str(kind).lower() or "sale" in str(kind).lower() else ["o", "s", "^", "D"][index % 4]
            ax.scatter(rows["minutes"], rows["position"], label=f"{kind} ({len(rows)})", marker=marker, s=19, alpha=0.65)
        ax.set_yticks(range(len(order)), [player_label(player_id) for player_id in order], fontsize=8)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        _style(ax)
    else:
        _empty(ax, "No timed item transactions were recovered for this match.")
    _save(fig, output, "04_item_transaction_timing.png", "Each marker is a retained transaction record. Overlapping records may coincide. Item names, IDs, and source details remain in purchases.parquet; unknown transaction types are not relabeled as purchases.")

    events = tables["events"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Major objective timeline · {title_suffix}")
    objective = pd.DataFrame()
    if {"minutes", "event_type"}.issubset(events.columns):
        text = events["event_type"].fillna("").astype(str)
        if "target" in events:
            text = text + " " + events["target"].astype("string").fillna("")
        objective = events.loc[text.str.contains("mid.?boss|rejuven|guardian|walker|patron|shrine|urn|objective|base", case=False, regex=True)].dropna(subset=["minutes"]).copy()
    if not objective.empty:
        objective["label"] = objective["event_type"].astype(str)
        if "target" in objective:
            targets = objective["target"].astype("string").fillna("")
            objective["label"] += targets.map(lambda value: f" / {value}" if value else "")
        labels = list(objective.sort_values("minutes")["label"].unique())
        positions = {label: index for index, label in enumerate(labels)}
        objective["position"] = objective["label"].map(positions)
        if "team" in objective:
            for index, (team, rows) in enumerate(objective.groupby("team", dropna=False)):
                label = "Team unknown" if pd.isna(team) else f"Team {team:g}" if isinstance(team, (float, int)) else f"Team {team}"
                ax.scatter(rows["minutes"], rows["position"], s=45, marker="D", color=TEAM_COLORS[index % len(TEAM_COLORS)], label=label)
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        else:
            ax.scatter(objective["minutes"], objective["position"], s=45, marker="D", color=TEAM_COLORS[0])
        ax.set_yticks(range(len(labels)), [textwrap.fill(label, 36) for label in labels], fontsize=8)
        _style(ax)
    else:
        _empty(ax, "No recognized major objective events were recovered. This does not establish that no objectives occurred.")
    _save(fig, output, "05_major_objective_timeline.png", "Only recorded objective events appear. Labels preserve event type and target. Filtering recognizes Midboss, Rejuvenator, Guardian, Walker, Patron, shrine, urn, base, and objective names; all other events remain in events.parquet.")
    print(f"Analysis: wrote five PNGs for {title_suffix.lower()} to {output.resolve()}")
    return output


if __name__ == "__main__":
    make_plots()
