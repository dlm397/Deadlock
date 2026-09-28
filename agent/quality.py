# Inventories distinguish actual payload observations from advertised schema.
from collections import Counter, defaultdict
from math import isfinite
from statistics import median
from .io import save_json


def fields_inventory(value):
    fields = defaultdict(lambda: {"observations": 0, "nulls": 0, "types": set()})

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for child in value:
                visit(child, path + "[]")
        else:
            fields[path]["observations"] += 1
            fields[path]["nulls"] += value is None
            fields[path]["types"].add(type(value).__name__)
    visit(value, "")
    return {k: {**v, "types": sorted(v["types"])} for k, v in sorted(fields.items())}


def audit_metadata(payload, folder):
    inventory = fields_inventory(payload)
    save_json(folder / "metadata_fields.json", inventory)
    for category, terms in {"player": ["players[]"], "team": ["teams[]", "team_score"],
                            "objective": ["objectives[]", "mid_boss[]"],
                            "item_inventory": ["items[]", "ability_stats[]"]}.items():
        save_json(folder / f"{category}_properties.json",
                  {k: v for k, v in inventory.items() if any(term in k for term in terms)})


def audit_demo_schema(schema, folder):
    save_json(folder / "api_demo_schema.json", schema)
    for kind in ("entity", "event"):
        save_json(folder / f"declared_{kind}_tables.json", {
            "status": "declared_by_API_not_evidence_of_occurrence",
            "tables": [x for x in schema.get("tables", []) if x.get("kind") == kind]})

# Small, nonfatal checks of the research tables; missing is never treated as zero.

def _number(value):
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _first(row, *names):
    return next((row[name] for name in names if row.get(name) is not None), None)


def _id(value):
    if value is None:
        return None
    # Integer identifiers can arrive from a nullable numeric column as floats.
    if isinstance(value, float) and isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def validate_tables(tables: dict[str, list[dict]]) -> list[str]:
    # Return warnings, never reject a match or mutate its observations.
    warnings = []
    matches = tables.get("matches", [])
    players = tables.get("players", [])
    states = tables.get("player_state", [])
    for name in ("matches", "players", "player_state", "team_state", "events", "purchases"):
        if not tables.get(name):
            warnings.append(f"{name}: no rows; this coverage is unavailable in this run.")

    by_match = {_id(row.get("match_id")): row for row in matches}
    if len(by_match) != len(matches):
        warnings.append("matches: duplicate match_id values.")
    if None in by_match:
        warnings.append("matches: a match_id is missing.")
    player_keys = [(_id(row.get("match_id")), _id(row.get("player_id"))) for row in players]
    duplicates = sum(count - 1 for count in Counter(player_keys).values())
    if duplicates:
        warnings.append(f"players: {duplicates} duplicate (match_id, player_id) rows.")
    if any(player_id is None for _, player_id in player_keys):
        warnings.append("players: missing player_id values.")
    known_players = set(player_keys)
    players_by_match = defaultdict(list)
    for row in players:
        players_by_match[_id(row.get("match_id"))].append(row)

    for match_id, match in by_match.items():
        rows = players_by_match[match_id]
        expected = _number(match.get("expected_players")) or 12
        if len(rows) != expected:
            warnings.append(f"Match {match_id}: {len(rows)} players; expected {int(expected)} for a standard match.")
        teams = {_id(row.get("team")) for row in rows if row.get("team") is not None}
        winner = _id(_first(match, "winning_team", "winner"))
        if winner is None or (teams and winner not in teams):
            warnings.append(f"Match {match_id}: winning team {winner!r} is missing or absent from player teams {sorted(teams)}.")
        duration = _number(_first(match, "duration_s", "duration", "duration_seconds"))
        if duration is None or duration <= 0:
            warnings.append(f"Match {match_id}: missing or nonpositive match duration.")

    time_tables = ("events", "purchases", "player_state", "team_state")
    for name in ("players", *time_tables):
        counts = Counter()
        for row in tables.get(name, []):
            match_id = _id(row.get("match_id"))
            if match_id not in by_match:
                counts["rows reference an unknown match_id"] += 1
            if name != "players":
                for field in ("player_id", "other_player_id"):
                    player_id = _id(row.get(field))
                    if player_id is not None and (match_id, player_id) not in known_players:
                        counts[f"rows reference an unknown {field}"] += 1
                if name in ("player_state", "purchases") and row.get("player_id") is None:
                    counts["rows have no player_id"] += 1
            for field in ("net_worth", "final_net_worth", "souls", "unspent_souls", "team_net_worth", "total_net_worth"):
                value = _number(row.get(field))
                if value is not None and value < 0:
                    counts[f"rows have negative {field}"] += 1
            if name in time_tables:
                game_time = _number(row.get("game_time"))
                if game_time is None:
                    counts["rows have no numeric game_time"] += 1
                else:
                    match = by_match.get(match_id, {})
                    duration = _number(_first(match, "duration_s", "duration", "duration_seconds"))
                    if game_time < 0:
                        counts["rows precede game time zero (inspect pregame semantics)"] += 1
                    if duration is not None and game_time > duration + 5:
                        counts["rows occur more than 5 seconds after match end"] += 1
        warnings.extend(f"{name}: {count} {message}." for message, count in counts.items())

    known_items = {_id(_first(row, "item_id", "id")) for row in tables.get("items", [])}
    known_items.discard(None)
    purchased_items = {_id(row.get("item_id")) for row in tables.get("purchases", [])}
    purchased_items.discard(None)
    if purchased_items and not known_items:
        warnings.append("purchases: no item definitions are available to validate item IDs.")
    elif purchased_items - known_items:
        unknown = sorted(purchased_items - known_items)
        warnings.append(f"purchases: {len(unknown)} item IDs absent from downloaded definitions; examples: {', '.join(unknown[:8])}.")

    state_groups = defaultdict(list)
    for row in states:
        state_groups[(_id(row.get("match_id")), _id(row.get("player_id")))].append(row)
    unordered = duplicates = source_in_future = incorrect_age = 0
    for rows in state_groups.values():
        times = [_number(row.get("game_time")) for row in rows]
        times = [value for value in times if value is not None]
        unordered += times != sorted(times)
        duplicates += len(times) - len(set(times))
        for row in rows:
            game_time = _number(row.get("game_time"))
            source_time = _number(row.get("source_game_time"))
            age = _number(_first(row, "state_age", "state_age_s"))
            if game_time is not None and source_time is not None:
                source_in_future += source_time > game_time + 0.01
                incorrect_age += age is not None and abs(age - (game_time - source_time)) > 0.1
    if unordered:
        warnings.append(f"player_state: {unordered} player series are not ordered by game_time.")
    if duplicates:
        warnings.append(f"player_state: {duplicates} duplicate (match_id, player_id, game_time) rows.")
    if source_in_future:
        warnings.append(f"player_state: {source_in_future} rows use a source observation from the future.")
    if incorrect_age:
        warnings.append(f"player_state: {incorrect_age} rows have inconsistent state_age and source_game_time.")

    missing_series = known_players - set(state_groups)
    if missing_series:
        warnings.append(f"player_state: {len(missing_series)} players have no state observations.")
    if states:
        for field in ("net_worth", "health", "x", "y", "z", "unspent_souls"):
            missing = sum(_number(row.get(field)) is None for row in states)
            if missing:
                warnings.append(f"player_state: {field} missing in {missing}/{len(states)} rows ({missing / len(states):.1%}); missing values are not zero.")
        ages = [_number(_first(row, "state_age", "state_age_s")) for row in states]
        ages = [age for age in ages if age is not None]
        if ages and max(ages) > 0:
            warnings.append(f"player_state: carried-forward observations have median age {median(ages):.0f}s and maximum age {max(ages):.0f}s; sampling frequency is not source frequency.")

    disagreed = stale = missing_final = 0
    for player in players:
        key = (_id(player.get("match_id")), _id(player.get("player_id")))
        rows = [row for row in state_groups[key] if _number(row.get("game_time")) is not None and _number(row.get("net_worth")) is not None]
        replay_rows = [row for row in rows if row.get("state_source") == "replay"]
        if replay_rows:
            rows = replay_rows  # Independent check, not metadata compared with itself.
        final = _number(_first(player, "final_net_worth", "net_worth"))
        if not rows or final is None:
            missing_final += 1
            continue
        latest = max(rows, key=lambda row: float(row["game_time"]))
        times = sorted({_number(row.get("game_time")) for row in rows})
        deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
        tolerance_s = max(30, 2 * median(deltas)) if deltas else 30
        duration = _number(_first(by_match.get(key[0], {}), "duration_s", "duration", "duration_seconds"))
        observation_time = _number(_first(latest, "source_game_time", "game_time"))
        if duration is not None and observation_time is not None and duration - observation_time > tolerance_s:
            stale += 1
        elif abs(float(latest["net_worth"]) - final) > max(500, abs(final) * 0.2):
            disagreed += 1
    if disagreed:
        warnings.append(f"player_state: {disagreed} final net-worth values differ from metadata by more than max(500 souls, 20%).")
    if stale:
        warnings.append(f"player_state: final agreement cannot be checked for {stale} players because the last source observation is stale.")
    if missing_final:
        warnings.append(f"player_state: final agreement cannot be checked for {missing_final} players with missing net worth or state.")
    for match in matches:
        if match.get("demo_kill_count_difference") not in (None, 0):
            warnings.append(f"match {match.get('match_id')}: local/API replay kill counts differ by {match['demo_kill_count_difference']}; check match boundaries.")
    return warnings
