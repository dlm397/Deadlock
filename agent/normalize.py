# Normalize source observations without inventing measurements between snapshots.

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from math import floor, isfinite
from typing import Any


COMBAT_TYPES = {0: "out", 1: "player", 2: "enemy_npc", 3: "neutral"}
MOVE_TYPES = {
    0: "normal", 1: "ability", 2: "ability_debuff", 3: "ground_dash",
    4: "slide", 5: "rope_climbing", 6: "ziplining", 7: "in_air", 8: "air_dash",
}
OBJECTIVE_TYPES = {
    0: "core", 1: "guardian", 2: "guardian", 3: "guardian", 4: "guardian",
    5: "walker", 6: "walker", 7: "walker", 8: "walker", 9: "patron",
    10: "shrine", 11: "shrine", 12: "base_guardian", 13: "base_guardian",
    14: "base_guardian", 15: "base_guardian",
}
STAT_TYPES = {0: "damage", 1: "healing", 2: "heal_prevented", 3: "mitigated", 4: "lethal_damage", 5: "regen"}
PANEL_STATS = (
    "net_worth", "kills", "deaths", "assists", "max_health", "weapon_power",
    "tech_power", "spirit_power", "player_damage", "boss_damage", "player_healing", "level",
    "ability_points", "shots_hit", "shots_missed",
)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _at(values: Any, index: int | None) -> Any:
    return values[index] if isinstance(values, list) and index is not None and 0 <= index < len(values) else None


def _power_fields(row: dict, snapshot: dict, source_time: float | None, time: float | None) -> None:
    # Exact-version assets label TechPower as Spirit Power; retain the native name.
    for field, native in (("spirit_power", "tech_power"), ("weapon_power", "weapon_power")):
        value = snapshot.get(native)
        observed = source_time if value is not None else None
        row[field] = value
        row[f"{field}_source_game_time"] = observed
        row[f"{field}_state_age"] = time - observed if time is not None and observed is not None else None
        row[f"{field}_quality"] = ("observed_sparse_snapshot" if time == observed else "metadata_asof") if observed is not None else "unavailable"
        row.setdefault("field_provenance", {})[field] = f"metadata_stats:{native}" if observed is not None else "unavailable"


def _complete_sum(rows: list[dict], field: str) -> tuple[float | None, int]:
    known = [row[field] for row in rows if _number(row.get(field))]
    return (sum(known) if rows and len(known) == len(rows) else None, len(known))


def _identity(player: dict) -> dict:
    # account_id remains null for anonymous/missing identities; the slot is always
    # retained as the within-match joining key, never substituted into account_id.
    return {"player_id": player.get("account_id"), "account_id": player.get("account_id"),
            "player_slot": player.get("player_slot"), "hero_id": player.get("hero_id"),
            "team": player.get("team")}


def aggregate_team_state(players: list[dict], state_rows: list[dict], provenance: dict,
                         objectives: list[dict] | None = None) -> list[dict]:
    # Require a complete roster and observations before reporting a team total.
    expected = defaultdict(int)
    for player in players:
        expected[player.get("team")] += 1
    grouped = defaultdict(list)
    for row in state_rows:
        grouped[(row["game_time"], row.get("team"))].append(row)
    team_panel = {}
    for (time, team), rows in grouped.items():
        roster_complete = len(rows) == expected[team] and expected[team] > 0
        tr = {**provenance, "game_time": time, "team": team, "player_count": len(rows),
              "expected_players": expected[team], "roster_complete": roster_complete,
              "source": "aggregate_player_state", "aggregation_quality": "sum_of_available_player_state",
              "team_total_souls": None, "team_net_worth": None, "net_worth_advantage": None,
              "alive_players": None, "team_kills": None, "team_deaths": None}
        for output, field in (("team_net_worth", "net_worth"), ("team_total_souls", "souls"),
                              ("team_unspent_souls", "unspent_souls"), ("team_kills", "kills"),
                              ("team_deaths", "deaths"), ("team_assists", "assists")):
            value, count = _complete_sum(rows, field)
            tr[output] = value if roster_complete else None
            tr[f"{field}_observed_players"] = count
        # Friendly aliases for plotting; they have exactly the same completeness.
        tr.update(net_worth=tr["team_net_worth"], souls=tr["team_total_souls"],
                  kills=tr["team_kills"], deaths=tr["team_deaths"])
        alive = [r["alive"] for r in rows if r.get("alive") is not None]
        tr["alive_observed_players"] = len(alive)
        tr["alive_players"] = sum(alive) if roster_complete and len(alive) == len(rows) else None
        source_times = [r["source_game_time"] for r in rows if r.get("source_game_time") is not None]
        tr["source_game_time_min"] = min(source_times) if source_times else None
        tr["source_game_time_max"] = max(source_times) if source_times else None
        tr["state_age_max"] = time - min(source_times) if source_times else None
        tr["economy_complete"] = roster_complete and len(source_times) == len(rows) and tr["net_worth_observed_players"] == len(rows)
        tr["economy_simultaneous_observations"] = tr["economy_complete"] and len(set(source_times)) == 1
        tr["player_state_sources"] = sorted({r.get("source", "unknown") for r in rows})
        tr["objective_state"] = [{"objective_id": r.get("objective_id", r.get("team_objective_id")),
                                  "objective_type": r.get("objective_type"),
                                  "destroyed": r.get("destroyed_time_s") > 0 and r["destroyed_time_s"] <= time if _number(r.get("destroyed_time_s")) else None}
                                 for r in objectives or [] if r.get("team") == team]
        tr["objective_state_quality"] = "reconstructed_for_objectives_reported_in_metadata" if objectives is not None else "unavailable"
        team_panel[(time, team)] = tr
    for (time, team), row in team_panel.items():
        opponent = team_panel.get((time, 1 - team), {}) if team in (0, 1) else {}
        own, other = row["team_net_worth"], opponent.get("team_net_worth")
        row["net_worth_advantage"] = own - other if own is not None and other is not None else None
        row["team_net_worth_advantage"] = row["net_worth_advantage"]
    return list(team_panel.values())


def _path_rows(path: dict, paths: dict, base: dict, duration: float | None) -> list[dict]:
    interval = paths.get("interval_s")
    n = max((len(path.get(k) or []) for k in ("x_pos", "y_pos", "health", "combat_type", "move_type")), default=0)
    result = []
    for index in range(n):
        time = index * interval if _number(interval) and interval > 0 else None
        # Keep out-of-duration source observations: the validation layer can flag
        # them, and the analytical panel below only samples within the duration.
        row = {**base, "path_index": index, "game_time": time,
               "path_interval_s": interval, "path_version": paths.get("version"),
               "x_quantized": _at(path.get("x_pos"), index),
               "y_quantized": _at(path.get("y_pos"), index),
               "health_percent": _at(path.get("health"), index),
               "combat_type_code": _at(path.get("combat_type"), index),
               "move_type_code": _at(path.get("move_type"), index),
               "z": None, "health": None,
               "observation_quality": "observed_quantized_path"}
        for axis in ("x", "y"):
            raw, low, high, resolution = (row[f"{axis}_quantized"], path.get(f"{axis}_min"),
                                           path.get(f"{axis}_max"), paths.get(f"{axis}_resolution"))
            row[axis] = low + raw * (high - low) / resolution if all(_number(v) for v in (raw, low, high, resolution)) and resolution > 0 else None
            row[f"{axis}_resolution"] = resolution
            row[f"{axis}_min"], row[f"{axis}_max"] = low, high
        sentinel = row["x_quantized"] == 0 and row["y_quantized"] == 0 and row["health_percent"] == 0
        if sentinel:
            row["x"], row["y"] = None, None
        row["position_quality"] = "unavailable_dead_sentinel" if sentinel else "reconstructed_from_quantized_path"
        row["alive"] = row["health_percent"] > 0 if _number(row["health_percent"]) else None
        row["alive_quality"] = "reconstructed_from_health_percent" if row["alive"] is not None else "unavailable"
        row["combat_type"] = COMBAT_TYPES.get(row["combat_type_code"])
        row["move_type"] = MOVE_TYPES.get(row["move_type_code"])
        result.append(row)
    return result


def normalize_metadata(payload: dict, provenance: dict, item_definitions: list[dict], sample_interval: float) -> dict[str, list[dict]]:
    # Definitions must match the game version; [] leaves item types unclassified.
    if not _number(sample_interval) or sample_interval <= 0:
        raise ValueError("sample_interval must be a positive finite number")
    info = payload.get("match_info")
    if not isinstance(info, dict):
        raise ValueError("Metadata response has no match_info object")
    players = info.get("players") or []
    duration = info.get("duration_s")
    if not _number(duration) or duration < 0:
        duration = None
    base = {**provenance, "match_id": info.get("match_id", provenance.get("match_id"))}
    definitions = {item.get("id"): item for item in item_definitions if item.get("id") is not None}
    tables = {name: [] for name in (
        "matches", "players", "purchases", "events", "player_state", "team_state",
        "metadata_player_stats", "metadata_player_paths", "metadata_item_records",
        "metadata_objectives", "metadata_midboss", "metadata_damage", "metadata_custom_stats",
        "metadata_gold_sources", "metadata_teams",
    )}
    slots = {p.get("player_slot"): p for p in players if p.get("player_slot") is not None}
    paths = info.get("match_paths") or {}
    paths_by_slot = {p.get("player_slot"): p for p in paths.get("paths", [])}
    custom_names = {r.get("id"): r.get("name") for r in info.get("custom_user_stats", [])}
    panel_times = [] if duration is None else [i * sample_interval for i in range(floor(duration / sample_interval) + 1)]
    if duration is not None and (not panel_times or panel_times[-1] != duration):
        panel_times.append(duration)

    def event(kind: str, time: Any, raw: dict, **fields: Any) -> None:
        tables["events"].append({**base, "event_id": f"metadata:{base['match_id']}:{len(tables['events'])}",
                                 "event_type": kind, "game_time": time, "player_id": None,
                                 "other_player_id": None, "team": None, "target": None,
                                 "value": None, "x": None, "y": None, "z": None,
                                 "observation_quality": "observed_metadata_event",
                                 "raw_event_data": raw, **fields})

    for player in players:
        pb = {**base, **_identity(player)}
        stats = player.get("stats") or []
        timed_stats = sorted((s for s in stats if _number(s.get("time_stamp_s"))), key=lambda s: s["time_stamp_s"])
        stamps = [s["time_stamp_s"] for s in timed_stats]
        for index, stat in enumerate(stats):
            sb = {**pb, "game_time": stat.get("time_stamp_s"), "source_stat_index": index}
            stat_row = {**sb, **stat, "observation_quality": "observed_sparse_snapshot"}
            _power_fields(stat_row, stat, stat.get("time_stamp_s"), stat.get("time_stamp_s"))
            tables["metadata_player_stats"].append(stat_row)
            for value in stat.get("custom_user_stats", []):
                tables["metadata_custom_stats"].append({**sb, **value, "stat_name": custom_names.get(value.get("id"))})
            for value in stat.get("gold_sources", []):
                tables["metadata_gold_sources"].append({**sb, **{k: v for k, v in value.items() if k != "source"}, "gold_source_code": value.get("source")})

        last = timed_stats[-1] if timed_stats else {}
        at_end = bool(last) and last.get("time_stamp_s") == duration
        # Only snapshots stamped at match end supplement the final scoreboard.
        # The latest snapshot always survives separately, even if not final.
        pr = {**player, **pb, "win": player.get("team") == info.get("winning_team") if info.get("winning_team") in (0, 1) and player.get("team") in (0, 1) else None,
              "final_stats_time": last.get("time_stamp_s"), "final_stats_at_match_end": at_end,
              "latest_metadata_stats": last or None}
        if at_end:
            for key, value in last.items():
                if key not in pr:
                    pr[key] = value
                else:
                    pr[f"final_stats_{key}"] = value
        for key in PANEL_STATS:
            pr.setdefault(key, None)
        _power_fields(pr, last if at_end else {}, duration if at_end else None, duration)
        pr["final_net_worth"] = player.get("net_worth")
        pr["rank_badge"] = (player.get("player_rank_data") or {}).get("initial_display_rank")
        pr["objective_damage"] = last.get("boss_damage") if at_end else None
        pr["healing"] = last.get("player_healing") if at_end else None
        hits, misses = pr.get("shots_hit"), pr.get("shots_missed")
        pr["shot_accuracy"] = hits / (hits + misses) if _number(hits) and _number(misses) and hits + misses > 0 else None

        item_records = []
        for index, item in enumerate(player.get("items") or []):
            definition = definitions.get(item.get("item_id"), {})
            kind = definition.get("type")
            rec = {**pb, **item, "game_time": item.get("game_time_s"), "item_record_index": index,
                   "item_name": definition.get("name"), "definition_type": kind,
                   "definition_match": "exact_game_version" if definition else "unavailable",
                   "item_cost": definition.get("cost") if kind == "upgrade" else None,
                   "raw_item_data": item}
            item_records.append(rec)
            tables["metadata_item_records"].append(rec)
            transaction_type = "purchase" if kind == "upgrade" else "unclassified_item_record"
            if kind == "ability":
                transaction_type = "ability_unlock" if item.get("upgrade_id") == 0 else "ability_upgrade" if item.get("upgrade_id") is not None else "ability_change"
            event(transaction_type, item.get("game_time_s"), item,
                  **_identity(player), target=item.get("item_id"), item_name=definition.get("name"),
                  item_id=item.get("item_id"), definition_type=kind)
            if kind != "ability":
                tables["purchases"].append({**rec, "transaction_type": transaction_type,
                                             "souls_before": None, "souls_after": None, "net_worth": None,
                                             "amount_paid": None, "transaction_quality": "observed_acquisition" if kind == "upgrade" else "unclassified"})
            sold = item.get("sold_time_s")
            if _number(sold) and sold > 0:
                removal = {**rec, "game_time": sold, "transaction_type": "item_removed",
                           "removal_reason": "sale_or_upgrade_unknown", "souls_before": None,
                           "souls_after": None, "net_worth": None, "amount_paid": None,
                           "transaction_quality": "observed_removal_reason_unknown"}
                tables["purchases"].append(removal)
                event("item_removed", sold, item, **_identity(player), target=item.get("item_id"),
                      item_id=item.get("item_id"), removal_reason="sale_or_upgrade_unknown")

        inventory_complete = all(r["definition_type"] in ("ability", "upgrade") and r.get("game_time_s") is not None and (r["definition_type"] != "upgrade" or r.get("sold_time_s") is not None) for r in item_records) and bool(definitions) and "items" in player

        def inventory_at(time: float) -> list[int]:
            return sorted({r["item_id"] for r in item_records if r["definition_type"] == "upgrade"
                           and _number(r.get("game_time_s")) and r["game_time_s"] <= time
                           and _number(r.get("sold_time_s")) and (r["sold_time_s"] == 0 or r["sold_time_s"] > time)})

        def abilities_at(time: float) -> dict:
            current = {}
            for rec in sorted(item_records, key=lambda r: r["game_time_s"] if _number(r.get("game_time_s")) else float("inf")):
                if rec["definition_type"] == "ability" and _number(rec.get("game_time_s")) and rec["game_time_s"] <= time:
                    current[str(rec["item_id"])] = {k: rec.get(k) for k in ("upgrade_id", "upgrade_info", "game_time_s")}
            return current

        pr["final_inventory"] = inventory_at(duration) if duration is not None else None
        pr["final_inventory_complete"] = inventory_complete
        pr["final_inventory_quality"] = "reconstructed_from_item_records" if inventory_complete else "partial_or_unavailable"
        pr["final_ability_states"] = abilities_at(duration) if duration is not None else None
        tables["players"].append(pr)

        for index, death in enumerate(player.get("death_details") or []):
            killer = slots.get(death.get("killer_player_slot"), {})
            pos, killer_pos = death.get("death_pos") or {}, death.get("killer_pos") or {}
            fields = {**_identity(player), "other_player_id": killer.get("account_id"),
                      "killer_player_slot": death.get("killer_player_slot"),
                      "death_record_index": index, "death_duration_s": death.get("death_duration_s"),
                      "time_to_kill_s": death.get("time_to_kill_s"),
                      **{axis: pos.get(axis) for axis in ("x", "y", "z")},
                      **{f"killer_{axis}": killer_pos.get(axis) for axis in ("x", "y", "z")}}
            # A death row encodes both victim and killer; do not double count it
            # as an additional kill event or infer unrecorded assists.
            event("death", death.get("game_time_s"), death, **fields)
        for ping in player.get("pings") or []:
            event("player_ping", ping.get("game_time_s"), ping, **_identity(player), target=ping.get("ping_type"), value=ping.get("ping_data"))
        abandon = player.get("abandon_match_time_s")
        if _number(abandon) and abandon > 0:
            event("match_abandon", abandon, {"abandon_match_time_s": abandon}, **_identity(player))

        path = paths_by_slot.get(player.get("player_slot"))
        path_rows = _path_rows(path, paths, pb, duration) if path else []
        tables["metadata_player_paths"].extend(path_rows)
        path_times = [r["game_time"] for r in path_rows if _number(r["game_time"])]
        for time in panel_times:
            idx = bisect_right(stamps, time) - 1
            past = timed_stats[idx] if idx >= 0 else {}
            source_time = past.get("time_stamp_s")
            row = {**pb, **{k: None for k in PANEL_STATS}, **past,
                   "game_time": time, "source_game_time": source_time,
                   "state_age": time - source_time if source_time is not None else None,
                   "state_quality": "observed_sparse_snapshot" if source_time == time else "reconstructed_last_observation_carried_forward" if source_time is not None else "unavailable_before_first_snapshot",
                   "souls": None, "unspent_souls": None, "health": None, "z": None,
                   "velocity_x": None, "velocity_y": None, "velocity_z": None,
                   "inventory": inventory_at(time), "inventory_complete": inventory_complete,
                   "inventory_quality": "reconstructed_from_item_records" if inventory_complete else "partial_or_unavailable",
                   "ability_states": abilities_at(time), "ability_levels": None}
            _power_fields(row, past, source_time, time)
            pidx = bisect_right(path_times, time) - 1
            prow = path_rows[pidx] if pidx >= 0 else {}
            ptime = prow.get("game_time")
            # Never propagate movement beyond the coverage of a source path.
            if ptime is not None and time - ptime >= paths.get("interval_s", 0):
                prow, ptime = {}, None
            row.update({key: prow.get(key) for key in ("x", "y", "health_percent", "alive", "alive_quality", "combat_type_code", "combat_type", "move_type_code", "move_type", "position_quality")})
            row["path_source_game_time"] = ptime
            row["path_state_age"] = time - ptime if ptime is not None else None
            row["path_quality"] = "observed_path_sample" if ptime == time else "reconstructed_last_path_sample" if ptime is not None else "unavailable"
            tables["player_state"].append(row)

    for objective in info.get("objectives") or []:
        oid, team = objective.get("team_objective_id"), objective.get("team")
        legacy = objective.get("legacy_objective_id")
        if oid is None and isinstance(legacy, int) and 0 <= legacy < 32:
            oid, team = legacy % 16, legacy // 16
        row = {**base, **objective, "objective_id": oid, "team": team,
               "objective_type": OBJECTIVE_TYPES.get(oid, "unknown"),
               "team_semantics": "owner_of_objective"}
        tables["metadata_objectives"].append(row)
        time = objective.get("destroyed_time_s")
        if _number(time) and time > 0:
            event("objective_destroyed", time, objective, team=team, target=oid,
                  objective_type=row["objective_type"], attacking_team=1 - team if team in (0, 1) else None,
                  team_semantics="owner_of_objective")
        first_damage = objective.get("first_damage_time_s")
        if _number(first_damage) and first_damage > 0:
            event("objective_first_damage", first_damage, objective, team=team, target=oid,
                  objective_type=row["objective_type"], team_semantics="owner_of_objective")

    for index, boss in enumerate(info.get("mid_boss") or []):
        row = {**base, **boss, "midboss_index": index, "claim_time": None,
               "claim_time_quality": "unavailable_in_metadata"}
        tables["metadata_midboss"].append(row)
        event("midboss_killed", boss.get("destroyed_time_s"), boss, team=boss.get("team_killed"), midboss_index=index)
        if boss.get("team_claimed") in (0, 1):
            # Metadata knows the claimant but exposes NO claim timestamp.
            event("rejuvenator_claim_team", None, boss, team=boss.get("team_claimed"),
                  midboss_index=index, associated_midboss_destroyed_time=boss.get("destroyed_time_s"),
                  time_quality="unavailable_in_metadata")

    for pause in info.get("match_pauses") or []:
        pp = slots.get(pause.get("player_slot"), {})
        event("match_pause", pause.get("game_time_s"), pause, **_identity(pp), value=pause.get("pause_duration_s"))
    for replay in info.get("watched_death_replays") or []:
        event("watched_death_replay", replay.get("game_time_s"), replay, **_identity(slots.get(replay.get("player_slot"), {})))
    for team in info.get("teams") or []:
        tables["metadata_teams"].append({**base, **team})

    matrix = info.get("damage_matrix") or {}
    details = matrix.get("source_details") or {}
    for dealer in matrix.get("damage_dealers") or []:
        dealer_slot = dealer.get("dealer_player_slot")
        for source in dealer.get("damage_sources") or []:
            index = source.get("source_details_index")
            stat_type = _at(details.get("stat_type"), index)
            for target in source.get("damage_to_players") or []:
                vals = target.get("damage") or []
                tables["metadata_damage"].append({**base,
                    "dealer_player_slot": dealer_slot, "player_id": slots.get(dealer_slot, {}).get("account_id"),
                    "target_player_slot": target.get("target_player_slot"),
                    "other_player_id": slots.get(target.get("target_player_slot"), {}).get("account_id"),
                    "source_details_index": index, "source_name": _at(details.get("source_name"), index),
                    "stat_type_code": stat_type, "stat_type": STAT_TYPES.get(stat_type),
                    "final_value": vals[-1] if vals else None, "damage_values": vals,
                    "sample_time_s": matrix.get("sample_time_s"),
                    "time_alignment": "raw_arrays_preserved_no_assumed_padding",
                    "raw_damage_data": target})

    tables["team_state"] = aggregate_team_state(players, tables["player_state"], base, tables["metadata_objectives"])

    mr = {**info, **base, "duration": duration,
          "metadata_envelope_extra": {k: v for k, v in payload.items() if k != "match_info"},
          "first_midboss_time": None, "first_midboss_team": None,
          "first_rejuvenator_claim_time": None, "first_rejuvenator_claim_team": None,
          "rejuvenator_claim_time_quality": "unavailable_in_metadata"}
    bosses = sorted((b for b in info.get("mid_boss", []) if _number(b.get("destroyed_time_s"))), key=lambda b: b["destroyed_time_s"])
    if bosses:
        mr.update(first_midboss_time=bosses[0].get("destroyed_time_s"), first_midboss_team=bosses[0].get("team_killed"),
                  first_rejuvenator_claim_team=bosses[0].get("team_claimed"))
    for team in (0, 1):
        team_players = [p for p in players if p.get("team") == team]
        mr[f"team{team}_final_net_worth"], mr[f"team{team}_net_worth_observed_players"] = _complete_sum(team_players, "net_worth")
        mr[f"team{team}_player_count"] = len(team_players)
    tables["matches"].append(mr)
    return tables
