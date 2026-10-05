"""Small, explicit research schemas and metadata/replay normalization."""

from collections import defaultdict
from datetime import datetime, timezone
from math import isfinite

import pyarrow as pa


# Variable selection lives here. Identifiers never pass through float columns.
U32, U64, F64, BOOL, TEXT = pa.uint32(), pa.uint64(), pa.float64(), pa.bool_(), pa.string()
KEY_FIELDS = [("match_id", U64), ("player_slot", U32), ("account_id", U32),
              ("hero_id", U32), ("team_id", U32)]
MATCH_SCHEMA = pa.schema([
    ("match_id", U64),
    ("start_time_utc", pa.timestamp("ms", tz="UTC")), ("duration_s", U32),
    ("player_count", U32), ("game_mode", TEXT), ("match_mode", TEXT),
    ("match_outcome", TEXT), ("winning_team_id", U32),
    ("client_version", U32), ("replay_build_num", U32), ("game_mode_version", U32),
    ("map_name", TEXT), ("replay_patch_version", U32),
])
END_MATCH_SCHEMA = pa.schema(KEY_FIELDS + [
    ("won", BOOL),
    ("initial_rank_badge", U32), ("assigned_lane", U32),
    *[(name, U64) for name in ("kills", "deaths", "assists", "net_worth", "level",
                              "last_hits", "denies", "hero_damage", "objective_damage",
                              "hero_healing", "self_healing", "hero_damage_taken")],
    ("combat_stats_time_s", F64), ("combat_stats_at_end", BOOL),
])
TIME_SLICE_SCHEMA = pa.schema(KEY_FIELDS + [
    ("game_time_s", F64), ("observed_time_s", F64), ("is_final_observation", BOOL),
    *[(name, U64) for name in ("net_worth", "unspent_souls", "kills", "deaths", "assists",
                              "last_hits", "denies", "level")],
    ("alive", BOOL), ("health", pa.int64()), ("max_health", pa.int64()),
    ("x", F64), ("y", F64), ("z", F64),
    *[(name, U64) for name in ("hero_damage", "objective_damage", "hero_healing",
                              "self_healing", "hero_damage_taken")],
])
EVENT_SCHEMA = pa.schema([
    ("match_id", U64), ("event_id", TEXT), ("event_time_s", F64), ("event_type", TEXT),
    ("participant_role", TEXT), ("player_slot", U32), ("actor_player_slot", U32),
    ("target_player_slot", U32), ("team_id", U32), ("item_id", U32),
    ("ability_id", U32), ("objective_id", U32), ("value", F64), ("duration_s", F64),
    ("x", F64), ("y", F64), ("z", F64), ("objective_team_id", U32),
    ("claiming_team_id", U32), ("upgrade_id", U32), ("upgrade_info", U32),
    ("imbued_ability_id", U32), ("time_to_kill_s", F64),
    ("related_event_id", TEXT), ("event_source", TEXT),
])

GAME_MODES = {0: "Invalid", 1: "Normal", 2: "1v1Test", 3: "Sandbox",
              4: "StreetBrawl", 5: "ExploreNYC", 6: "Internal"}
MATCH_MODES = {0: "Invalid", 1: "Unranked", 2: "PrivateLobby", 3: "CoopBot",
               4: "Ranked", 5: "ServerTest", 6: "Tutorial", 7: "HeroLabs",
               8: "NewPlayerPlacement"}
OUTCOMES = {0: "TeamWin", 1: "Error", 2: "MatchDraw"}
COMBAT_FIELDS = {"hero_damage": "player_damage", "objective_damage": "boss_damage",
                 "hero_healing": "player_healing", "self_healing": "self_healing",
                 "hero_damage_taken": "player_damage_taken"}


def _int(value, name="integer"):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    if isinstance(value, str) and not value.isdecimal():
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    return int(value)


def _number(value, name="number"):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _enum(value, mapping):
    if type(value) is int:
        return value if value in mapping else None
    if not isinstance(value, str):
        return None
    # These are the protocol names and the API/lake's readable spellings.
    token = value.rsplit("_", 1)[-1] if value.startswith("k_") else value
    token = token.replace("_", "").lower()
    return next((key for key, name in mapping.items()
                 if name.replace("_", "").lower() == token), None)


def _team(value):
    return _enum(value, {0: "Team0", 1: "Team1"})


def _info(metadata):
    info = metadata.get("match_info")
    if not isinstance(info, dict):
        raise ValueError("metadata has no match_info")
    return info


def _roster(info):
    players = info.get("players")
    if not isinstance(players, list):
        raise ValueError("metadata has no complete player roster")
    if not all(isinstance(player, dict) for player in players):
        raise ValueError("metadata contains a malformed player record")
    slots = [_int(p.get("player_slot"), "player_slot") for p in players]
    if any(slot is None or not 0 <= slot < 2**32 for slot in slots) or len(slots) != len(set(slots)):
        raise ValueError("missing or conflicting player slots in metadata roster")
    return dict(zip(slots, players))


def _identity(info, player):
    match_id = _int(info.get("match_id"), "match_id")
    if match_id is None:
        raise ValueError("metadata match_id is missing")
    return {"match_id": match_id,
            "player_slot": _int(player.get("player_slot"), "player_slot"),
            "account_id": _int(player.get("account_id"), "account_id"),
            "hero_id": _int(player.get("hero_id"), "hero_id"),
            "team_id": _team(player.get("team"))}


def _table(rows, schema):
    """Enforce exact integer types before Arrow could silently truncate a float."""
    for row in rows:
        for field in schema:
            value = row.get(field.name)
            if pa.types.is_integer(field.type):
                value = _int(value, field.name)
                if value is not None and pa.types.is_unsigned_integer(field.type) and value < 0:
                    raise ValueError(f"{field.name} must not be negative")
                row[field.name] = value
            elif pa.types.is_floating(field.type):
                row[field.name] = _number(value, field.name)
            elif pa.types.is_boolean(field.type) and value is not None and type(value) is not bool:
                raise ValueError(f"{field.name} must be boolean or missing")
    return pa.Table.from_pylist(rows, schema=schema)


def eligibility(metadata):
    """Return eligible/ineligible/unverified and a short reason; never infer mode."""
    try:
        info = _info(metadata)
        duration = _int(info.get("duration_s"), "duration_s")
        players = info.get("players")
        if not isinstance(players, list):
            return "unverified", "complete roster unavailable"
        _roster(info)
        # Conflicting roster identities are not reliable eligibility evidence.
        accounts = [_int(p.get("account_id"), "account_id") for p in players
                    if p.get("account_id") not in (None, 0)]
        if len(accounts) != len(set(accounts)):
            return "unverified", "conflicting account IDs in roster"
        if duration is not None and duration <= 900:
            return "ineligible", f"duration {duration}s is not strictly greater than 900s"
        if len(players) != 12:
            return "ineligible", f"expected 12 players, found {len(players)}"
        game_mode = _enum(info.get("game_mode"), GAME_MODES)
        match_mode = _enum(info.get("match_mode"), MATCH_MODES)
        if game_mode is not None and game_mode not in (0, 1):
            return "ineligible", f"ruleset {GAME_MODES[game_mode]} is excluded"
        if match_mode is not None and match_mode not in (0, 1, 4):
            return "ineligible", f"matchmaking mode {MATCH_MODES[match_mode]} is excluded"
        if duration is None or game_mode in (None, 0) or match_mode in (None, 0):
            return "unverified", "duration or ruleset/matchmaking mode missing or unknown"
        if any(_team(p.get("team")) is None for p in players):
            return "unverified", "player team missing or outside the two playing teams"
        return "eligible", ""
    except (ValueError, TypeError, KeyError) as error:
        return "unverified", str(error)


def match(metadata, header=None):
    """One row of match metadata, optionally enriched by a replay header."""
    info, summary = _info(metadata), header or {}
    roster = _roster(info)
    match_id = _int(info.get("match_id"), "match_id")
    if match_id is None:
        raise ValueError("metadata match_id is missing")
    start = info.get("start_time")
    if start is not None:
        start = datetime.fromtimestamp(_int(start, "start_time"), timezone.utc)
    row = {
        "match_id": match_id, "start_time_utc": start,
        "duration_s": _int(info.get("duration_s"), "duration_s"), "player_count": len(roster),
        "game_mode": GAME_MODES.get(_enum(info.get("game_mode"), GAME_MODES)),
        "match_mode": MATCH_MODES.get(_enum(info.get("match_mode"), MATCH_MODES)),
        "match_outcome": OUTCOMES.get(_enum(info.get("match_outcome"), OUTCOMES)),
        "winning_team_id": _team(info.get("winning_team")),
        "client_version": summary.get("client_version", summary.get("replay_game_version")),
        "replay_build_num": summary.get("replay_build_num"),
        "game_mode_version": info.get("game_mode_version"),
        "map_name": summary.get("map_name"), "replay_patch_version": summary.get("replay_patch_version"),
    }
    return _table([row], MATCH_SCHEMA)


def end_match(metadata, replay_summary=None):
    """Player-only final statistics; replay_summary is accepted for compatibility."""
    info = _info(metadata)
    roster = _roster(info)
    duration = _int(info.get("duration_s"), "duration_s")
    outcome = _enum(info.get("match_outcome"), OUTCOMES)
    winner = _team(info.get("winning_team"))
    rows = []
    for slot, player in sorted(roster.items()):
        exact = [s for s in player.get("stats", []) if duration is not None and s.get("time_stamp_s") == duration]
        stat = exact[-1] if exact else {}
        team = _team(player.get("team"))
        row = {**_identity(info, player),
               "won": team == winner if outcome == 0 and team is not None and winner is not None else None,
               # Packed rank 0 is the API's unknown/placement sentinel, not rank 0.
               "initial_rank_badge": (player.get("player_rank_data") or {}).get("initial_display_rank") or None,
               "assigned_lane": player.get("assigned_lane"),
               "combat_stats_time_s": duration if exact else None,
               "combat_stats_at_end": bool(exact)}
        for key in ("kills", "deaths", "assists", "net_worth", "level", "last_hits", "denies"):
            row[key] = player.get(key)
            if row[key] is None and key in stat:
                row[key] = stat[key]
        row.update({key: stat.get(native) for key, native in COMBAT_FIELDS.items()})
        rows.append(row)
    return _table(rows, END_MATCH_SCHEMA)


def time_slices(metadata, rows):
    """Keep replay state only, using metadata solely for authoritative identities."""
    info = _info(metadata)
    roster, output, seen, groups = _roster(info), [], set(), defaultdict(set)
    for raw in rows:
        slot = _int(raw.get("player_slot"), "player_slot")
        if slot not in roster:
            raise ValueError(f"replay has player slot {slot} outside metadata roster")
        identity = _identity(info, roster[slot])
        for field in ("match_id", "account_id", "hero_id", "team_id"):
            value = _int(raw.get(field), field)
            if value is not None and identity[field] is not None and value != identity[field]:
                raise ValueError(f"replay {field} conflicts with metadata for slot {slot}")
        time = _number(raw.get("game_time_s"), "game_time_s")
        observed = _number(raw.get("observed_time_s"), "observed_time_s")
        if time is None or time < 0 or observed is not None and (observed < 0 or observed > time + 1e-8):
            raise ValueError("invalid replay observation time or future backfill")
        key = (slot, time)
        if key in seen:
            raise ValueError(f"duplicate time-slice key {key}")
        seen.add(key)
        groups[time].add(slot)
        row = {field.name: raw.get(field.name) for field in TIME_SLICE_SCHEMA}
        row.update(identity)
        output.append(row)
    if any(slots != set(roster) for slots in groups.values()):
        raise ValueError("time slices do not contain the complete roster at every timestamp")
    output.sort(key=lambda row: (row["game_time_s"], row["player_slot"]))
    return _table(output, TIME_SLICE_SCHEMA)


def events(metadata, replay_events, definitions=None):
    """Reconcile trustworthy metadata events with selected, decoded replay messages.

    Hero kills have one shared event_id and victim/killer/assister rows. Metadata
    objectives and item records anchor identity; unknown notifications are not
    promoted to ability casts, claims, actual payments, or guessed objective IDs.
    """
    info, rows = _info(metadata), []
    roster = _roster(info)
    mid = _int(info.get("match_id"), "match_id")
    duration = _number(info.get("duration_s"))
    definitions = definitions.get("items", []) if isinstance(definitions, dict) else definitions or []
    items = {r["id"]: r for r in definitions if "id" in r}
    # Metadata duration has whole-second precision. The parser supplies only
    # observed active-gameplay events; retain its final fractional second.
    replay = [r for r in replay_events if r.get("game_time_s") is not None
              and 0 <= r["game_time_s"] <= duration + 1]
    kills = [r for r in replay if r.get("event_type", "").removeprefix("k_EUserMsg_") == "HeroKilled"]
    used_kills = set()

    def known_slot(value):
        return value if value in roster else None

    def emit(event_id, event_type, time, role=None, slot=None, source="metadata", **fields):
        slot = known_slot(slot)
        row = {"match_id": mid, "event_id": event_id, "event_time_s": time,
               "event_type": event_type, "participant_role": role, "player_slot": slot,
               "team_id": _team(roster[slot].get("team")) if slot is not None else None,
               "event_source": source, **fields}
        rows.append(row)
        return row

    def emit_kill(event_id, time, victim, killer, assisters, death=None, source="metadata"):
        death = death or {}
        common = {"actor_player_slot": killer, "target_player_slot": victim}
        pos = death.get("death_pos") or {}
        emit(event_id, "hero_kill", time, "victim", victim, source,
             **common, duration_s=death.get("death_duration_s"),
             time_to_kill_s=death.get("time_to_kill_s"), **{k: pos.get(k) for k in "xyz"})
        if killer is not None:
            pos = death.get("killer_pos") or {}
            emit(event_id, "hero_kill", time, "killer", killer, source,
                 **common, **{k: pos.get(k) for k in "xyz"})
        for assist in sorted({s for s in assisters if s in roster and s not in (victim, killer)}):
            emit(event_id, "hero_kill", time, "assister", assist, source, **common)

    for slot, player in sorted(roster.items()):
        for index, death in enumerate(player.get("death_details", [])):
            time = death.get("game_time_s")
            candidates = [(abs(r["game_time_s"] - time), i, r) for i, r in enumerate(kills)
                          if time is not None and i not in used_kills
                          and r.get("target_player_slot") == slot and abs(r["game_time_s"] - time) <= 2]
            candidates.sort(key=lambda candidate: candidate[0])
            match = candidates[0] if candidates and (len(candidates) == 1 or candidates[0][0] < candidates[1][0]) else None
            killer = known_slot(death.get("killer_player_slot"))
            assisters, source = [], "metadata"
            if match:
                _, replay_index, rec = match
                used_kills.add(replay_index)
                time, source = rec["game_time_s"], "metadata+replay"
                killer = known_slot(rec.get("actor_player_slot")) if rec.get("actor_player_slot") is not None else killer
                assisters = rec.get("assister_player_slots") or []
            emit_kill(f"{mid}:death:{slot}:{index}", time, slot, killer, assisters, death, source)
    for index, rec in enumerate(kills):
        if index not in used_kills:
            # An ambiguous replay report is not evidence of an additional death
            # when it could describe an existing metadata event (including ties).
            victim = known_slot(rec.get("target_player_slot"))
            if any(
                    row["event_type"] == "hero_kill" and row["event_time_s"] is not None
                    and (victim is None or row.get("target_player_slot") == victim)
                    and abs(row["event_time_s"] - rec["game_time_s"]) <= 2 for row in rows):
                continue
            emit_kill(f"{mid}:replay-kill:{index}", rec["game_time_s"],
                      known_slot(rec.get("target_player_slot")), known_slot(rec.get("actor_player_slot")),
                      rec.get("assister_player_slots") or [], source="replay")

    item_records = []
    for slot, player in sorted(roster.items()):
        for index, item in enumerate(player.get("items", [])):
            item_id, time = item.get("item_id"), item.get("game_time_s")
            kind = items.get(item_id, {}).get("type")
            upgrade = item.get("upgrade_id")
            # The protocol does not document the meaning of metadata upgrade_id
            # or the packing of upgrade_info. Replay change enums can refine a
            # matched record; zero alone is not proof that an ability unlocked.
            event_type = "ability_progression" if kind == "ability" else "item_acquired" if kind == "upgrade" else "item_record"
            fields = {"actor_player_slot": slot, "item_id": item_id if kind != "ability" else None,
                      "ability_id": item_id if kind == "ability" else None, "upgrade_id": upgrade,
                      "upgrade_info": item.get("upgrade_info"), "imbued_ability_id": item.get("imbued_ability_id")}
            row = emit(f"{mid}:item:{slot}:{index}", event_type, time, "player", slot, **fields)
            item_records.append((slot, item_id, False, row))
            if item.get("sold_time_s") is not None and item["sold_time_s"] > 0:
                row = emit(f"{mid}:item-removal:{slot}:{index}", "item_removed" if kind != "ability" else "ability_removed",
                           item["sold_time_s"], "player", slot, **fields)
                item_records.append((slot, item_id, True, row))

    # Purchased/sold notifications corroborate item records; never turn a cost
    # from static definitions into actual payment. Unknown change enums are skipped.
    for index, rec in enumerate(replay):
        name = rec.get("event_type", "").removeprefix("k_EUserMsg_")
        if name not in ("AbilitiesChanged", "ItemPurchaseNotification"):
            continue
        data, slot, time = rec.get("data") or {}, known_slot(rec.get("actor_player_slot")), rec["game_time_s"]
        if slot is None:
            continue
        item_id = data.get("abilityId")
        if item_id is None:
            continue
        kind = items.get(item_id, {}).get("type")
        if name == "AbilitiesChanged":
            # protobufjs Message.toJSON converts enum numbers to names. Accept
            # these verified values in either form; other versioned enums stay
            # uninterpreted (notably old EFailure/new ELeveledUp share value 4).
            change = data.get("change")
            if isinstance(change, str):
                change = {"EPurchased": 0, "EUpgraded": 1, "ESold": 2}.get(change)
        else:
            # The proto2 bool has an implicit false default. Its JSON omission
            # still means a purchase notification, unlike a missing state stat.
            sell = data.get("sell", False)
            change = 2 if sell is True else 0 if sell is False else None
        if change not in (0, 1, 2):
            continue
        removal = change == 2
        candidates = [(abs(row["event_time_s"] - time), row) for s, iid, removed, row in item_records
                      if s == slot and iid == item_id and removed == removal and row["event_time_s"] is not None
                      and abs(row["event_time_s"] - time) <= 2]
        if candidates:
            candidates.sort(key=lambda candidate: candidate[0])
            if len(candidates) == 1 or candidates[0][0] < candidates[1][0]:
                row = candidates[0][1]
                # Timing is already a verified gameplay clock. Keep exact metadata
                # upgrade details while adopting the more precise replay timestamp.
                row["event_time_s"] = time
                if row["event_source"] == "metadata":
                    row["event_source"] = "metadata+replay"
                if kind == "ability":
                    row["event_type"] = {0: "ability_unlock", 1: "ability_upgrade", 2: "ability_removed"}[change]
            continue
        # Two replay message types may describe the same acquisition.
        if any(s == slot and iid == item_id and removed == removal and row["event_time_s"] is not None
               and abs(row["event_time_s"] - time) <= 0.1 for s, iid, removed, row in item_records):
            continue
        if kind == "ability":
            event_type = {0: "ability_unlock", 1: "ability_upgrade", 2: "ability_removed"}[change]
        elif kind == "upgrade":
            event_type = "item_removed" if removal else "item_acquired" if change == 0 else "item_upgrade"
        else:
            event_type = "item_removed" if removal else "item_record"
        row = emit(f"{mid}:replay-item:{index}", event_type, time, "player", slot, "replay",
                   actor_player_slot=slot, item_id=item_id if kind != "ability" else None,
                   ability_id=item_id if kind == "ability" else None)
        item_records.append((slot, item_id, removal, row))

    for index, objective in enumerate(info.get("objectives", [])):
        oid, team = objective.get("team_objective_id"), _team(objective.get("team"))
        legacy = objective.get("legacy_objective_id")
        if oid is None and type(legacy) is int and 0 <= legacy < 32:
            team, oid = divmod(legacy, 16)
        for field, kind in (("first_damage_time_s", "objective_first_damage"),
                            ("destroyed_time_s", "objective_destroyed")):
            time = objective.get(field)
            if time is not None and time > 0:
                emit(f"{mid}:objective:{index}:{kind}", kind, time,
                     objective_id=oid, objective_team_id=team)
    for index, boss in enumerate(info.get("mid_boss", [])):
        event_id = f"{mid}:midboss:{index}"
        emit(event_id, "midboss_killed", boss.get("destroyed_time_s"),
             "killing_team", team_id=_team(boss.get("team_killed")))
        claim = _team(boss.get("team_claimed"))
        if claim is not None:
            emit(f"{event_id}:claim", "rejuvenator_claim", None, "claiming_team", team_id=claim,
                 claiming_team_id=claim, related_event_id=event_id)
    rows.sort(key=lambda row: (row["event_time_s"] is None,
                              row["event_time_s"] if row["event_time_s"] is not None else 0,
                              row["event_id"], row.get("participant_role") or ""))
    return _table(rows, EVENT_SCHEMA)
