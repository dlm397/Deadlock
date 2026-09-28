# Join observed replay state to metadata; keep source-specific timing.
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .normalize import aggregate_team_state
from .io import read_json, save_json


CONTROLLER_FIELDS = {
    "net_worth": "m_iGoldNetWorth", "kills": "m_iPlayerKills", "deaths": "m_iDeaths",
    "assists": "m_iPlayerAssists", "hero_id": "m_nHeroID", "health": "m_iHealth",
    "max_health": "m_iHealthMax", "alive": "m_bAlive", "level": "m_iLevel",
    "player_damage": "m_iHeroDamage", "boss_damage": "m_iObjectiveDamage",
    "player_healing": "m_iHeroHealing", "self_healing": "m_iSelfHealing",
    "has_rejuvenator": "m_bHasRejuvenator", "has_rebirth": "m_bHasRebirth",
    "max_ammo": "m_iMaxAmmo", "health_regen": "m_flHealthRegen",
    "respawn_time_engine": "m_flRespawnTime", "connection_state_code": "m_iConnected",
    "denies": "m_iDenies", "last_hits": "m_iLastHits", "kill_streak": "m_iKillStreak",
    "ability_points_net_worth": "m_iAPNetWorth",
}


def merge_player_fields(row, metadata, raw):
    controller, pawn = raw.get("raw_controller") or {}, raw.get("raw_pawn") or {}
    provenance = row.setdefault("field_provenance", {})
    for target, field in CONTROLLER_FIELDS.items():
        key = next((k for k in (field, f"m_PlayerDataGlobal.{field}") if controller.get(k) is not None), None)
        if key is not None:
            row[target] = controller[key]
            provenance[target] = f"replay_controller:{key}"
        elif raw.get(target) is not None:
            row[target] = raw[target]
            provenance[target] = "replay_exporter"
        else:
            row[target] = metadata.get(target)
            provenance[target] = "metadata_asof" if row[target] is not None else "unavailable"
    # ECurrencyType: https://docs.deadworks.net/api-reference/players/#ecurrencytype
    pawn_fields = {"unspent_souls": "m_nCurrencies.0000", "spent_souls": "m_nSpentCurrencies.0000",
                   "unspent_ability_points": "m_nCurrencies.0001", "ability_unlock_tokens": "m_nCurrencies.0002",
                   "network_disconnected": "m_bNetworkDisconnected"}
    for target, field in pawn_fields.items():
        row[target] = pawn.get(field)
        provenance[target] = f"replay_pawn:{field}" if row[target] is not None else "unavailable"
    row["unspent_souls_source_game_time"] = raw.get("source_game_time") if row["unspent_souls"] is not None else None
    row["unspent_souls_state_age"] = row.get("state_age") if row["unspent_souls"] is not None else None
    row["unspent_souls_quality"] = "observed_replay_currency" if row["unspent_souls"] is not None else "unavailable"
    for field in ("spirit_power", "weapon_power"):
        provenance[field] = metadata.get("field_provenance", {}).get(field, "unavailable")


def ndjson(path):
    path = Path(path)
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def replay_version(summary):
    # Sample engine buildNum=10854; asset client version in directory=6698.
    header = summary.get("header") or {}
    match = re.search(r"(?:^|[/\\])citadel_v(\d+)(?:[/\\]|$)", header.get("gameDirectory", ""))
    return int(match.group(1)) if match else summary.get("game_version")


def decode_legacy_events(events, schema):
    # Decode using this replay's descriptors; numeric event IDs can change.
    descriptors = schema.get("observed_events", {}).get("GE_Source1LegacyGameEventList", {}).get("fields", {}).get("descriptors", {}).get("example", [])
    lookup = {d["eventid"]: d for d in descriptors}
    rows = []
    for event in events:
        if event.get("event_type") != "GE_Source1LegacyGameEvent":
            continue
        data = event.get("raw_event_data") or {}
        if isinstance(data, str):
            data = json.loads(data)
        descriptor = lookup.get(data.get("eventid"), {})
        decoded = {}
        for field, value in zip(descriptor.get("keys", []), data.get("keys", [])):
            present_values = [v for k, v in value.items() if k.startswith("val")]
            decoded[field["name"]] = present_values[0] if len(present_values) == 1 else None
        rows.append({**event, "event_type": descriptor.get("name", f"unknown_legacy_{data.get('eventid')}"),
                     "network_event_type": "GE_Source1LegacyGameEvent", "decoded_event_data": decoded,
                     "raw_event_data": data, "descriptor": descriptor})
    return rows


def merge_replay(tables, status, provenance):
    if not status.get("replay_parsed"):
        return
    folder = Path(status["parser_output"])
    demo_provenance_path = folder.parent / "replay.dem.bz2.provenance.json"
    demo_provenance = read_json(demo_provenance_path, {})
    replay_provenance = {**provenance, "retrieved_at": demo_provenance.get("retrieved_at"),
                         "source_url": demo_provenance.get("source"),
                         "parsed_at": status.get("parser_summary", {}).get("retrieved_at"),
                         "metadata_retrieved_at": provenance.get("retrieved_at")}
    players = {row["player_id"]: row for row in tables["players"]}
    duration = tables["matches"][0].get("duration_s", tables["matches"][0].get("duration"))
    panel = {(r["player_id"], r["game_time"]): r for r in tables["player_state"]}
    entity_players = defaultdict(set)
    for raw in ndjson(folder / "player_state.ndjson.gz"):
        pid, t = raw.get("player_id"), raw.get("game_time")
        if pid not in players:
            continue
        for key in ("controller_index", "pawn_index"):
            if raw.get(key) is not None:
                entity_players[raw[key]].add(pid)
        # Arbitrary off-grid observations remain in the source archive.
        if (pid, t) not in panel:
            continue
        meta = panel[(pid, t)]
        p = raw.get("raw_pawn") or {}
        row = {**meta, **replay_provenance, **raw, "team": players[pid].get("team"),
               "source": "local_replay:deadem", "state_source": "replay",
               "state_quality": "observed_replay_state_sampled_asof",
               "observation_kind": "observed_replay_state_on_sample_grid",
               "metadata_source_game_time": meta.get("source_game_time"),
               "metadata_state_age": meta.get("state_age"), "metadata_net_worth": meta.get("net_worth")}
        field_provenance = {key: ("metadata_native_path" if key in ("health_percent", "move_type", "combat_type")
                                  else "metadata_asof") for key in meta}
        for key in raw:
            field_provenance[key] = "replay"
        row["pawn_health"] = p.get("m_iHealth")
        row["pawn_max_health"] = p.get("m_iMaxHealth")
        row["state_age"] = t - raw["source_game_time"] if raw.get("source_game_time") is not None else None
        row["field_provenance"] = field_provenance
        merge_player_fields(row, meta, raw)
        if raw.get("inventory") is not None:
            row["inventory"] = raw["inventory"]
            row["inventory_complete"] = all(x is not None for x in raw["inventory"])
            row["inventory_quality"] = "observed_replay_array"
            field_provenance["inventory"] = "replay_controller:m_vecUpgrades"
        else:
            row["inventory"] = meta.get("inventory")
            row["inventory_quality"] = meta.get("inventory_quality")
            field_provenance["inventory"] = "metadata_reconstructed"
        if raw.get("ability_upgrade_states") is not None:
            row["ability_states"] = raw["ability_upgrade_states"]
            field_provenance["ability_states"] = "replay_controller:m_vecAbilityUpgradeState"
        row["ability_levels"] = None  # m_nUpgradeInfo bit semantics not established.
        row["field_provenance"] = field_provenance
        # Full dictionaries stay in the compressed, timestamped source archive.
        row.pop("raw_controller", None)
        row.pop("raw_pawn", None)
        row["raw_state_file"] = f"raw/{provenance['match_id']}/sources/replay/player_state.ndjson.gz"
        panel[(pid, t)] = row
    tables["player_state"] = sorted(panel.values(), key=lambda r: (r["player_id"], r["game_time"]))

    def player_for_entity(index):
        candidates = entity_players.get(index, set())
        return next(iter(candidates)) if len(candidates) == 1 else None

    # Render/sound/transport messages stay in the complete compressed archive.
    audiovisual = {"k_EUserMsg_ParticipantStartSoundEvent", "k_EUserMsg_ParticipantStopSoundEvent",
                   "k_EUserMsg_ParticipantStopSoundEventHash", "k_EUserMsg_ParticipantSetSoundEventParams",
                   "k_EUserMsg_ParticipantSetLibraryStackFields", "k_EUserMsg_PostProcessingAnim",
                   "k_EUserMsg_TriggerDamageFlash"}
    counts = Counter()
    kill_count = 0
    for index, raw in enumerate(ndjson(folder / "events.ndjson.gz")):
        name, t = raw.get("event_type", "unknown"), raw.get("game_time")
        counts[name] += 1
        if not (name.startswith("k_EUserMsg_") or name == "GE_Source1LegacyGameEvent") or name in audiovisual:
            continue
        if t is None or t < 0 or (duration is not None and t > duration):
            continue
        event = {**replay_provenance, **raw, "source": "local_replay:deadem",
                 "event_id": f"{provenance['match_id']}:replay:{index}",
                 "observation_kind": "observed_network_message"}
        data = raw.get("raw_event_data") or {}
        actor = data.get("entindexAttacker", data.get("entindexCaster"))
        victim = data.get("entindexVictim")
        event["player_id"] = player_for_entity(actor)
        event["other_player_id"] = player_for_entity(victim)
        event["team"] = players.get(event["player_id"], {}).get("team")
        event["target"] = data.get("abilityId", data.get("entityKilledClass"))
        event["value"] = data.get("damage", data.get("delta"))
        pos = data.get("origin", data.get("entityPosition", {})) or {}
        if isinstance(pos, dict):
            event.update({axis: pos.get(axis) for axis in ("x", "y", "z")})
        if name == "k_EUserMsg_HeroKilled":
            kill_count += 1
            event["assister_player_ids"] = [player_for_entity(x) for x in data.get("entindexAssisters", [])]
        tables["events"].append(event)
    tables["replay_event_counts"] = [{**provenance, "event_type": name, "count": count,
                                      "source": "local_replay:deadem", "scope": "entire_demo_including_pregame_postgame"}
                                     for name, count in counts.items()]
    tables["matches"][0]["replay_kill_event_count"] = kill_count
    schema = read_json(folder / "schema.json", {})
    legacy = decode_legacy_events(tables["events"], schema)
    tables["legacy_events"] = legacy
    save_json(folder.parent / "schema/observed_legacy_events.json",
              {name: {"count": sum(r["event_type"] == name for r in legacy),
                      "fields": sorted({k for r in legacy if r["event_type"] == name for k in r["decoded_event_data"]})}
               for name in {r["event_type"] for r in legacy}})
    query = status.get("demo_query", {})
    if query.get("status") == "done":
        import pyarrow.parquet as pq
        api_count = pq.read_metadata(query["path"]).num_rows
        tables["matches"][0]["api_demo_kill_event_count"] = api_count
        tables["matches"][0]["demo_kill_count_difference"] = kill_count - api_count


def aggregate_teams(tables, provenance):
    tables["team_state"] = aggregate_team_state(tables["players"], tables["player_state"],
                                                 provenance, tables.get("metadata_objectives"))
    enrich_purchase_state(tables["purchases"], tables["player_state"])


def enrich_purchase_state(purchases, states):
    # These are past sampled balances, not exact pre/post-transaction amounts.
    from bisect import bisect_right
    groups = defaultdict(list)
    for row in states:
        groups[(row["match_id"], row["player_id"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r["game_time"])
    for purchase in purchases:
        rows = groups.get((purchase["match_id"], purchase["player_id"]), [])
        time = purchase.get("game_time")
        idx = bisect_right([r["game_time"] for r in rows], time) - 1 if time is not None else -1
        past = rows[idx] if idx >= 0 else {}
        for field in ("net_worth", "unspent_souls"):
            observed = past.get(f"{field}_source_game_time", past.get("source_game_time")) if past.get(field) is not None else None
            purchase[field] = past.get(field)
            purchase[f"{field}_source_time"] = observed
            purchase[f"{field}_state_age"] = time - observed if time is not None and observed is not None else None
            purchase[f"{field}_quality"] = "last_observed_before_transaction_not_exact" if observed is not None else "unavailable"
