import gzip
import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from math import floor, isfinite
from pathlib import Path

from .sources import read_json


PANEL_FIELDS = ('net_worth', 'kills', 'deaths', 'assists', 'max_health', 'weapon_power',
                'tech_power', 'spirit_power', 'player_damage', 'boss_damage', 'player_healing',
                'level', 'ability_points', 'shots_hit', 'shots_missed')
OBJECTIVES = dict(enumerate(('core', *['guardian'] * 4, *['walker'] * 4, 'patron',
                            'shrine', 'shrine', *['base_guardian'] * 4)))
COMBAT_TYPES = ('out', 'player', 'enemy_npc', 'neutral')
MOVE_TYPES = ('normal', 'ability', 'ability_debuff', 'ground_dash', 'slide',
              'rope_climbing', 'ziplining', 'in_air', 'air_dash')
STAT_TYPES = ('damage', 'healing', 'heal_prevented', 'mitigated', 'lethal_damage', 'regen')
AUDIOVISUAL = {'ParticipantStartSoundEvent', 'ParticipantStopSoundEvent', 'ParticipantStopSoundEventHash',
               'ParticipantSetSoundEventParams', 'ParticipantSetLibraryStackFields',
               'PostProcessingAnim', 'TriggerDamageFlash'}
CONTROLLER_FIELDS = {
    'net_worth': 'm_iGoldNetWorth', 'kills': 'm_iPlayerKills', 'deaths': 'm_iDeaths',
    'assists': 'm_iPlayerAssists', 'hero_id': 'm_nHeroID', 'health': 'm_iHealth',
    'max_health': 'm_iHealthMax', 'alive': 'm_bAlive', 'level': 'm_iLevel',
    'player_damage': 'm_iHeroDamage', 'boss_damage': 'm_iObjectiveDamage',
    'player_healing': 'm_iHeroHealing', 'self_healing': 'm_iSelfHealing',
    'has_rejuvenator': 'm_bHasRejuvenator', 'has_rebirth': 'm_bHasRebirth',
    'max_ammo': 'm_iMaxAmmo', 'health_regen': 'm_flHealthRegen',
    'respawn_time_engine': 'm_flRespawnTime', 'connection_state_code': 'm_iConnected',
    'denies': 'm_iDenies', 'last_hits': 'm_iLastHits', 'kill_streak': 'm_iKillStreak',
    'ability_points_net_worth': 'm_iAPNetWorth',
}


def ndjson(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def replay_version(summary):
    match = re.search(r'(?:^|[/\\])citadel_v(\d+)(?:[/\\]|$)',
                      (summary.get('header') or {}).get('gameDirectory', ''))
    return int(match.group(1)) if match else summary.get('game_version')


def identity(player):
    return {**{key: player.get(key) for key in ('account_id', 'player_slot', 'hero_id', 'team')},
            'player_id': player.get('account_id')}


def at(values, index):
    return values[index] if values is not None and isinstance(index, int) and 0 <= index < len(values) else None


def power_fields(row, snapshot, source_time, time):
    for field, native in (('spirit_power', 'tech_power'), ('weapon_power', 'weapon_power')):
        observed = source_time if snapshot.get(native) is not None else None
        row.update({field: snapshot.get(native), f'{field}_source_game_time': observed,
                    f'{field}_state_age': time - observed if time is not None and observed is not None else None,
                    f'{field}_quality': ('observed_sparse_snapshot' if time == observed else 'metadata_asof')
                    if observed is not None else 'unavailable'})
        row.setdefault('field_provenance', {})[field] = f'metadata_stats:{native}' if observed is not None else 'unavailable'


def inventory_at(records, time):
    inventory, abilities = set(), {}
    for rec in records:
        stamp = rec.get('game_time_s')
        if stamp is None or stamp > time:
            continue
        if rec['definition_type'] == 'ability':
            abilities[str(rec['item_id'])] = {k: rec.get(k) for k in ('upgrade_id', 'upgrade_info', 'game_time_s')}
        elif rec['definition_type'] == 'upgrade' and rec.get('sold_time_s') is not None:
            if rec['sold_time_s'] == 0 or rec['sold_time_s'] > time:
                inventory.add(rec['item_id'])
    return sorted(inventory), abilities


def normalize(metadata, replay_folder, definitions, provenance, interval=20):
    info = metadata['match_info']
    players, duration = info['players'], info['duration_s']
    assert isfinite(interval) and interval > 0, 'Sampling interval must be positive'
    assert info['match_id'] == provenance['match_id'], 'Metadata belongs to another match'
    assert info.get('winning_team') in (None, 0, 1), 'Unknown winner encoding'
    assert len({p['player_slot'] for p in players}) == len(players), 'Duplicate player slots'
    assert all(p['team'] in (0, 1) for p in players), 'Unknown team encoding'
    accounts = [p['account_id'] for p in players if p.get('account_id') is not None]
    assert len(accounts) == len(set(accounts)), 'Duplicate player accounts'
    base = dict(provenance)
    exact = base.get('game_version') is not None and base.get('game_version') == base.get('definitions_version')
    asset_version = definitions.get('items_provenance', {}).get('definitions_version', base.get('definitions_version'))
    assert not exact or asset_version == base['game_version'], 'Item definitions are from another game version'
    items = {r['id']: r for r in definitions.get('items', [])} if exact else {}
    names = ('matches players game_versions metadata_teams metadata_damage metadata_custom_stats metadata_gold_sources '
             'player_state team_state metadata_player_stats metadata_player_paths purchases metadata_item_records '
             'events metadata_objectives metadata_midboss legacy_events replay_event_counts').split()
    tables = {name: [] for name in names}
    slots = {p['player_slot']: p for p in players}
    paths = info.get('match_paths') or {}
    paths_by_slot = {p['player_slot']: p for p in paths.get('paths', [])}
    custom_names = {r['id']: r.get('name') for r in info.get('custom_user_stats', [])}
    grid = [n * interval for n in range(floor(duration / interval) + 1)]
    if grid[-1] != duration:
        grid.append(duration)

    def event(kind, time, raw, **fields):
        tables['events'].append({**base, 'event_id': f"metadata:{base['match_id']}:{len(tables['events'])}",
                                'event_type': kind, 'game_time': time, 'player_id': None,
                                'other_player_id': None, 'team': None, 'target': None, 'value': None,
                                'x': None, 'y': None, 'z': None, 'observation_quality': 'observed_metadata_event',
                                'raw_event_data': raw, **fields})

    for player in players:
        pb = {**base, **identity(player)}
        stats = sorted((s for s in player.get('stats', []) if s.get('time_stamp_s') is not None), key=lambda s: s['time_stamp_s'])
        stamps = [s['time_stamp_s'] for s in stats]
        for index, stat in enumerate(player.get('stats', [])):
            sb = {**pb, 'game_time': stat.get('time_stamp_s'), 'source_stat_index': index}
            row = {**sb, **stat, 'observation_quality': 'observed_sparse_snapshot'}
            power_fields(row, stat, stat.get('time_stamp_s'), stat.get('time_stamp_s'))
            tables['metadata_player_stats'].append(row)
            tables['metadata_custom_stats'].extend({**sb, **r, 'stat_name': custom_names.get(r['id'])} for r in stat.get('custom_user_stats', []))
            tables['metadata_gold_sources'].extend({**sb, **{k: v for k, v in r.items() if k != 'source'},
                                                    'gold_source_code': r.get('source')} for r in stat.get('gold_sources', []))
        last = stats[-1] if stats else {}
        at_end = bool(last) and last['time_stamp_s'] == duration
        final = {**player, **pb, 'win': player['team'] == info['winning_team'] if info.get('winning_team') is not None else None,
                 'final_stats_time': last.get('time_stamp_s'), 'final_stats_at_match_end': at_end,
                 'latest_metadata_stats': last or None, 'final_net_worth': player.get('net_worth'),
                 'rank_badge': (player.get('player_rank_data') or {}).get('initial_display_rank'),
                 'objective_damage': last.get('boss_damage') if at_end else None,
                 'healing': last.get('player_healing') if at_end else None}
        if at_end:
            for key, value in last.items():
                final[f'final_stats_{key}' if key in final else key] = value
        for field in PANEL_FIELDS:
            final.setdefault(field, None)
        power_fields(final, last if at_end else {}, duration if at_end else None, duration)
        hits, misses = final.get('shots_hit'), final.get('shots_missed')
        final['shot_accuracy'] = hits / (hits + misses) if hits is not None and misses is not None and hits + misses > 0 else None

        records = []
        for index, item in enumerate(player.get('items', [])):
            definition = items.get(item['item_id'], {})
            kind = definition.get('type')
            rec = {**pb, **item, 'game_time': item.get('game_time_s'), 'item_record_index': index,
                   'item_name': definition.get('name'), 'definition_type': kind,
                   'definition_match': 'exact_game_version' if definition else 'unavailable',
                   'item_cost': definition.get('cost') if kind == 'upgrade' else None, 'raw_item_data': item}
            records.append(rec)
            tables['metadata_item_records'].append(rec)
            transaction = 'purchase' if kind == 'upgrade' else 'unclassified_item_record'
            if kind == 'ability':
                transaction = 'ability_unlock' if item.get('upgrade_id') == 0 else 'ability_upgrade' if item.get('upgrade_id') is not None else 'ability_change'
            event(transaction, rec['game_time'], item, **identity(player), target=item['item_id'],
                  item_id=item['item_id'], item_name=definition.get('name'), definition_type=kind)
            if kind != 'ability':
                tables['purchases'].append({**rec, 'transaction_type': transaction, 'souls_before': None,
                                           'souls_after': None, 'amount_paid': None,
                                           'transaction_quality': 'observed_acquisition' if kind == 'upgrade' else 'unclassified'})
            if (item.get('sold_time_s') or 0) > 0:
                tables['purchases'].append({**rec, 'game_time': item['sold_time_s'], 'transaction_type': 'item_removed',
                                           'removal_reason': 'sale_or_upgrade_unknown', 'souls_before': None,
                                           'souls_after': None, 'amount_paid': None, 'transaction_quality': 'observed_removal_reason_unknown'})
                event('item_removed', item['sold_time_s'], item, **identity(player), target=item['item_id'],
                      item_id=item['item_id'], removal_reason='sale_or_upgrade_unknown')
        records.sort(key=lambda r: r['game_time_s'] if r.get('game_time_s') is not None else float('inf'))
        inventory_complete = bool(items) and 'items' in player and all(r['definition_type'] in ('upgrade', 'ability')
            and r.get('game_time_s') is not None and (r['definition_type'] != 'upgrade' or r.get('sold_time_s') is not None) for r in records)
        quality = 'reconstructed_from_item_records' if inventory_complete else 'partial_or_unavailable'
        final['final_inventory'], final['final_ability_states'] = inventory_at(records, duration)
        final.update(final_inventory_complete=inventory_complete, final_inventory_quality=quality)
        tables['players'].append(final)

        for index, death in enumerate(player.get('death_details', [])):
            pos, killer_pos = death.get('death_pos') or {}, death.get('killer_pos') or {}
            event('death', death.get('game_time_s'), death, **identity(player),
                  other_player_id=slots.get(death.get('killer_player_slot'), {}).get('account_id'),
                  killer_player_slot=death.get('killer_player_slot'), death_record_index=index,
                  death_duration_s=death.get('death_duration_s'), time_to_kill_s=death.get('time_to_kill_s'),
                  **{axis: pos.get(axis) for axis in 'xyz'}, **{f'killer_{axis}': killer_pos.get(axis) for axis in 'xyz'})
        for ping in player.get('pings', []):
            event('player_ping', ping.get('game_time_s'), ping, **identity(player), target=ping.get('ping_type'), value=ping.get('ping_data'))
        if (player.get('abandon_match_time_s') or 0) > 0:
            event('match_abandon', player['abandon_match_time_s'], {'abandon_match_time_s': player['abandon_match_time_s']}, **identity(player))

        path = paths_by_slot.get(player['player_slot'], {})
        path_rows = []
        for index in range(max((len(path.get(k) or []) for k in ('x_pos', 'y_pos', 'health', 'combat_type', 'move_type')), default=0)):
            step = paths.get('interval_s')
            row = {**pb, 'path_index': index, 'game_time': index * step if step else None,
                   'path_interval_s': step, 'path_version': paths.get('version'), 'health': None, 'z': None,
                   'health_percent': at(path.get('health'), index), 'combat_type_code': at(path.get('combat_type'), index),
                   'move_type_code': at(path.get('move_type'), index), 'observation_quality': 'observed_quantized_path'}
            for axis in 'xy':
                raw, low, high, resolution = at(path.get(f'{axis}_pos'), index), path.get(f'{axis}_min'), path.get(f'{axis}_max'), paths.get(f'{axis}_resolution')
                row.update({f'{axis}_quantized': raw, f'{axis}_min': low, f'{axis}_max': high, f'{axis}_resolution': resolution,
                            axis: low + raw * (high - low) / resolution if None not in (raw, low, high, resolution) and resolution > 0 else None})
            sentinel = row['x_quantized'] == row['y_quantized'] == row['health_percent'] == 0
            if sentinel:
                row.update(x=None, y=None)
            row.update(position_quality='unavailable_dead_sentinel' if sentinel else 'reconstructed_from_quantized_path',
                       alive=row['health_percent'] > 0 if row['health_percent'] is not None else None,
                       alive_quality='reconstructed_from_health_percent' if row['health_percent'] is not None else 'unavailable',
                       combat_type=at(COMBAT_TYPES, row['combat_type_code']), move_type=at(MOVE_TYPES, row['move_type_code']))
            path_rows.append(row)
        tables['metadata_player_paths'].extend(path_rows)
        path_times = [r['game_time'] for r in path_rows]
        for time in grid:
            index = bisect_right(stamps, time) - 1
            past = stats[index] if index >= 0 else {}
            source_time = past.get('time_stamp_s')
            inventory, abilities = inventory_at(records, time)
            row = {**pb, **dict.fromkeys(PANEL_FIELDS), **past, 'game_time': time, 'source_game_time': source_time,
                   'state_age': time - source_time if source_time is not None else None,
                   'state_quality': 'observed_sparse_snapshot' if source_time == time else 'metadata_asof' if source_time is not None else 'unavailable_before_first_snapshot',
                   'souls': None, 'unspent_souls': None, 'spent_souls': None, 'health': None, 'z': None,
                   'velocity_x': None, 'velocity_y': None, 'velocity_z': None, 'inventory': inventory,
                   'inventory_complete': inventory_complete, 'inventory_quality': quality, 'ability_states': abilities, 'ability_levels': None,
                   'net_worth_source_game_time': source_time if past.get('net_worth') is not None else None,
                   'field_provenance': {'net_worth': 'metadata_asof' if past.get('net_worth') is not None else 'unavailable'}}
            power_fields(row, past, source_time, time)
            pindex = bisect_right(path_times, time) - 1
            path_row = path_rows[pindex] if pindex >= 0 else {}
            ptime = path_row.get('game_time')
            if ptime is not None and time - ptime >= paths['interval_s']:
                path_row, ptime = {}, None
            row.update({key: path_row.get(key) for key in ('x', 'y', 'health_percent', 'alive', 'alive_quality',
                        'combat_type_code', 'combat_type', 'move_type_code', 'move_type', 'position_quality')})
            row.update(path_source_game_time=ptime, path_state_age=time - ptime if ptime is not None else None,
                       path_quality='observed_path_sample' if ptime == time else 'reconstructed_last_path_sample' if ptime is not None else 'unavailable')
            tables['player_state'].append(row)

    for objective in info.get('objectives', []):
        oid, team = objective.get('team_objective_id'), objective.get('team')
        if oid is None and objective.get('legacy_objective_id') is not None:
            team, oid = divmod(objective['legacy_objective_id'], 16)
        kind = OBJECTIVES.get(oid, 'unknown')
        tables['metadata_objectives'].append({**base, **objective, 'objective_id': oid, 'team': team,
                                            'objective_type': kind, 'team_semantics': 'owner_of_objective'})
        for field, name in (('destroyed_time_s', 'objective_destroyed'), ('first_damage_time_s', 'objective_first_damage')):
            if (objective.get(field) or 0) > 0:
                event(name, objective[field], objective, team=team, target=oid, objective_type=kind,
                      team_semantics='owner_of_objective', **({'attacking_team': 1 - team if team in (0, 1) else None} if field == 'destroyed_time_s' else {}))
    for index, boss in enumerate(info.get('mid_boss', [])):
        tables['metadata_midboss'].append({**base, **boss, 'midboss_index': index, 'claim_time': None, 'claim_time_quality': 'unavailable_in_metadata'})
        event('midboss_killed', boss.get('destroyed_time_s'), boss, team=boss.get('team_killed'), midboss_index=index)
        if boss.get('team_claimed') in (0, 1):
            event('rejuvenator_claim_team', None, boss, team=boss['team_claimed'], midboss_index=index,
                  associated_midboss_destroyed_time=boss.get('destroyed_time_s'), time_quality='unavailable_in_metadata')
    for pause in info.get('match_pauses', []):
        event('match_pause', pause.get('game_time_s'), pause, **identity(slots.get(pause.get('player_slot'), {})), value=pause.get('pause_duration_s'))
    for replay in info.get('watched_death_replays', []):
        event('watched_death_replay', replay.get('game_time_s'), replay, **identity(slots.get(replay.get('player_slot'), {})))
    tables['metadata_teams'] = [{**base, **row} for row in info.get('teams', [])]
    matrix = info.get('damage_matrix') or {}
    details = matrix.get('source_details') or {}
    for dealer in matrix.get('damage_dealers', []):
        for source in dealer.get('damage_sources', []):
            index = source.get('source_details_index')
            for target in source.get('damage_to_players', []):
                values = target.get('damage') or []
                tables['metadata_damage'].append({**base, 'dealer_player_slot': dealer.get('dealer_player_slot'),
                    'player_id': slots.get(dealer.get('dealer_player_slot'), {}).get('account_id'),
                    'target_player_slot': target.get('target_player_slot'), 'other_player_id': slots.get(target.get('target_player_slot'), {}).get('account_id'),
                    'source_details_index': index, 'source_name': at(details.get('source_name'), index),
                    'stat_type_code': at(details.get('stat_type'), index), 'stat_type': at(STAT_TYPES, at(details.get('stat_type'), index)),
                    'final_value': values[-1] if values else None, 'damage_values': values, 'sample_time_s': matrix.get('sample_time_s'),
                    'time_alignment': 'raw_arrays_preserved_no_assumed_padding', 'raw_damage_data': target})
    bosses = sorted((b for b in info.get('mid_boss', []) if b.get('destroyed_time_s') is not None), key=lambda b: b['destroyed_time_s'])
    first = bosses[0] if bosses else {}
    match = {**info, **base, 'duration': duration, 'metadata_envelope_extra': {k: v for k, v in metadata.items() if k != 'match_info'},
             'first_midboss_time': first.get('destroyed_time_s'), 'first_midboss_team': first.get('team_killed'),
             'first_rejuvenator_claim_team': first.get('team_claimed'), 'first_rejuvenator_claim_time': None,
             'rejuvenator_claim_time_quality': 'unavailable_in_metadata'}
    for team in (0, 1):
        roster = [p for p in players if p['team'] == team]
        values = [p['net_worth'] for p in roster if p.get('net_worth') is not None]
        match.update({f'team{team}_final_net_worth': sum(values) if roster and len(values) == len(roster) else None,
                      f'team{team}_net_worth_observed_players': len(values), f'team{team}_player_count': len(roster)})
    tables['matches'] = [match]
    summary = read_json(Path(replay_folder) / 'summary.json', {}) if replay_folder else {}
    if replay_folder:
        assert summary.get('sample_interval') == interval, 'Replay sample interval differs; reparse the demo'
        assert replay_version(summary) == base.get('game_version'), 'Replay game version differs'
        merge_replay(tables, Path(replay_folder), base, summary)
    tables['game_versions'] = [{**base, 'exact_assets_available': exact and bool(items),
                                'game_mode_version': info.get('game_mode_version'), 'replay_header': summary.get('header'),
                                'replay_parser': summary.get('parser'), 'replay_parser_version': summary.get('parser_version'),
                                'replay_decoder_extensions': summary.get('decoder_extensions', [])}]
    tables['team_state'] = aggregate_teams(tables, base)
    states = defaultdict(list)
    for row in tables['player_state']:
        states[row['player_slot']].append(row)
    for purchase in tables['purchases']:
        rows, time = states[purchase['player_slot']], purchase.get('game_time')
        index = bisect_right([r['game_time'] for r in rows], time) - 1 if time is not None else -1
        past = rows[index] if index >= 0 else {}
        for field in ('net_worth', 'unspent_souls'):
            observed = past.get(f'{field}_source_game_time', past.get('source_game_time')) if past.get(field) is not None else None
            purchase.update({field: past.get(field), f'{field}_source_time': observed,
                             f'{field}_state_age': time - observed if observed is not None else None,
                             f'{field}_quality': 'last_observed_before_transaction_not_exact' if observed is not None else 'unavailable'})
    return tables


def merge_replay(tables, folder, base, summary):
    downloaded = read_json(folder.parent / 'replay.dem.bz2.provenance.json', {})
    replay_source = {**base, 'source_url': downloaded.get('source'), 'retrieved_at': downloaded.get('retrieved_at'),
                     'metadata_retrieved_at': base.get('retrieved_at'), 'parsed_at': summary.get('retrieved_at')}
    players = {p['player_id']: p for p in tables['players'] if p['player_id'] is not None}
    panel = {(r['player_slot'], r['game_time']): r for r in tables['player_state']}
    entities, seen = defaultdict(set), set()
    for raw in ndjson(folder / 'player_state.ndjson.gz'):
        pid, time = raw.get('player_id'), raw.get('game_time')
        if pid not in players:
            continue
        player = players[pid]
        assert raw.get('player_slot') == player['player_slot'], 'Replay and metadata player slots disagree'
        assert raw.get('hero_id') in (None, 0, player['hero_id']), 'Replay and metadata heroes disagree'
        assert raw.get('team_engine') in (None, player['team'] + 2), 'Replay and metadata teams disagree'
        for key in ('controller_index', 'pawn_index'):
            if raw.get(key) is not None:
                entities[raw[key]].add(pid)
        key = (player['player_slot'], time)
        if key not in panel:
            continue
        assert key not in seen, 'Duplicate replay observations'
        seen.add(key)
        observed = raw.get('source_game_time')
        assert observed is None or observed <= time, 'Replay state contains future information'
        meta, controller, pawn = panel[key], raw.get('raw_controller') or {}, raw.get('raw_pawn') or {}
        row = {**meta, **replay_source, **{k: v for k, v in raw.items() if k not in ('raw_controller', 'raw_pawn')},
               'team': player['team'], 'source': 'local_replay:deadem', 'state_source': 'replay',
               'state_quality': 'observed_replay_state_sampled_asof', 'observation_kind': 'observed_replay_state_on_sample_grid',
               'metadata_source_game_time': meta.get('source_game_time'), 'metadata_state_age': meta.get('state_age'),
               'metadata_net_worth': meta.get('net_worth'), 'state_age': time - observed if observed is not None else None,
               'pawn_health': pawn.get('m_iHealth'), 'pawn_max_health': pawn.get('m_iMaxHealth'),
               'field_provenance': dict(meta.get('field_provenance', {})),
               'raw_state_file': f"raw/{base['match_id']}/sources/replay/player_state.ndjson.gz"}
        position_fields = sum(raw.get(axis) is not None for axis in 'xyz')
        row['position_quality'] = ('reconstructed_from_replay_cells' if position_fields == 3 else
                                   'partial_reconstructed_from_replay_cells' if position_fields else 'unavailable')
        for axis in 'xyz':
            row['field_provenance'][axis] = 'replay_position_cells' if raw.get(axis) is not None else 'unavailable'
        for field, native in CONTROLLER_FIELDS.items():
            source = next((k for k in (native, f'm_PlayerDataGlobal.{native}') if controller.get(k) is not None), None)
            row[field] = controller[source] if source else raw.get(field, meta.get(field))
            if row[field] is None:
                row[field] = meta.get(field)
            lineage = f'replay_controller:{source}' if source else 'replay_exporter' if raw.get(field) is not None else 'metadata_asof' if row[field] is not None else 'unavailable'
            row['field_provenance'][field] = lineage
            if field == 'net_worth':
                row['net_worth_source_game_time'] = observed if lineage.startswith('replay') else meta.get('net_worth_source_game_time')
        for field, native in {'unspent_souls': 'm_nCurrencies.0000', 'spent_souls': 'm_nSpentCurrencies.0000',
                              'unspent_ability_points': 'm_nCurrencies.0001', 'ability_unlock_tokens': 'm_nCurrencies.0002',
                              'network_disconnected': 'm_bNetworkDisconnected'}.items():
            row[field] = pawn.get(native)
            row['field_provenance'][field] = f'replay_pawn:{native}' if row[field] is not None else 'unavailable'
        row.update(unspent_souls_source_game_time=observed if row['unspent_souls'] is not None else None,
                   unspent_souls_state_age=row['state_age'] if row['unspent_souls'] is not None else None,
                   unspent_souls_quality='observed_replay_currency' if row['unspent_souls'] is not None else 'unavailable')
        if raw.get('inventory') is not None:
            row.update(inventory=raw['inventory'], inventory_complete=all(x is not None for x in raw['inventory']), inventory_quality='observed_replay_array')
            row['field_provenance']['inventory'] = 'replay_controller:m_vecUpgrades'
        else:
            row['inventory'] = meta.get('inventory')
            row['field_provenance']['inventory'] = 'metadata_reconstructed'
        row['ability_states'] = raw.get('ability_upgrade_states') if raw.get('ability_upgrade_states') is not None else meta.get('ability_states')
        row['field_provenance']['ability_states'] = ('replay_controller:m_vecAbilityUpgradeState'
            if raw.get('ability_upgrade_states') is not None else 'metadata_reconstructed')
        row['field_provenance']['abilities'] = 'replay_entities' if raw.get('abilities') is not None else 'unavailable'
        panel[key] = row
    assert seen, 'No replay observations align with metadata'
    tables['player_state'] = sorted(panel.values(), key=lambda r: (r['player_slot'], r['game_time']))
    entity_player = {index: next(iter(ids)) for index, ids in entities.items() if len(ids) == 1}
    schema = read_json(folder / 'schema.json', {})
    descriptors = {d['eventid']: d for d in schema.get('observed_events', {}).get('GE_Source1LegacyGameEventList', {}).get('fields', {}).get('descriptors', {}).get('example', [])}
    counts, kills = Counter(), 0
    for index, raw in enumerate(ndjson(folder / 'events.ndjson.gz')):
        name, time = raw.get('event_type', 'unknown'), raw.get('game_time')
        counts[name] += 1
        audiovisual = name.removeprefix('k_EUserMsg_') in AUDIOVISUAL
        if audiovisual or not (name.startswith('k_EUserMsg_') or name == 'GE_Source1LegacyGameEvent'):
            continue
        if time is None or not 0 <= time <= tables['matches'][0]['duration_s']:
            continue
        data = raw.get('raw_event_data') or {}
        actor = entity_player.get(data.get('entindexAttacker', data.get('entindexCaster')))
        row = {**replay_source, **raw, 'source': 'local_replay:deadem', 'event_id': f"{base['match_id']}:replay:{index}",
               'observation_kind': 'observed_network_message', 'player_id': actor,
               'other_player_id': entity_player.get(data.get('entindexVictim')), 'team': players.get(actor, {}).get('team'),
               'target': data.get('abilityId', data.get('entityKilledClass')), 'value': data.get('damage', data.get('delta'))}
        pos = data.get('origin', data.get('entityPosition', {})) or {}
        if isinstance(pos, dict):
            row.update({axis: pos.get(axis) for axis in 'xyz'})
        if name == 'k_EUserMsg_HeroKilled':
            kills += 1
            row['assister_player_ids'] = [entity_player.get(x) for x in data.get('entindexAssisters', [])]
        tables['events'].append(row)
        if name == 'GE_Source1LegacyGameEvent':
            descriptor = descriptors.get(data.get('eventid'), {})
            decoded = {}
            for key, value in zip(descriptor.get('keys', []), data.get('keys', [])):
                values = [v for k, v in value.items() if k.startswith('val')]
                decoded[key['name']] = values[0] if len(values) == 1 else None
            tables['legacy_events'].append({**row, 'event_type': descriptor.get('name', f"unknown_legacy_{data.get('eventid')}"),
                                            'network_event_type': name, 'decoded_event_data': decoded, 'descriptor': descriptor})
    tables['matches'][0]['replay_kill_event_count'] = kills
    tables['replay_event_counts'] = [{**base, 'event_type': name, 'count': count, 'source': 'local_replay:deadem',
                                    'scope': 'entire_demo_including_pregame_postgame'} for name, count in counts.items()]


def aggregate_teams(tables, base):
    rosters = {team: {p['player_slot'] for p in tables['players'] if p['team'] == team} for team in (0, 1)}
    groups = defaultdict(list)
    for row in tables['player_state']:
        groups[row['game_time'], row['team']].append(row)
    panel = {}
    for (time, team), rows in groups.items():
        complete = len(rows) == len(rosters[team]) and {r['player_slot'] for r in rows} == rosters[team]
        result = {**base, 'game_time': time, 'team': team, 'player_count': len(rows), 'expected_players': len(rosters[team]),
                  'roster_complete': complete, 'source': 'aggregate_player_state', 'aggregation_quality': 'sum_of_available_player_state'}
        for target, field in (('team_net_worth', 'net_worth'), ('team_total_souls', 'souls'), ('team_unspent_souls', 'unspent_souls'),
                              ('team_kills', 'kills'), ('team_deaths', 'deaths'), ('team_assists', 'assists'), ('alive_players', 'alive')):
            values = [r[field] for r in rows if r.get(field) is not None]
            result[target] = sum(values) if complete and len(values) == len(rows) else None
            result[f'{field}_observed_players'] = len(values)
        times = [r.get('net_worth_source_game_time', r.get('source_game_time')) for r in rows]
        known = [t for t in times if t is not None]
        result.update(net_worth=result['team_net_worth'], souls=result['team_total_souls'], kills=result['team_kills'], deaths=result['team_deaths'],
                      source_game_time_min=min(known) if known else None, source_game_time_max=max(known) if known else None,
                      state_age_max=time - min(known) if known else None,
                      economy_complete=complete and len(known) == len(rows) and result['team_net_worth'] is not None,
                      player_state_sources=sorted({r.get('source', 'unknown') for r in rows}))
        result['economy_simultaneous_observations'] = result['economy_complete'] and len(set(known)) == 1
        result['objective_state'] = [{'objective_id': o['objective_id'], 'objective_type': o['objective_type'],
                                      'destroyed': 0 < o['destroyed_time_s'] <= time if o.get('destroyed_time_s') is not None else None}
                                     for o in tables['metadata_objectives'] if o['team'] == team]
        result['objective_state_quality'] = 'reconstructed_for_objectives_reported_in_metadata'
        panel[time, team] = result
    for (time, team), row in panel.items():
        own, other = row['team_net_worth'], panel.get((time, 1 - team), {}).get('team_net_worth')
        row['net_worth_advantage'] = own - other if own is not None and other is not None else None
        row['team_net_worth_advantage'] = row['net_worth_advantage']
    return list(panel.values())
