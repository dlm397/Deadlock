import {createReadStream, readFileSync, mkdirSync, openSync, writeSync, closeSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {Parser, InterceptorStage as Stage} from 'deadem';
import {FieldDecoderDescriptor} from '@deademx/engine';

const [input, metadataPath, output, intervalText = '20'] = process.argv.slice(2);
const interval = Number(intervalText);
if (!input || !metadataPath || !output || !Number.isFinite(interval) || interval <= 0)
  throw Error('Usage: node replay.mjs demo metadata.json output-folder positive-interval');
const metadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
const info = metadata.match_info ?? metadata;
const roster = info.players;
const duration = info.duration_s;
if (!Array.isArray(roster) || roster.length !== 12 || !Number.isFinite(duration))
  throw Error('Replay extraction requires eligible match metadata with exactly 12 players');
const slots = new Map(roster.map(p => [p.player_slot, p]));
if (slots.size !== 12) throw Error('Duplicate or missing metadata player slots');
const accounts = new Map(roster.filter(p => p.account_id != null && String(p.account_id) !== '0')
  .map(p => [String(p.account_id), p.player_slot]));
const stringify = value => JSON.stringify(value, (_, v) => typeof v === 'bigint' ? String(v) :
  ArrayBuffer.isView(v) ? Array.from(v) : v);
mkdirSync(output, {recursive: true});

function writer(name) {
  const fd = openSync(join(output, name), 'w');
  let buffer = '', rows = 0;
  return {
    add(row) { buffer += stringify(row) + '\n'; rows++; if (buffer.length >= 262144) this.flush(); },
    flush() { if (buffer) writeSync(fd, buffer); buffer = ''; },
    close() { this.flush(); closeSync(fd); return rows; }
  };
}
const samples = writer('samples.ndjson'), events = writer('replay_events.ndjson');
const warnings = new Set(['hero_damage_taken is unavailable: no verified cumulative replay field.']);
const summary = {success: false, parser: 'deadem', parser_version: '4.0.1', interval_s: interval,
  client_version: null, replay_build_num: null, replay_patch_version: null, map_name: null,
  final_observed_game_time_s: null, replay_reaches_match_end: null};
const parser = new Parser();

// Source 2 decoder extensions used by the original project with deadem 4.0.1.
parser._engine.registry.registerFieldTypeDecoder('CUtlBinaryBlock', FieldDecoderDescriptor.BINARY_BLOCK);
parser._engine.registry.registerFieldTypeDecoder('CGlobalSymbol', FieldDecoderDescriptor.STRING);
const factory = parser._engine.getDemoPacketHandler()._fieldFactory;
const resolveDecoder = factory._resolveDecoder.bind(factory);
factory._resolveDecoder = (name, definition, instructions) => {
  const decoder = resolveDecoder(name, definition, instructions);
  if (definition.baseType === 'CUtlBinaryBlock') return new decoder.constructor(buffer => {
    const bits = buffer.readUVarInt32() * 8;
    if (bits > buffer.getUnreadCount()) throw Error('Truncated replay binary block');
    return buffer.read(bits, true);
  }, decoder.storage);
  if (instructions.encoder !== 'fixed8') return decoder;
  return new decoder.constructor(buffer => {
    if (buffer.getUnreadCount() < 8) throw Error('Truncated replay fixed8 field');
    const value = buffer.readBitsAsUInt(8);
    return definition.baseType === 'int8' && value >= 128 ? value - 256 : value;
  }, decoder.storage);
};

const statFields = {
  net_worth: 'm_iGoldNetWorth', kills: 'm_iPlayerKills', deaths: 'm_iDeaths', assists: 'm_iPlayerAssists',
  last_hits: 'm_iLastHits', denies: 'm_iDenies', level: 'm_iLevel', alive: 'm_bAlive',
  hero_damage: 'm_iHeroDamage', objective_damage: 'm_iObjectiveDamage', hero_healing: 'm_iHeroHealing',
  self_healing: 'm_iSelfHealing'
};
const nullableState = Object.fromEntries([...Object.keys(statFields), 'unspent_souls', 'health', 'max_health',
  'x', 'y', 'z', 'hero_damage_taken'].map(k => [k, null]));
const finite = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
const field = (entity, name) => entity?.getField(name) ?? null;
const globalField = (controller, name) => field(controller, name) ?? field(controller, `m_PlayerDataGlobal.${name}`);
const tickInterval = () => finite(parser.getDemo().server?.tickInterval);
let demoTick = null, serverTick = null, netTick = null, tickHasEntities = false;
let queuedEvents = [], previous = null, gridIndex = 0, pendingSample = null;
let stopped = false, terminalTime = null, gameOverSeen = false, gameOverThisTick = false, lastClock = null;
const clockMethods = new Set();

function accountFromSteam(value) {
  try {
    const id = BigInt(String(value)) - 76561197960265728n;
    return id >= 0n && id <= 4294967295n ? String(id) : null;
  } catch { return null; }
}

function controllerSlot(controller) {
  if (!controller) return null;
  const slot = field(controller, 'm_unLobbyPlayerSlot');
  const account = accountFromSteam(field(controller, 'm_steamID'));
  const accountSlot = account == null ? null : accounts.get(account);
  if (slots.has(slot)) {
    const expected = slots.get(slot).account_id;
    if (account != null && expected != null && String(expected) !== '0' && String(expected) !== account) {
      warnings.add('A replay controller disagrees with the metadata roster; its state was left missing.');
      return null;
    }
    return slot;
  }
  return accountSlot ?? null;
}

function entitySlot(entity, visited = new Set()) {
  if (!entity || visited.has(entity.handle)) return null;
  visited.add(entity.handle);
  if (entity.class.name === 'CCitadelPlayerController') return controllerSlot(entity);
  for (const name of ['m_hController', 'm_hOwnerEntity']) {
    const handle = field(entity, name);
    if (handle != null) {
      const slot = entitySlot(parser.getDemo().getEntityByHandle(handle), visited);
      if (slot != null) return slot;
    }
  }
  return null;
}
const indexSlot = index => Number.isInteger(index) && index >= 0 ? entitySlot(parser.getDemo().getEntity(index)) : null;
const handleSlot = handle => Number.isInteger(handle) && handle >= 0 && handle !== 16777215 ?
  entitySlot(parser.getDemo().getEntityByHandle(handle)) : null;

function connectionSlot(slot) {
  if (slot == null || slot < 0) return null;
  for (const controller of parser.getDemo().getEntitiesByClassName('CCitadelPlayerController'))
    // Source 2 CBasePlayerController uses connection slot = entity index - 1.
    if (controller.active && controller.index - 1 === slot) return controllerSlot(controller);
  return null;
}

function clockState() {
  const rules = parser.getDemo().getEntitiesByClassName('CCitadelGameRulesProxy')[0];
  const replayMatch = field(rules, 'm_pGameRules.m_unMatchID');
  if (['number', 'bigint', 'string'].includes(typeof replayMatch) && String(replayMatch) !== '0') {
    summary.replay_match_id = String(replayMatch);
    if (summary.replay_match_id !== String(info.match_id)) {
      summary.eligibility_conflict = true;
      throw Error('unverified_eligibility: replay match ID disagrees with metadata');
    }
  }
  summary.replay_game_mode = field(rules, 'm_pGameRules.m_eGameMode');
  summary.replay_match_mode = field(rules, 'm_pGameRules.m_eMatchMode');
  for (const key of ['game_mode', 'match_mode']) {
    const actual = summary[`replay_${key}`], expected = info[key];
    if (Number.isInteger(actual) && actual > 0 && Number.isInteger(expected) && actual !== expected) {
      summary.eligibility_conflict = true;
      throw Error(`unverified_eligibility: replay ${key} disagrees with metadata`);
    }
  }
  const anchor = finite(field(rules, 'm_pGameRules.m_flMatchClockAtLastUpdate'));
  const anchorTick = finite(field(rules, 'm_pGameRules.m_nMatchClockUpdateTick'));
  const start = finite(field(rules, 'm_pGameRules.m_flGameStartTime'));
  const pausedTicks = finite(field(rules, 'm_pGameRules.m_nTotalPausedTicks'));
  const pauseStart = finite(field(rules, 'm_pGameRules.m_nPauseStartTick'));
  const tick = serverTick ?? netTick;
  const secondsPerTick = tickInterval();
  const state = field(rules, 'm_pGameRules.m_eGameState');
  // Valve's tracked EGameState schema is verified from citadel build 6395 onward.
  // Do not apply older third-party examples' conflicting enum numbers across builds.
  const currentStateSchema = summary.client_version >= 6395;
  if (currentStateSchema && state !== 7 && state !== 8) return null;
  const postgame = currentStateSchema && state === 8;
  const paused = field(rules, 'm_pGameRules.m_bGamePaused') === true ||
    field(rules, 'm_pGameRules.m_bServerPaused') === true;
  if (!(start > 0) || tick == null || secondsPerTick == null) return null;
  const effectiveTick = paused && pauseStart > 0 && tick > pauseStart ? pauseStart : tick;
  const legacyTime = pausedTicks == null ? null : (effectiveTick - pausedTicks) * secondsPerTick - start;
  // Never clamp negative pregame time to zero. State enum numbers changed across builds.
  if (legacyTime != null && legacyTime < 0) return null;
  if (anchor != null && anchorTick != null) {
    const time = paused || postgame ? anchor : anchor + Math.max(tick - anchorTick, 0) * secondsPerTick;
    if (time < 0) return null;
    clockMethods.add('match_clock_anchor_and_server_tick');
    return {time, paused, postgame, game_state: finite(state), server_tick: tick};
  }
  if (legacyTime != null) {
    clockMethods.add('game_start_and_pause_adjusted_server_tick');
    return {time: legacyTime, paused, postgame, game_state: finite(state), server_tick: tick};
  }
  return null;
}

function position(pawn) {
  const result = {x: null, y: null, z: null};
  if (!pawn?.active) return result;
  const cells = {}, offsets = {};
  for (const [name, value] of pawn.fieldEntries()) {
    if (!name.startsWith('CBodyComponent.')) continue;
    const axis = name.at(-1);
    if (!'XYZ'.includes(axis)) continue;
    if (name.endsWith(`.m_cell${axis}`)) cells[axis] = finite(value);
    if (name.endsWith(`.m_vec${axis}`)) offsets[axis] = finite(value);
  }
  for (const axis of 'XYZ') if (cells[axis] != null && offsets[axis] != null)
    result[axis.toLowerCase()] = cells[axis] * 512 - 16384 + offsets[axis];
  return result;
}

function snapshot(time, clock) {
  const controllers = new Map();
  for (const controller of parser.getDemo().getEntitiesByClassName('CCitadelPlayerController')) {
    if (!controller.active) continue;
    const slot = controllerSlot(controller);
    if (slot != null) controllers.set(slot, controller);
  }
  const rows = roster.map(player => {
    const identity = {match_id: info.match_id, player_slot: player.player_slot, account_id: player.account_id ?? null,
      hero_id: player.hero_id ?? null, team_id: player.team ?? player.team_id ?? null};
    const controller = controllers.get(player.player_slot);
    if (!controller) return {...identity, observed_time_s: null, ...nullableState};
    const hero = globalField(controller, 'm_nHeroID');
    if (hero != null && hero !== 0 && player.hero_id != null && hero !== player.hero_id) {
      warnings.add('A replay hero disagrees with the metadata roster; its state was left missing.');
      return {...identity, observed_time_s: null, ...nullableState};
    }
    const pawnHandle = field(controller, 'm_hHeroPawn') ?? field(controller, 'm_hPawn');
    const pawn = pawnHandle == null ? null : parser.getDemo().getEntityByHandle(pawnHandle);
    const activePawn = pawn?.active ? pawn : null;
    const values = Object.fromEntries(Object.entries(statFields).map(([column, native]) => {
      const value = globalField(controller, native);
      return [column, column === 'alive' ? (typeof value === 'boolean' ? value : value === 0 ? false : value === 1 ? true : null) : finite(value)];
    }));
    // Valve ECurrencyType: EGold = 0. This is wallet balance, not net worth.
    const currency = field(activePawn, 'm_nCurrencies.0000') ?? field(activePawn, 'm_nCurrencies')?.[0];
    return {...identity, observed_time_s: time, ...values, unspent_souls: finite(currency),
      health: finite(field(activePawn, 'm_iHealth') ?? globalField(controller, 'm_iHealth')),
      max_health: finite(field(activePawn, 'm_iMaxHealth') ?? globalField(controller, 'm_iHealthMax')),
      ...position(activePawn), hero_damage_taken: null};
  });
  return {time, rows, global: {observed_time_s: time, game_state: clock.game_state,
    paused: clock.paused, server_tick: clock.server_tick}};
}

function emitSample(time, observed, isFinal = false) {
  if (pendingSample) for (const row of pendingSample.rows) samples.add(row);
  const identities = roster.map(p => ({match_id: info.match_id, player_slot: p.player_slot,
    account_id: p.account_id ?? null, hero_id: p.hero_id ?? null, team_id: p.team ?? p.team_id ?? null,
    observed_time_s: null, ...nullableState}));
  pendingSample = {time, rows: (observed?.rows ?? identities).map(row => ({...row,
    global: observed?.global ?? null, game_time_s: time,
    is_final_observation: isFinal && row.observed_time_s != null}))};
}

function acceptSnapshot(current) {
  // Delay all-missing spans until another valid observation. They then produce
  // missing grid rows; an unavailable trailing span is not a final observation.
  if (!current.rows.some(row => row.observed_time_s != null)) return;
  if (previous && current.time < previous.time) {
    warnings.add('The replay game clock moved backwards; those observations were skipped.');
    return;
  }
  const maxAge = (tickInterval() ?? 0) * 2.1;
  while (gridIndex * interval <= current.time) {
    const target = gridIndex++ * interval;
    const observed = current.time === target ? current :
      previous && previous.time <= target && target - previous.time <= maxAge ? previous : null;
    if (!observed) warnings.add('Some grid timestamps had no timely prior replay state; values are missing.');
    emitSample(target, observed);
  }
  // Multiple complete network updates can share the same frozen game clock.
  if (pendingSample && pendingSample.time === current.time)
    pendingSample.rows = current.rows.map(row => ({...row, global: current.global,
      game_time_s: pendingSample.time, is_final_observation: false}));
  previous = current;
}

function finishTick() {
  if (stopped) { queuedEvents = []; tickHasEntities = false; return; }
  const clock = clockState();
  if (!clock) { queuedEvents = []; tickHasEntities = false; return; }
  lastClock = clock.time;
  const terminal = gameOverThisTick || clock.postgame;
  // Metadata duration has whole-second precision. Preserve the fractional active
  // ending rather than truncating it to that integer; the first end marker stops us.
  if (clock.time >= 0 && clock.time <= duration + 1) {
    for (const event of queuedEvents) events.add({...event, game_time_s: clock.time});
    if (tickHasEntities && !clock.paused && !terminal) acceptSnapshot(snapshot(clock.time, clock));
  }
  if (terminal) { terminalTime = clock.time; stopped = true; }
  queuedEvents = [];
  tickHasEntities = false;
  gameOverThisTick = false;
}

const selectedEvents = new Set(['k_EUserMsg_HeroKilled', 'k_EUserMsg_BossKilled', 'k_EUserMsg_RejuvStatus',
  'k_EUserMsg_AbilitiesChanged', 'k_EUserMsg_ItemPurchaseNotification']);
parser.registerPreInterceptor(Stage.DEMO_PACKET, packet => {
  if (demoTick != null && packet.tick !== demoTick) {
    finishTick();
    // An event-only tick must use its new net_Tick, never the prior entity tick.
    serverTick = null;
    netTick = null;
  }
  demoTick = packet.tick;
});
parser.registerPostInterceptor(Stage.MESSAGE_PACKET, (packet, message) => {
  const name = message.type.code, data = message.data;
  if (name === 'net_Tick') { netTick = finite(data.tick); return; }
  if (name === 'svc_PacketEntities') {
    serverTick = (Object.hasOwn(data, 'serverTick') ? finite(data.serverTick) : null) ?? netTick;
    tickHasEntities = true;
    return;
  }
  if (name === 'k_EUserMsg_GameOver' && data.justATest !== true) {
    gameOverSeen = true;
    gameOverThisTick = true;
  }
  if (stopped || !selectedEvents.has(name)) return;
  const row = {match_id: info.match_id, event_type: name, tick: packet.tick, network_tick: serverTick ?? netTick,
    data: JSON.parse(stringify(data)), actor_player_slot: null, target_player_slot: null, assister_player_slots: []};
  if (name === 'k_EUserMsg_HeroKilled') {
    row.scorer_player_slot = indexSlot(data.entindexScorer);
    row.attacker_player_slot = indexSlot(data.entindexAttacker);
    row.actor_player_slot = row.scorer_player_slot ?? row.attacker_player_slot;
    row.target_player_slot = indexSlot(data.entindexVictim);
    row.assister_player_slots = (data.entindexAssisters ?? []).map(indexSlot);
  } else if (name === 'k_EUserMsg_BossKilled') {
    row.actor_player_slot = handleSlot(data.entityKiller);
  } else if (name === 'k_EUserMsg_RejuvStatus') {
    row.actor_player_slot = handleSlot(data.playerPawn);
  } else if (name === 'k_EUserMsg_AbilitiesChanged') {
    row.actor_player_slot = connectionSlot(data.purchaserPlayerSlot);
  } else if (name === 'k_EUserMsg_ItemPurchaseNotification') {
    row.actor_player_slot = connectionSlot(data.userid);
  }
  queuedEvents.push(row);
});
parser.registerPostInterceptor(Stage.DEMO_PACKET, packet => {
  if (packet.type.code !== 'DEM_FileHeader') return;
  const header = packet.data;
  summary.client_version = Number(header.gameDirectory?.match(/citadel_v(\d+)/)?.[1]) || null;
  summary.replay_build_num = Object.hasOwn(header, 'buildNum') ? header.buildNum : null;
  summary.replay_patch_version = Object.hasOwn(header, 'patchVersion') ? header.patchVersion : null;
  summary.map_name = header.mapName || null;
});

try {
  await parser.parse(createReadStream(input));
  finishTick();
  if (!previous) throw Error('No player states with a verified active-game clock were observed');
  summary.success = true;
} catch (error) {
  summary.error = String(error.message ?? error);
  process.exitCode = 1;
} finally {
  if (previous) {
    summary.final_observed_game_time_s = previous.time;
    if (pendingSample && pendingSample.time === previous.time) {
      pendingSample.rows = previous.rows.map(row => ({...row, game_time_s: previous.time,
        global: previous.global, is_final_observation: row.observed_time_s != null}));
    } else emitSample(previous.time, previous, true);
  }
  if (pendingSample) for (const row of pendingSample.rows) samples.add(row);
  const endingClock = terminalTime ?? lastClock;
  summary.terminal_game_time_s = terminalTime;
  summary.terminal_duration_difference_s = terminalTime == null ? null : terminalTime - duration;
  summary.replay_reaches_match_end = (terminalTime != null || gameOverSeen) && endingClock != null &&
    Math.abs(endingClock - duration) <= 2 ? true :
    previous && previous.time < duration - 2 ? false : null;
  if (summary.replay_reaches_match_end !== true) warnings.add('Replay completion could not be verified; available observations are retained.');
  summary.tick_interval_s = tickInterval();
  summary.clock_method = [...clockMethods].join(';') || null;
  summary.sample_rows = samples.close();
  summary.event_rows = events.close();
  summary.warnings = [...warnings];
  writeFileSync(join(output, 'summary.json'), stringify(summary));
  await parser.dispose();
}
