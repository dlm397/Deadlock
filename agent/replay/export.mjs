// Small bridge around the existing deadem parser, NOT a replay decoder.
// Called by Python: node replay/export.mjs input.dem output_directory interval_seconds
import { createReadStream, mkdirSync, openSync, closeSync, writeSync, writeFileSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { join } from 'node:path';
import { Parser, InterceptorStage } from 'deadem';

const [input, output, intervalText = '10'] = process.argv.slice(2);
const interval = Number(intervalText);
if (!input || !output || !(interval > 0)) throw new Error('Expected demo, output directory, positive interval');
mkdirSync(output, { recursive: true });
const retrievedAt = new Date().toISOString();

function json(value) {
  return JSON.stringify(value, (_, v) => {
    if (typeof v === 'bigint') return v.toString();
    if (ArrayBuffer.isView(v)) return Array.from(v);
    return v;
  });
}
// Bounded buffers and independent gzip members avoid an unbounded stream queue
// inside deadem's synchronous interceptor callbacks. Python gzip reads all members.
function writer(name) {
  const fd = openSync(join(output, `${name}.ndjson.gz`), 'w');
  let buffer = '', rows = 0;
  const flush = () => { if (buffer) { writeSync(fd, gzipSync(buffer, { level: 9 })); buffer = ''; } };
  return {
    add(row) { buffer += json(row) + '\n'; rows++; if (buffer.length >= 1024 * 1024) flush(); },
    finish() { flush(); closeSync(fd); return rows; },
  };
}
const eventOut = writer('events'), entityOut = writer('entity_samples'), playerOut = writer('player_state');
const schema = { parser: 'deadem', parser_version: '4.0.1', retrieved_at: retrievedAt,
  declared_entity_classes: {}, observed_entity_classes: {}, observed_events: {},
  observations: 'Observed properties are fields carried by entity updates; declared classes do not imply occurrence.',
  reconstructed: { game_time: '(network tick - completed paused ticks) * server tick interval - game start time; frozen at pause start during an active pause',
    player_id: 'SteamID64 minus 76561197960265728, when SteamID is present',
    position: 'Source 2 Deadlock cell coordinate: cell * 512 - 16384 + offset',
    state_sampling: 'Last complete replay state at or immediately before the requested game-time grid; source_game_time and sample_age_seconds retained' },
};
const summary = { parser: 'deadem', parser_version: '4.0.1', input, retrieved_at: retrievedAt,
  exporter_sha256: createHash('sha256').update(readFileSync(new URL(import.meta.url))).digest('hex'),
  sample_interval: interval, success: false, header: null, file_info: null, server_info: null,
  first_game_time: null, last_game_time: null, warnings: [] };
const parser = new Parser();
let netTick = null, rules = null, nextSample = 0, previousSnapshot = null, lastTick = null;
const seenEvents = new Map();
let progressAt = Date.now();
parser.registerPreInterceptor(InterceptorStage.DEMO_PACKET, packet => {
  if (Date.now() - progressAt > 10000) {
    console.log(`Parsing ${packet.type.code} tick=${packet.tick}`);
    progressAt = Date.now();
  }
});
const cleanType = v => v === null ? 'null' : ArrayBuffer.isView(v) ? v.constructor.name : Array.isArray(v) ? 'array' : typeof v;

function auditFields(destination, values) {
  for (const [key, value] of Object.entries(values ?? {})) {
    const field = destination[key] ??= { types: [], count: 0, example: value };
    field.count++;
    const type = cleanType(value);
    if (!field.types.includes(type)) field.types.push(type);
  }
}
function gameTime() {
  const tickInterval = parser.getDemo().server?.tickInterval;
  const start = rules?.getField('m_pGameRules.m_flGameStartTime');
  const paused = rules?.getField('m_pGameRules.m_nTotalPausedTicks');
  if (netTick === null || !Number.isFinite(tickInterval) || !(start > 0) || !Number.isFinite(paused)) return null;
  const isPaused = rules.getField('m_pGameRules.m_bGamePaused');
  const pauseStart = rules.getField('m_pGameRules.m_nPauseStartTick');
  const clockTick = isPaused && Number.isFinite(pauseStart) && pauseStart > 0 ? pauseStart : netTick;
  return (clockTick - paused) * tickInterval - start;
}
function steamAccount(v) {
  if (v === null || v === undefined) return null;
  try { const id = BigInt(String(v)) - 76561197960265728n; return id >= 0n && id <= 4294967295n ? Number(id) : null; }
  catch { return null; }
}
function coordinate(fields, axis) {
  const cellName = Object.keys(fields).find(k => k.startsWith('CBodyComponent.') && k.endsWith(`.m_cell${axis}`));
  const offsetName = Object.keys(fields).find(k => k.startsWith('CBodyComponent.') && k.endsWith(`.m_vec${axis}`));
  if (!cellName || !offsetName) return null;
  const cell = fields[cellName], offset = fields[offsetName];
  return Number.isFinite(cell) && Number.isFinite(offset) ? cell * 512 - 16384 + offset : null;
}
function controllerState(controller, time, tick) {
  const demo = parser.getDemo(), c = controller.unpackFlattened();
  const handle = controller.getField('m_hHeroPawn') ?? controller.getField('m_hPawn');
  const pawn = handle === undefined ? null : demo.getEntityByHandle(handle);
  const p = pawn?.unpackFlattened() ?? {};
  const global = key => controller.getField(key) ?? controller.getField(`m_PlayerDataGlobal.${key}`) ?? null;
  const abilityHandles = pawn?.getField('m_vecAbilities') ?? pawn?.getField('m_CCitadelAbilityComponent.m_vecAbilities');
  const abilities = Array.isArray(abilityHandles) ? abilityHandles.map(h => {
    const entity = demo.getEntityByHandle(h);
    return entity ? { handle: h, entity_class: entity.class.name, fields: entity.unpackFlattened() } : { handle: h };
  }) : null;
  return { source_game_time: time, tick, network_tick: netTick, player_id: steamAccount(c.m_steamID),
    steam_id: c.m_steamID == null ? null : String(c.m_steamID), player_slot: c.m_unLobbyPlayerSlot ?? null,
    controller_index: controller.index, pawn_index: pawn?.index ?? null,
    hero_id: global('m_nHeroID') ?? p.m_nHeroID ?? null, team_engine: c.m_iTeamNum ?? p.m_iTeamNum ?? null,
    net_worth: global('m_iGoldNetWorth'), kills: global('m_iPlayerKills'),
    deaths: global('m_iDeaths'), assists: global('m_iPlayerAssists'),
    health: p.m_iHealth ?? global('m_iHealth'), max_health: p.m_iMaxHealth ?? global('m_iHealthMax'),
    x: coordinate(p, 'X'), y: coordinate(p, 'Y'), z: coordinate(p, 'Z'),
    inventory: controller.getField('m_vecUpgrades') ?? null,
    ability_upgrade_states: controller.getField('m_vecAbilityUpgradeState') ?? null,
    abilities, raw_controller: c, raw_pawn: p };
}
function snapshot(time, tick) {
  const entities = [], players = [];
  for (const entity of parser.getDemo().getEntities()) {
    if (!entity.active) continue;
    entities.push({ source_game_time: time, tick, entity_index: entity.index, entity_handle: entity.handle,
      entity_class: entity.class.name, fields: entity.unpackFlattened() });
    if (entity.class.name === 'CCitadelPlayerController') players.push(controllerState(entity, time, tick));
  }
  // Snapshot contains live references, so copy now before the parser changes state.
  return JSON.parse(json({ time, entities, players }));
}
function writeSnapshot(snap, sampleTime) {
  const common = { game_time: sampleTime, sample_age_seconds: sampleTime - snap.time };
  for (const row of snap.entities) entityOut.add({ ...common, ...row });
  for (const row of snap.players) playerOut.add({ ...common, ...row });
}
parser.registerPostInterceptor(InterceptorStage.ENTITY_PACKET, (packet, _, changes) => {
  for (const change of changes) {
    const entity = change.entity, name = entity.class.name;
    if (name === 'CCitadelGameRulesProxy') rules = entity;
    const inventory = schema.observed_entity_classes[name] ??= { entity_handles: [], operations: {}, properties: {} };
    if (!inventory.entity_handles.includes(entity.handle)) inventory.entity_handles.push(entity.handle);
    inventory.operations[change.operation.code] = (inventory.operations[change.operation.code] ?? 0) + 1;
    auditFields(inventory.properties, change.getChanges());
  }
});
parser.registerPostInterceptor(InterceptorStage.MESSAGE_PACKET, (packet, message) => {
  const name = message.type.code, data = message.data;
  if (name === 'net_Tick') { netTick = data.tick; return; }
  if (name === 'svc_ServerInfo') summary.server_info = data;
  // Transport messages are represented by parsed entities or original demo bytes.
  if (/^(net_Tick|svc_PacketEntities|svc_CreateStringTable|svc_UpdateStringTable|svc_UserCmds)$/.test(name)) return;
  const inventory = schema.observed_events[name] ??= { message_id: message.type.id, count: 0, fields: {} };
  inventory.count++;
  auditFields(inventory.fields, data);
  eventOut.add({ tick: packet.tick, network_tick: netTick, game_time: gameTime(), event_type: name,
    message_id: message.type.id, raw_event_data: data });
});
parser.registerPostInterceptor(InterceptorStage.DEMO_PACKET, packet => {
  lastTick = packet.tick;
  if (packet.type.code === 'DEM_FileHeader') {
    summary.header = packet.data;
    summary.game_version = Number(packet.data.gameDirectory?.match(/citadel_v(\d+)/)?.[1]) || null;
    console.log('Header', json(packet.data));
  }
  if (packet.type.code === 'DEM_FileInfo') summary.file_info = packet.data;
  const time = gameTime();
  if (time === null || time < 0 || !Number.isFinite(time)) return;
  summary.first_game_time ??= time;
  summary.last_game_time = time;
  // Capture only near a grid boundary. When a boundary is crossed, use the previous
  // complete packet snapshot, never a future state or linear interpolation.
  if (time >= nextSample) {
    if (previousSnapshot && previousSnapshot.time <= nextSample) writeSnapshot(previousSnapshot, nextSample);
    else writeSnapshot(snapshot(time, packet.tick), time);
    nextSample = (Math.floor(time / interval) + 1) * interval;
    previousSnapshot = null;
  }
  const tickInterval = parser.getDemo().server?.tickInterval ?? 0;
  if (time >= nextSample - 2.1 * tickInterval && (!previousSnapshot || time > previousSnapshot.time + 0.000001))
    previousSnapshot = snapshot(time, packet.tick);
});

try {
  await parser.parse(createReadStream(input));
  summary.success = true;
} catch (error) {
  summary.error = error.stack ?? String(error);
  process.exitCode = 1;
} finally {
  const demo = parser.getDemo();
  for (const clazz of demo.getClasses()) schema.declared_entity_classes[clazz.name] = {
    class_id: clazz.id, fields: clazz.serializer.fields.map(f => ({ name: f.name, send_node: f.sendNode, type: f.definition.toString() })) };
  summary.tick_interval = demo.server?.tickInterval ?? null;
  summary.last_tick = lastTick;
  summary.events = eventOut.finish();
  summary.entity_samples = entityOut.finish();
  summary.player_state = playerOut.finish();
  summary.entity_classes_observed = Object.keys(schema.observed_entity_classes).length;
  summary.event_names_observed = Object.keys(schema.observed_events).length;
  for (const [name, pattern] of Object.entries({player_properties: /Player/, team_properties: /Team/, objective_properties: /Boss|Guardian|Walker|Patron|Idol|Urn|Tower|Rejuv/i, item_inventory_properties: /Ability|Upgrade|Item|Weapon/})) {
    schema[name] = Object.fromEntries(Object.entries(schema.observed_entity_classes).filter(([k]) => pattern.test(k)));
  }
  writeFileSync(join(output, 'schema.json'), json(schema));
  writeFileSync(join(output, 'summary.json'), json(summary));
  await parser.dispose();
  console.log(json(summary));
}
