import {createReadStream, mkdirSync, openSync, writeSync, closeSync, writeFileSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import {join} from 'node:path';
import {Parser, InterceptorStage as Stage} from 'deadem';
import {FieldDecoderDescriptor} from '@deademx/engine';

const [input, output, stepText = '20'] = process.argv.slice(2);
const step = Number(stepText);
if (!input || !output || !(step > 0)) throw Error('Expected demo, output directory, positive interval');
mkdirSync(output, {recursive: true});
const json = value => JSON.stringify(value, (_, v) => typeof v === 'bigint' ? String(v) :
  ArrayBuffer.isView(v) ? Array.from(v) : v);

// Synchronous parser callbacks need bounded buffers, not an unbounded write queue.
function writer(name) {
  const file = openSync(join(output, `${name}.ndjson.gz`), 'w');
  let buffer = '', rows = 0;
  function flush() {
    if (buffer) writeSync(file, gzipSync(buffer, {level: 9}));
    buffer = '';
  }
  return {
    add(row) { buffer += json(row) + '\n'; rows++; if (buffer.length >= 1024 * 1024) flush(); },
    close() { flush(); closeSync(file); return rows; }
  };
}

const players = writer('player_state'), events = writer('events'), entities = writer('entity_samples');
const parser = new Parser();
// Deadem 4.0.1 omits these Source 2 encodings used by newer replays.
parser._engine.registry.registerFieldTypeDecoder('CUtlBinaryBlock', FieldDecoderDescriptor.BINARY_BLOCK);
parser._engine.registry.registerFieldTypeDecoder('CGlobalSymbol', FieldDecoderDescriptor.STRING);
const factory = parser._engine.getDemoPacketHandler()._fieldFactory;
const resolveDecoder = factory._resolveDecoder.bind(factory);
factory._resolveDecoder = (name, definition, instructions) => {
  const decoder = resolveDecoder(name, definition, instructions);
  // Binary animation/state blobs are bytes; the upstream text decoder is lossy.
  if (definition.baseType === 'CUtlBinaryBlock')
    return new decoder.constructor(buffer => {
      const bits = buffer.readUVarInt32() * 8;
      if (bits > buffer.getUnreadCount()) throw Error('Replay contains a truncated binary block');
      return buffer.read(bits, true);
    }, decoder.storage);
  if (instructions.encoder !== 'fixed8') return decoder;
  return new decoder.constructor(buffer => {
    if (buffer.getUnreadCount() < 8) throw Error('Replay contains a truncated fixed8 field');
    const value = buffer.readBitsAsUInt(8);
    return definition.baseType === 'int8' && value >= 128 ? value - 256 : value;
  }, decoder.storage);
};
const summary = {parser: 'deadem', parser_version: '4.0.1', sample_interval: step,
  decoder_extensions: ['fixed8', 'CUtlBinaryBlock', 'CGlobalSymbol'],
  retrieved_at: new Date().toISOString(), success: false};
const schema = {observed_events: {}};
let rules, netTick, previous, nextTime = 0;

function gameTime() {
  const tickLength = parser.getDemo().server?.tickInterval;
  const start = rules?.getField('m_pGameRules.m_flGameStartTime');
  const pausedTicks = rules?.getField('m_pGameRules.m_nTotalPausedTicks');
  if (netTick == null || !Number.isFinite(tickLength) || !(start > 0) || !Number.isFinite(pausedTicks)) return null;
  const pauseStart = rules.getField('m_pGameRules.m_nPauseStartTick');
  const tick = rules.getField('m_pGameRules.m_bGamePaused') && pauseStart > 0 ? pauseStart : netTick;
  return (tick - pausedTicks) * tickLength - start;
}

function position(fields, axis) {
  const cell = Object.keys(fields).find(k => k.startsWith('CBodyComponent.') && k.endsWith(`.m_cell${axis}`));
  const offset = Object.keys(fields).find(k => k.startsWith('CBodyComponent.') && k.endsWith(`.m_vec${axis}`));
  return Number.isFinite(fields[cell]) && Number.isFinite(fields[offset]) ? fields[cell] * 512 - 16384 + fields[offset] : null;
}

function playerState(controller, time, tick) {
  const demo = parser.getDemo(), c = controller.unpackFlattened();
  const pawnHandle = controller.getField('m_hHeroPawn') ?? controller.getField('m_hPawn');
  const pawn = pawnHandle == null ? null : demo.getEntityByHandle(pawnHandle);
  const p = pawn?.unpackFlattened() ?? {};
  const global = key => controller.getField(key) ?? controller.getField(`m_PlayerDataGlobal.${key}`) ?? null;
  const account = c.m_steamID == null ? null : BigInt(String(c.m_steamID)) - 76561197960265728n;
  const handles = pawn?.getField('m_vecAbilities') ?? pawn?.getField('m_CCitadelAbilityComponent.m_vecAbilities');
  return {
    source_game_time: time, tick, network_tick: netTick,
    player_id: account !== null && account >= 0n && account <= 4294967295n ? Number(account) : null,
    steam_id: c.m_steamID == null ? null : String(c.m_steamID), player_slot: c.m_unLobbyPlayerSlot ?? null,
    controller_index: controller.index, pawn_index: pawn?.index ?? null,
    hero_id: global('m_nHeroID') ?? p.m_nHeroID ?? null, team_engine: c.m_iTeamNum ?? p.m_iTeamNum ?? null,
    net_worth: global('m_iGoldNetWorth'), kills: global('m_iPlayerKills'),
    deaths: global('m_iDeaths'), assists: global('m_iPlayerAssists'),
    health: p.m_iHealth ?? global('m_iHealth'), max_health: p.m_iMaxHealth ?? global('m_iHealthMax'),
    x: position(p, 'X'), y: position(p, 'Y'), z: position(p, 'Z'),
    inventory: controller.getField('m_vecUpgrades') ?? null,
    ability_upgrade_states: controller.getField('m_vecAbilityUpgradeState') ?? null,
    abilities: Array.isArray(handles) ? handles.map(handle => {
      const entity = demo.getEntityByHandle(handle);
      return entity ? {handle, entity_class: entity.class.name, fields: entity.unpackFlattened()} : {handle};
    }) : null,
    raw_controller: c, raw_pawn: p
  };
}

function snapshot(time, tick) {
  const state = {time, players: [], entities: []};
  for (const entity of parser.getDemo().getEntities()) {
    if (!entity.active) continue;
    state.entities.push({source_game_time: time, tick, entity_index: entity.index,
      entity_handle: entity.handle, entity_class: entity.class.name, fields: entity.unpackFlattened()});
    if (entity.class.name === 'CCitadelPlayerController') state.players.push(playerState(entity, time, tick));
  }
  // Detach arrays from mutable parser state before the next packet.
  return JSON.parse(json(state));
}

parser.registerPostInterceptor(Stage.ENTITY_PACKET, (_, __, changes) => {
  for (const {entity} of changes) if (entity.class.name === 'CCitadelGameRulesProxy') rules = entity;
});

parser.registerPostInterceptor(Stage.MESSAGE_PACKET, (packet, message) => {
  const name = message.type.code, data = message.data;
  if (name === 'net_Tick') { netTick = data.tick; return; }
  if (name === 'svc_ServerInfo') summary.server_info = data;
  if (/^(svc_PacketEntities|svc_CreateStringTable|svc_UpdateStringTable|svc_UserCmds)$/.test(name)) return;
  if (name === 'GE_Source1LegacyGameEventList')
    schema.observed_events[name] = {fields: {descriptors: {example: data.descriptors}}};
  events.add({tick: packet.tick, network_tick: netTick, game_time: gameTime(),
    event_type: name, message_id: message.type.id, raw_event_data: data});
});

parser.registerPostInterceptor(Stage.DEMO_PACKET, packet => {
  if (packet.type.code === 'DEM_FileHeader') {
    summary.header = packet.data;
    summary.game_version = Number(packet.data.gameDirectory?.match(/citadel_v(\d+)/)?.[1]) || null;
  }
  if (packet.type.code === 'DEM_FileInfo') summary.file_info = packet.data;
  const time = gameTime();
  if (time === null || time < 0 || !Number.isFinite(time)) return;
  summary.first_game_time ??= time;
  summary.last_game_time = time;
  if (time >= nextTime) {
    // Use the last complete packet at/before the boundary, never a future snapshot.
    const snap = previous && previous.time <= nextTime ? previous : snapshot(time, packet.tick);
    const sampleTime = snap === previous ? nextTime : time;
    const timing = {game_time: sampleTime, sample_age_seconds: sampleTime - snap.time};
    for (const row of snap.players) players.add({...timing, ...row});
    for (const row of snap.entities) entities.add({...timing, ...row});
    nextTime = (Math.floor(time / step) + 1) * step;
    previous = null;
  }
  const tickLength = parser.getDemo().server?.tickInterval ?? 0;
  if (time >= nextTime - 2.1 * tickLength && (!previous || time > previous.time + 0.000001))
    previous = snapshot(time, packet.tick);
});

try {
  await parser.parse(createReadStream(input));
  summary.success = true;
} catch (error) {
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.tick_interval = parser.getDemo().server?.tickInterval ?? null;
  summary.player_state = players.close();
  summary.events = events.close();
  summary.entity_samples = entities.close();
  writeFileSync(join(output, 'schema.json'), json(schema));
  writeFileSync(join(output, 'summary.json'), json(summary));
  await parser.dispose();
}
