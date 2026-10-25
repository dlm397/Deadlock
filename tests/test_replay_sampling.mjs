// Bounded offline checks of the actual parser script using synthetic engine updates.
// No replay files, HTTP calls, or third-party parser internals are executed.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {basename, join} from 'node:path';

const source = readFileSync(new URL('../app/replay.mjs', import.meta.url), 'utf8')
  .replace(/^import .*;\r?\n/gm, '');
const players = Array.from({length: 12}, (_, slot) => ({player_slot: slot,
  account_id: slot === 10 ? 0 : 1000 + slot, hero_id: slot + 1, team: slot < 6 ? 0 : 1})).reverse();
const payload = {match_info: {match_id: '123', duration_s: 901, game_mode: 1, match_mode: 1, players}};
const output = new Map();
const registrations = new Map();
let current = {}, parserInstance;

class Decoder {
  constructor(decode, storage) { this.decode = decode; this.storage = storage; }
}
class FakeParser {
  constructor() {
    parserInstance = this;
    this.pre = new Map(); this.post = new Map();
    this.factory = {_resolveDecoder: () => new Decoder(() => null, 'storage')};
    this._engine = {
      registry: {registerFieldTypeDecoder: (name, decoder) => registrations.set(name, decoder)},
      getDemoPacketHandler: () => ({_fieldFactory: this.factory}),
    };
    const rules = {getField: key => ({
      'm_pGameRules.m_unMatchID': '123', 'm_pGameRules.m_eGameMode': 1,
      'm_pGameRules.m_eMatchMode': 1, 'm_pGameRules.m_flMatchClockAtLastUpdate': current.time,
      'm_pGameRules.m_nMatchClockUpdateTick': current.tick,
      'm_pGameRules.m_flGameStartTime': 1000, 'm_pGameRules.m_nTotalPausedTicks': 0,
      'm_pGameRules.m_nPauseStartTick': current.paused ? current.tick : 0,
      'm_pGameRules.m_eGameState': current.state ?? 7,
      'm_pGameRules.m_bGamePaused': current.paused === true,
    })[key]};
    this.controllers = players.filter(p => p.player_slot !== 11).map(p => ({
      active: true, index: p.player_slot + 1, handle: p.player_slot + 100,
      class: {name: 'CCitadelPlayerController'},
      getField: key => ({
        m_unLobbyPlayerSlot: p.player_slot,
        m_steamID: 76561197960265728n + BigInt(p.player_slot === 9 ? 9000 : 1000 + p.player_slot),
        m_nHeroID: p.hero_id, m_iGoldNetWorth: current.value + p.player_slot,
        m_iPlayerKills: 0, m_bAlive: true,
      })[key],
    }));
    this.demo = {
      server: {tickInterval: 1 / 64},
      getEntitiesByClassName: name => name === 'CCitadelGameRulesProxy' ? [rules] : this.controllers,
      getEntity: index => this.controllers.find(c => c.index === index),
      getEntityByHandle: handle => this.controllers.find(c => c.handle === handle),
    };
  }
  getDemo() { return this.demo; }
  registerPreInterceptor(stage, callback) { this.pre.set(stage, callback); }
  registerPostInterceptor(stage, callback) { this.post.set(stage, callback); }
  async dispose() {}
  async parse() {
    this.post.get('demo')({type: {code: 'DEM_FileHeader'}, data: {gameDirectory: 'citadel_v6774'}});
    let frameIndex = 0;
    const frame = (time, value, options = {}) => {
      const packet = {tick: ++frameIndex, type: {code: 'DEM_Packet'}};
      this.pre.get('demo')(packet);
      current = {time, value, tick: 64000 + Math.round(time * 64), ...options};
      const message = (code, data) => this.post.get('message')(packet, {type: {code}, data});
      if (!options.noNetworkTick) message('net_Tick', {tick: current.tick});
      if (!options.noEntities) message('svc_PacketEntities', {serverTick: current.tick});
      for (const [code, data] of options.events ?? []) message(code, data);
    };
    frame(-1, 999, {state: 6});
    frame(0, 0);
    frame(19.984375, 10);
    frame(20.015625, 20);
    frame(20.015625, 999, {paused: true});
    frame(39.984375, 30);
    frame(40.015625, 40);
    // Without a new tick this event has no verified clock; it must not inherit 40.015625.
    frame(40.03125, 50, {noNetworkTick: true, noEntities: true,
      events: [['k_EUserMsg_ItemPurchaseNotification', {userid: 3, abilityId: 42}]]});
    frame(40.046875, 60, {events: [
      ['k_EUserMsg_ItemPurchaseNotification', {userid: 3, abilityId: 43}],
      ['k_EUserMsg_HeroKilled', {entindexScorer: 4, entindexAttacker: 5,
        entindexVictim: 6, entindexAssisters: [7]}],
    ]});
    frame(40.03125, 998); // Clock regression is ignored.
    frame(60.5, 70); // The 60-second grid has no timely prior observation.
    frame(61, 997, {state: 8, events: [['k_EUserMsg_GameOver', {}]]});
  }
}

const names = ['createReadStream', 'readFileSync', 'mkdirSync', 'openSync', 'writeSync',
  'closeSync', 'writeFileSync', 'join', 'Parser', 'Stage', 'FieldDecoderDescriptor', 'process'];
const args = [() => null, () => JSON.stringify(payload), () => {},
  path => { const name = basename(path); output.set(name, ''); return name; },
  (fd, data) => output.set(fd, output.get(fd) + data), () => {},
  (path, data) => output.set(basename(path), data), join, FakeParser,
  {DEMO_PACKET: 'demo', MESSAGE_PACKET: 'message'}, {BINARY_BLOCK: 'binary', STRING: 'string'},
  {argv: ['node', 'replay.mjs', 'synthetic.dem', 'metadata.json', 'output', '20'], exitCode: 0}];
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
await new AsyncFunction(...names, source)(...args);
assert.equal(args.at(-1).exitCode, 0);
const rows = output.get('samples.ndjson').trim().split('\n').map(JSON.parse);
const events = output.get('replay_events.ndjson').trim().split('\n').map(JSON.parse);
const summary = JSON.parse(output.get('summary.json'));
assert.deepEqual([...new Set(rows.map(row => row.game_time_s))], [0, 20, 40, 60, 60.5]);
assert.equal(rows.length, 5 * 12);
for (const time of [0, 20, 40, 60, 60.5]) {
  const snapshot = rows.filter(row => row.game_time_s === time);
  assert.equal(new Set(snapshot.map(row => row.player_slot)).size, 12);
  for (const row of snapshot) {
    assert(row.observed_time_s == null || row.observed_time_s <= time);
    assert(row.global == null || row.global.observed_time_s <= time);
  }
}
const at = (time, slot) => rows.find(row => row.game_time_s === time && row.player_slot === slot);
assert.equal(at(0, 0).net_worth, 0); // Measured zero stays zero.
assert.equal(at(0, 0).health, null); // Unavailable measurement stays missing.
assert.equal(at(20, 0).net_worth, 10); // 20.015625 cannot backfill target 20.
assert.equal(at(20, 0).observed_time_s, 19.984375);
assert.equal(at(40, 3).net_worth, 33); // Slot alignment is independent of roster order.
assert.equal(at(40, 9).observed_time_s, null); // Conflicting account is excluded.
assert.equal(at(40, 10).net_worth, 40); // Metadata's zero account is unknown, not a mismatch.
assert.equal(at(40, 11).observed_time_s, null); // Missing controller is not fabricated.
assert.equal(at(60, 0).observed_time_s, null);
assert.equal(at(60, 0).global, null);
assert.equal(at(60.5, 0).is_final_observation, true);
assert.equal(at(60.5, 0).net_worth, 70); // Postgame values were not used.
assert.equal(events.length, 2);
assert.equal(events[0].game_time_s, 40.046875);
assert.equal(events[0].actor_player_slot, 3);
assert.equal(events[1].scorer_player_slot, 3);
assert.equal(events[1].attacker_player_slot, 4);
assert.equal(events[1].target_player_slot, 5);
assert.deepEqual(events[1].assister_player_slots, [6]);
assert.equal(summary.final_observed_game_time_s, 60.5);
assert.equal(summary.terminal_game_time_s, 61);
assert.equal(summary.replay_reaches_match_end, false);
assert(summary.warnings.some(value => value.includes('backwards')));

// Preserve and exercise the existing decoder fixes, including truncated input guards.
assert.equal(registrations.get('CUtlBinaryBlock'), 'binary');
assert.equal(registrations.get('CGlobalSymbol'), 'string');
const fixed = parserInstance.factory._resolveDecoder('x', {baseType: 'int8'}, {encoder: 'fixed8'});
assert.equal(fixed.decode({getUnreadCount: () => 8, readBitsAsUInt: () => 255}), -1);
assert.throws(() => fixed.decode({getUnreadCount: () => 7}), /Truncated replay fixed8/);
const binary = parserInstance.factory._resolveDecoder('x', {baseType: 'CUtlBinaryBlock'}, {});
assert.equal(binary.decode({readUVarInt32: () => 2, getUnreadCount: () => 16,
  read: (bits, copy) => `${bits}:${copy}`}), '16:true');
assert.throws(() => binary.decode({readUVarInt32: () => 2, getUnreadCount: () => 15}), /Truncated replay binary/);
console.log('Replay timing, identity, missingness, events, and decoder checks passed (synthetic, offline).');
