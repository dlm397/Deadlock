-- Full-match schema v1. IDs retain integer precision; unknown observations are NULL.
-- Logical references are checked by the writer before an atomic match replacement.
CREATE TABLE IF NOT EXISTS schema_info (
    version UINTEGER PRIMARY KEY, created_at_utc TIMESTAMPTZ DEFAULT current_timestamp
);
INSERT INTO schema_info(version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);

CREATE TABLE IF NOT EXISTS replay_sources (
    source_id UUID PRIMARY KEY,
    match_id UBIGINT NOT NULL,
    path VARCHAR NOT NULL,
    size_bytes UBIGINT,
    mtime_ns UBIGINT,
    compression VARCHAR NOT NULL,
    registered_at_utc TIMESTAMPTZ NOT NULL,
    UNIQUE(path, size_bytes, mtime_ns)
);
CREATE TABLE IF NOT EXISTS processing_attempts (
    attempt_id UUID PRIMARY KEY,
    match_id UBIGINT NOT NULL,
    source_id UUID NOT NULL,
    status VARCHAR NOT NULL,
    stage VARCHAR,
    reason VARCHAR,
    started_at_utc TIMESTAMPTZ NOT NULL,
    finished_at_utc TIMESTAMPTZ,
    sampling_interval_s DOUBLE NOT NULL CHECK(sampling_interval_s > 0),
    parser_name VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    parser_revision VARCHAR NOT NULL,
    schema_version UINTEGER NOT NULL,
    metadata_source VARCHAR,
    metadata_fetched_at_utc TIMESTAMPTZ,
    metadata_sha256 VARCHAR,
    definitions_sha256 VARCHAR,
    definitions_available BOOLEAN,
    clock_method VARCHAR,
    tick_interval_s DOUBLE,
    replay_match_id UBIGINT,
    replay_game_mode INTEGER,
    replay_match_mode INTEGER,
    final_observed_game_time_s DOUBLE,
    terminal_game_time_s DOUBLE,
    terminal_duration_difference_s DOUBLE,
    replay_reaches_match_end BOOLEAN,
    warnings VARCHAR[],
    CHECK(status IN ('running','complete','failed','ineligible','unverified_eligibility','interrupted'))
);
CREATE TABLE IF NOT EXISTS matches (
    match_id UBIGINT PRIMARY KEY,
    successful_attempt_id UUID NOT NULL,
    start_time_utc TIMESTAMPTZ,
    duration_s UINTEGER NOT NULL,
    player_count UINTEGER NOT NULL,
    game_mode VARCHAR NOT NULL,
    match_mode VARCHAR NOT NULL,
    match_outcome VARCHAR,
    winning_team_id UTINYINT,
    client_version UINTEGER,
    replay_build_num UINTEGER,
    game_mode_version UINTEGER,
    map_name VARCHAR,
    replay_patch_version UINTEGER
);
CREATE TABLE IF NOT EXISTS match_players (
    match_id UBIGINT NOT NULL,
    player_slot UINTEGER NOT NULL,
    account_id UINTEGER,
    hero_id UINTEGER,
    team_id UTINYINT NOT NULL CHECK(team_id IN (0,1)),
    won BOOLEAN,
    initial_rank_badge UINTEGER,
    assigned_lane UINTEGER,
    pregame_hero_id UINTEGER,
    hero_build_id UBIGINT,
    kills UBIGINT, deaths UBIGINT, assists UBIGINT,
    net_worth UBIGINT, level UBIGINT, last_hits UBIGINT, denies UBIGINT,
    hero_damage UBIGINT, objective_damage UBIGINT, hero_healing UBIGINT,
    self_healing UBIGINT, hero_damage_taken UBIGINT,
    combat_stats_time_s DOUBLE, combat_stats_at_end BOOLEAN,
    PRIMARY KEY(match_id, player_slot)
);
CREATE TABLE IF NOT EXISTS match_bans (
    match_id UBIGINT NOT NULL, hero_id UINTEGER NOT NULL,
    PRIMARY KEY(match_id, hero_id)
);
CREATE TABLE IF NOT EXISTS player_snapshots (
    match_id UBIGINT NOT NULL,
    player_slot UINTEGER NOT NULL,
    game_time_s DOUBLE NOT NULL CHECK(game_time_s >= 0),
    observed_time_s DOUBLE,
    is_final_observation BOOLEAN NOT NULL,
    net_worth UBIGINT, unspent_souls UBIGINT,
    kills UBIGINT, deaths UBIGINT, assists UBIGINT,
    last_hits UBIGINT, denies UBIGINT, level UBIGINT,
    alive BOOLEAN, health BIGINT, max_health BIGINT,
    x DOUBLE, y DOUBLE, z DOUBLE,
    hero_damage UBIGINT, objective_damage UBIGINT,
    hero_healing UBIGINT, self_healing UBIGINT,
    PRIMARY KEY(match_id, player_slot, game_time_s),
    CHECK(observed_time_s IS NULL OR (observed_time_s >= 0 AND observed_time_s <= game_time_s))
);
CREATE TABLE IF NOT EXISTS global_snapshots (
    match_id UBIGINT NOT NULL,
    game_time_s DOUBLE NOT NULL,
    observed_time_s DOUBLE,
    is_final_observation BOOLEAN NOT NULL,
    game_state INTEGER,
    paused BOOLEAN,
    server_tick BIGINT,
    PRIMARY KEY(match_id, game_time_s),
    CHECK(observed_time_s IS NULL OR (observed_time_s >= 0 AND observed_time_s <= game_time_s))
);
CREATE TABLE IF NOT EXISTS hero_kills (
    match_id UBIGINT NOT NULL, event_id VARCHAR NOT NULL,
    event_time_s DOUBLE, event_source VARCHAR NOT NULL,
    victim_player_slot UINTEGER, killer_player_slot UINTEGER,
    scorer_player_slot UINTEGER, attacker_player_slot UINTEGER,
    demo_tick BIGINT, network_tick BIGINT,
    death_duration_s DOUBLE, time_to_kill_s DOUBLE,
    victim_x DOUBLE, victim_y DOUBLE, victim_z DOUBLE,
    killer_x DOUBLE, killer_y DOUBLE, killer_z DOUBLE,
    PRIMARY KEY(match_id,event_id)
);
CREATE TABLE IF NOT EXISTS kill_assists (
    match_id UBIGINT NOT NULL, event_id VARCHAR NOT NULL, player_slot UINTEGER NOT NULL,
    PRIMARY KEY(match_id,event_id,player_slot)
);
CREATE TABLE IF NOT EXISTS item_events (
    match_id UBIGINT NOT NULL, event_id VARCHAR NOT NULL,
    event_time_s DOUBLE, event_type VARCHAR NOT NULL, event_source VARCHAR NOT NULL,
    player_slot UINTEGER, item_id UINTEGER, ability_id UINTEGER,
    upgrade_id UINTEGER, upgrade_info UINTEGER, imbued_ability_id UINTEGER,
    PRIMARY KEY(match_id,event_id)
);
CREATE TABLE IF NOT EXISTS objective_events (
    match_id UBIGINT NOT NULL, event_id VARCHAR NOT NULL,
    event_time_s DOUBLE, event_type VARCHAR NOT NULL, event_source VARCHAR NOT NULL,
    objective_id UINTEGER, objective_team_id UTINYINT,
    PRIMARY KEY(match_id,event_id)
);
CREATE TABLE IF NOT EXISTS midboss_events (
    match_id UBIGINT NOT NULL, event_id VARCHAR NOT NULL,
    event_time_s DOUBLE, event_type VARCHAR NOT NULL, event_source VARCHAR NOT NULL,
    team_id UTINYINT, related_event_id VARCHAR,
    PRIMARY KEY(match_id,event_id)
);
-- Version-dependent protocol codes are deliberately preserved without guessed labels.
CREATE TABLE IF NOT EXISTS boss_notifications (
    match_id UBIGINT NOT NULL, event_id VARCHAR NOT NULL,
    event_time_s DOUBLE NOT NULL, demo_tick BIGINT, network_tick BIGINT,
    actor_player_slot UINTEGER, objective_team_code INTEGER,
    objective_mask_change INTEGER, entity_killed_class INTEGER, bosses_remaining INTEGER,
    x DOUBLE, y DOUBLE, z DOUBLE,
    PRIMARY KEY(match_id,event_id)
);
CREATE TABLE IF NOT EXISTS rejuv_events (
    match_id UBIGINT NOT NULL, event_id VARCHAR NOT NULL,
    event_time_s DOUBLE NOT NULL, demo_tick BIGINT, network_tick BIGINT,
    actor_player_slot UINTEGER, killing_team_code INTEGER, user_team_code INTEGER,
    event_type_code INTEGER,
    PRIMARY KEY(match_id,event_id)
);
CREATE TABLE IF NOT EXISTS item_definitions (
    client_version UINTEGER NOT NULL, item_id UINTEGER NOT NULL,
    name VARCHAR, kind VARCHAR,
    PRIMARY KEY(client_version,item_id)
);

-- Derived aggregates avoid storing a second representation of player statistics.
-- Every statistic is NULL unless all team players have that statistic observed.
CREATE OR REPLACE VIEW team_snapshots AS
SELECT s.match_id, s.game_time_s, p.team_id,
    count(*)::UINTEGER AS player_count,
    count(s.observed_time_s)::UINTEGER AS observed_player_count,
    min(s.observed_time_s) AS earliest_observed_time_s,
    max(s.observed_time_s) AS latest_observed_time_s,
    bool_or(s.is_final_observation) AS is_final_observation,
    CASE WHEN count(s.net_worth)=count(*) THEN sum(s.net_worth) END AS net_worth,
    CASE WHEN count(s.unspent_souls)=count(*) THEN sum(s.unspent_souls) END AS unspent_souls,
    CASE WHEN count(s.kills)=count(*) THEN sum(s.kills) END AS kills,
    CASE WHEN count(s.deaths)=count(*) THEN sum(s.deaths) END AS deaths,
    CASE WHEN count(s.assists)=count(*) THEN sum(s.assists) END AS assists,
    CASE WHEN count(s.alive)=count(*) THEN sum(s.alive::INTEGER) END AS alive_players,
    CASE WHEN count(s.hero_damage)=count(*) THEN sum(s.hero_damage) END AS hero_damage,
    CASE WHEN count(s.objective_damage)=count(*) THEN sum(s.objective_damage) END AS objective_damage,
    CASE WHEN count(s.hero_healing)=count(*) THEN sum(s.hero_healing) END AS hero_healing
FROM player_snapshots s JOIN match_players p USING(match_id,player_slot)
GROUP BY s.match_id,s.game_time_s,p.team_id;

CREATE OR REPLACE VIEW processing_status AS
WITH latest AS (
    SELECT * FROM processing_attempts
    QUALIFY row_number() OVER(PARTITION BY match_id ORDER BY started_at_utc DESC, attempt_id DESC)=1
)
SELECT l.match_id, CASE WHEN m.match_id IS NOT NULL THEN 'complete' ELSE l.status END AS status,
    l.status AS latest_status, l.attempt_id, m.successful_attempt_id,
    coalesce(s.sampling_interval_s,l.sampling_interval_s) AS sampling_interval_s,
    l.stage,l.reason,l.started_at_utc,l.finished_at_utc,l.source_id
FROM latest l LEFT JOIN matches m USING(match_id)
LEFT JOIN processing_attempts s ON s.attempt_id=m.successful_attempt_id;
