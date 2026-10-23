-- These views describe the bounded, locally ingested subset of the public lake.
-- Latest player versions are resolved during ingestion; require one complete
-- match version so partial updates cannot mix old and new player statistics.
CREATE OR REPLACE VIEW eligible_matches AS
SELECT match_id
FROM raw_match_player
GROUP BY match_id
HAVING count(*) = 12
   AND count(DISTINCT player_slot) = 12
   AND count(DISTINCT nullif(account_id, 0)) = count(nullif(account_id, 0))
   AND count(*) FILTER (WHERE team = 'Team0') = 6
   AND count(*) FILTER (WHERE team = 'Team1') = 6
   AND count(*) FILTER (WHERE duration_s > 900 AND game_mode = 'Normal'
                       AND match_mode IN ('Ranked', 'Unranked')
                       AND match_outcome = 'TeamWin'
                       AND winning_team IN ('Team0', 'Team1')
                       AND hero_id > 0 AND created_at IS NOT NULL) = 12
   AND count(start_time) = 12
   AND count(DISTINCT start_time) = 1
   AND count(DISTINCT duration_s) = 1
   AND count(DISTINCT game_mode) = 1
   AND count(DISTINCT match_mode) = 1
   AND count(DISTINCT match_outcome) = 1
   AND count(DISTINCT winning_team) = 1
   AND count(DISTINCT created_at) = 1;

CREATE TABLE IF NOT EXISTS selected_matches (match_id UBIGINT PRIMARY KEY);

CREATE OR REPLACE VIEW match_player AS
SELECT p.* FROM raw_match_player p
JOIN selected_matches s USING (match_id)
JOIN eligible_matches e USING (match_id);

CREATE OR REPLACE VIEW matches AS
SELECT match_id, min(start_time) AS start_time, min(duration_s) AS duration_s,
       min(game_mode) AS game_mode, min(match_mode) AS match_mode,
       min(match_outcome) AS match_outcome, min(winning_team) AS winning_team,
       min(created_at) AS created_at, count(*) AS player_count
FROM match_player GROUP BY match_id;
