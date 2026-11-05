# Description

My goal is to practice some statistical analysis on data I find interesting. To get that data, I had Codex compose a datapipeline to download, parse, and store game data. The rest is statistical analysis. Currently my interests are in machine learning, so that's what my first few statistical models are going to be on/in.

See `research/` for my work. I read Simon J. D. Prince’s [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) which was an excellent book. I compose a pre-train/fine-tune model to practice using what was introduced in this book, see [model outline](research/model_one/outline.pdf) for my detailed outline and results.

I am currently reading Anastasios N. Angelopoulos and Stephen Bates’s [*A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*](https://arxiv.org/abs/2107.07511), a shorter summary article about conformal prediction.

## Current Objectives
- Considering improvements to my first model, including hyperparameter adjustment, algorithm changes, and some small architectural changes.
- Consider the discussion in "Weather Models".
- Consider the topics covered in the conformal prediction article.

### Weather Models

Prince discusses in his book the topic of neural networks for weather prediction. These models suffer from the problem that standard prediction techniques don't work. When predicting weather we are concerned more with predicting rare outcomes (thunderstorms, hurricanes, etc.) rather than, the most often case of, the same as yesterday.

Hence, applying this to Deadlock. A large percentage of games, >60%, are blow-out games. I am less interested in how my model predicts games that are obvious from the first few minutes, than it's predictions on games that aren't so obvious.

## Sources
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) by Simon J. D. Prince (MIT Press, 2023)
- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511) by Anastasios N. Angelopoulos and Stephen Bates
- [Deadlock API data lake and MCP access](https://deadlock-api.com/data-dumps), used to discover matches with recorded replay salts.
- [Deadlock data-lake manifest](https://data.deadlock-api.com/v1/manifest.json), used by the end-match Parquet pipeline.
- [Deadlock API metadata implementation](https://github.com/deadlock-api/deadlock-api/blob/master/api/src/routes/v1/matches/metadata.rs) and [API mode definitions](https://github.com/deadlock-api/deadlock-api/blob/master/api/src/routes/v1/matches/types.rs).
- [Valve metadata protocol](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/citadel_gcmessages_common.proto), [replay header](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/demo.proto), and [user messages](https://github.com/SteamDatabase/Protobufs/blob/master/deadlock/citadel_usermessages.proto).
- [deadem parser/engine](https://github.com/Igor-Losev/deadem), pinned to `4.0.1`; the parser includes the original project's small decoder fixes for newer Source 2 fields.
- [Extracted Valve currency enum](https://github.com/SteamTracking/GameTracking-Deadlock/blob/master/DumpSource2/schemas/client/ECurrencyType.h), [player state schema](https://github.com/SteamTracking/GameTracking-Deadlock/blob/master/DumpSource2/schemas/client/PlayerDataGlobal_t.h), and [Deadworks game-clock implementation](https://github.com/Deadworks-net/deadworks/blob/76f73b504d7b811b678d092a1e762a8e08b5d2d9/managed/DeadworksManaged.Api/GameRules.cs).
