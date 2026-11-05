"""Small synthetic checks; never trains on the collected research dataset."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from app.normalize import END_MATCH_SCHEMA, MATCH_SCHEMA
import pretrain_static as stage1


def roster(match_id=1, rank=60, hero=1):
    return [{"match_id": match_id, "player_slot": slot, "team_id": slot // 6,
             "hero_id": hero + slot, "initial_rank_badge": rank} for slot in range(12)]


def write_fixture(root, count=20):
    (root / "end_match").mkdir(parents=True)
    (root / "matches").mkdir()
    for mid in range(1, count + 1):
        players = roster(mid, rank=10 + mid)
        players[0]["initial_rank_badge"] = None
        for player in players:
            player.update(kills=999, net_worth=999999, won=player["team_id"] == mid % 2)
        pq.write_table(pa.Table.from_pylist(players, schema=END_MATCH_SCHEMA), root / "end_match" / f"{mid}.parquet")
        pq.write_table(pa.Table.from_pylist([{"match_id": mid, "match_outcome": "TeamWin",
                                            "winning_team_id": mid % 2}], schema=MATCH_SCHEMA),
                       root / "matches" / f"{mid}.parquet")


class Stage1Tests(unittest.TestCase):
    def test_preprocessing_is_fitted_only_on_training_and_handles_unknowns(self):
        players = roster(rank=20)
        players[0]["initial_rank_badge"] = 10
        players[1]["initial_rank_badge"] = 30
        players[2]["initial_rank_badge"] = None
        preprocessing = stage1.fit_preprocessing([{"players": players}])
        heldout = roster(rank=999, hero=999)
        heldout[0]["initial_rank_badge"] = None
        heldout[1]["initial_rank_badge"] = 0
        heldout[2]["initial_rank_badge"] = 1
        heroes, ranks = stage1.encode_rosters([heldout], preprocessing)
        self.assertEqual(preprocessing["rank"]["median"], 20)
        self.assertEqual(preprocessing["rank"]["max"], 30)
        self.assertTrue((heroes == 0).all())
        torch.testing.assert_close(ranks.flatten()[:4], torch.tensor([0.5, 0.5, 0., 1.]))
        for constant in (None, 50):
            fitted = stage1.fit_preprocessing([{"players": roster(rank=constant)}])
            self.assertTrue((stage1.encode_rosters([heldout], fitted)[1] == 0).all())

    def test_split_reuse_does_not_reassign_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.csv"
            first = stage1.get_splits(list(range(20)), path, 42, .1, .1)
            contents = path.read_bytes()
            second = stage1.get_splits(list(reversed(range(20))), path, 999, .2, .2)
            self.assertEqual(first, second)
            self.assertEqual(contents, path.read_bytes())
            # Later stages may have only a subset of the original matches.
            subset = [next(int(mid) for mid, group in first.items() if group == split) for split in stage1.SPLITS]
            self.assertEqual(first, stage1.get_splits(subset, path, 1, .1, .1))
            with self.assertRaisesRegex(ValueError, "Extend it explicitly"):
                stage1.get_splits(list(range(21)), path, 42, .1, .1)
            with path.open("a", newline="") as stream:
                csv.writer(stream).writerow([0, "train"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                stage1.get_splits(list(range(20)), path, 42, .1, .1)

    def test_loader_excludes_outcome_features_and_rejects_bad_rosters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, count=3)
            matches, skipped = stage1.load_matches(root)
            self.assertEqual(skipped, [])
            self.assertEqual([m["label"] for m in matches], [1., 0., 1.])
            self.assertEqual(set(matches[0]["players"][0]), set(stage1.PLAYER_COLUMNS))
            # Draws are not negative labels.
            pq.write_table(pa.Table.from_pylist([{"match_id": 3, "match_outcome": "MatchDraw",
                                                 "winning_team_id": 0}], schema=MATCH_SCHEMA),
                           root / "matches" / "3.parquet")
            matches, skipped = stage1.load_matches(root)
            self.assertEqual(len(matches), 2)
            self.assertEqual(skipped[0]["match_id"], 3)
        invalid = roster()
        invalid[0]["team_id"] = 1
        with self.assertRaisesRegex(ValueError, "six players"):
            stage1.ordered_players(invalid)
        invalid = roster()
        invalid[0]["player_slot"] = 1
        with self.assertRaisesRegex(ValueError, "distinct"):
            stage1.ordered_players(invalid)

    def test_pooling_is_invariant_to_within_team_player_order_and_returns_logits(self):
        torch.manual_seed(42)
        model = stage1.StaticWinModel(13).eval()
        heroes = torch.arange(1, 13).reshape(1, 2, 6)
        ranks = torch.rand(1, 2, 6)
        permutation = [5, 2, 1, 3, 0, 4]
        with torch.inference_mode():
            torch.testing.assert_close(model(heroes, ranks), model(heroes[:, :, permutation], ranks[:, :, permutation]))
            model.win_head[-1].weight.zero_()
            model.win_head[-1].bias.fill_(2)
            self.assertEqual(model(heroes, ranks).item(), 2.)

    def test_training_stops_restores_transfers_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "data")
            args = stage1.build_parser().parse_args(["train", "--data-dir", str(root / "data"),
                                                   "--output-dir", str(root / "runs"), "--epochs", "15"])
            states, actual_losses = [], []
            evaluate = stage1.validation_bce

            def plateau(model, loader, device):
                actual_losses.append(evaluate(model, loader, device))
                states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
                # One best epoch followed by ten unimproved epochs.
                return .5 if len(states) == 1 else .6

            with patch.object(stage1, "validation_bce", side_effect=plateau):
                run_dir = stage1.train(args)
            self.assertEqual(len(states), 11)
            self.assertTrue(all(loss > 0 for loss in actual_losses))
            model, checkpoint = stage1.load_stage1(run_dir / "best_model.pt")
            self.assertEqual(checkpoint["best_epoch"], 1)
            self.assertFalse(model.training)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, states[0][key], rtol=0, atol=0)
            self.assertFalse(torch.equal(states[0]["win_head.2.bias"], states[-1]["win_head.2.bias"]))
            self.assertTrue(stage1.verify_reload(run_dir / "best_model.pt")["passed"])
            transfer = torch.load(run_dir / "static_encoder.pt", weights_only=True)
            target = stage1.StaticWinModel(len(transfer["preprocessing"]["hero_id_to_index"]) + 1)
            target.hero_embedding.load_state_dict(transfer["hero_embedding_state_dict"])
            target.player_mlp.load_state_dict(transfer["player_mlp_state_dict"])
            self.assertNotIn("model_state_dict", transfer)
            heroes, ranks = stage1.encode_rosters([roster()], checkpoint["preprocessing"])
            with torch.inference_mode():
                inputs = torch.cat((model.hero_embedding(heroes), ranks.unsqueeze(-1)), -1)
                other_inputs = torch.cat((target.hero_embedding(heroes), ranks.unsqueeze(-1)), -1)
                torch.testing.assert_close(model.player_mlp(inputs), target.player_mlp(other_inputs), rtol=0, atol=0)
            probability = stage1.win_probabilities(model, [roster()], checkpoint["preprocessing"]).item()
            self.assertTrue(0 < probability < 1)
            self.assertTrue((run_dir / "loss_curve.png").stat().st_size > 1000)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["epochs_completed"], 11)


if __name__ == "__main__":
    unittest.main()
