import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from agent.io import read_json
from agent.lake_storage import complete_lake_match, save_lake_match
from dataset import iter_lake_matches, lake_match_ids, match_ids, read_lake_match


class LakeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.root = self.output / "lake"
        self.folder = self.root / "99"
        self.rows = [
            {"match_id": 99, "account_id": 2**64 - 1, "player_slot": 0, "created_at": "2026-09-28T15:30:00",
             "stats": [{"value": 2**64 - 1, "nested": {"health": None}}], "future_field": "雪", "score": 0.1},
            {"match_id": 99, "account_id": None, "player_slot": 0, "created_at": "2026-09-28T15:31:00",
             "stats": [], "future_field": None, "score": None},
        ]
        self.schemas = {
            "match_player": [{"column_name": "match_id", "column_type": "BIGINT"},
                             {"column_name": "account_id", "column_type": "UBIGINT"},
                             {"column_name": "stats", "column_type": "MAP(UINTEGER, INTEGER)[]"}],
            "match_salts": [{"name": "match_id", "type": "Int64"},
                            {"name": "replay_salt", "type": "Nullable(UInt64)"}],
        }
        self.provenance = {"schemas": self.schemas, "source_url": "https://example.invalid/lake"}

    def save(self, rows=None):
        return save_lake_match(self.root, 99, {"match_player": self.rows if rows is None else rows,
                                             "match_salts": []}, self.provenance)

    def test_lossless_nullable_uint64_nested_data_and_new_columns(self):
        status = self.save()
        self.assertTrue(complete_lake_match(self.folder))
        self.assertEqual(status["row_counts"], {"match_player": 2, "match_salts": 0})
        self.assertEqual(status["column_encodings"]["match_player"]["stats"], "json")
        self.assertEqual(status["schemas"], self.schemas)
        result = read_lake_match(99, output_dir=self.output)
        self.assertEqual(str(result["account_id"].dtype), "UInt64")
        self.assertEqual(result.loc[0, "account_id"], 2**64 - 1)
        self.assertTrue(result["account_id"].isna().iloc[1])
        self.assertEqual(json.loads(result.loc[0, "stats"]), self.rows[0]["stats"])
        self.assertEqual(json.loads(result.loc[1, "stats"]), [])
        self.assertEqual(result.loc[0, "future_field"], "雪")
        self.assertEqual(result.loc[0, "score"], 0.1)
        self.assertEqual(result["created_at"].tolist(), [row["created_at"] for row in self.rows])
        metadata = pq.read_metadata(self.folder / "match_player.parquet")
        self.assertEqual(metadata.row_group(0).column(0).compression, "ZSTD")
        self.assertFalse((self.output / "raw").exists())
        self.assertEqual(match_ids(self.output), [])

    def test_empty_tables_keep_schema_and_integer_types(self):
        self.save()
        frame = read_lake_match(99, "match_salts", output_dir=self.output)
        self.assertEqual(list(frame.columns), ["match_id", "replay_salt"])
        self.assertEqual(len(frame), 0)
        self.assertEqual(str(frame["replay_salt"].dtype), "UInt64")

    def test_duplicate_rows_and_revisions_are_not_discarded(self):
        self.save(self.rows + [self.rows[0]])
        frame = read_lake_match(99, output_dir=self.output)
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame["player_slot"].tolist(), [0, 0, 0])
        self.assertEqual(frame["created_at"].tolist(), [self.rows[0]["created_at"], self.rows[1]["created_at"],
                                                       self.rows[0]["created_at"]])

    def test_projection_and_iteration_only_use_lake_matches(self):
        self.save()
        self.assertEqual(lake_match_ids(self.output), [99])
        [(mid, frame)] = list(iter_lake_matches(columns=["match_id", "account_id"], output_dir=self.output))
        self.assertEqual(mid, 99)
        self.assertEqual(list(frame.columns), ["match_id", "account_id"])
        self.assertEqual(int(frame.loc[0, "account_id"]), 2**64 - 1)

    def test_corrupt_file_is_not_complete_and_selected_table_is_checked(self):
        self.save()
        (self.folder / "match_salts.parquet").write_bytes(b"bad")
        self.assertFalse(complete_lake_match(self.folder))
        self.assertEqual(lake_match_ids(self.output), [])
        self.assertEqual(len(read_lake_match(99, output_dir=self.output)), 2)
        with self.assertRaisesRegex(ValueError, "incomplete or damaged"):
            read_lake_match(99, "match_salts", output_dir=self.output)
        with self.assertRaises(ValueError):
            read_lake_match(99, "../match_player", output_dir=self.output)

    def test_failure_during_write_preserves_existing_match(self):
        self.save()
        before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
        with patch("agent.lake_storage.pq.write_table", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.save([])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.folder.iterdir()})
        self.assertEqual([path.name for path in self.root.iterdir()], ["99"])

    def test_failure_during_commit_rolls_back_old_match(self):
        self.save()
        before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
        original = Path.replace

        def fail_staging(path, target):
            if path.name.startswith(".99-") and "backup" not in path.name and Path(target) == self.folder:
                raise PermissionError("rename denied")
            return original(path, target)

        with patch.object(Path, "replace", fail_staging):
            with self.assertRaisesRegex(PermissionError, "rename denied"):
                self.save([])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.folder.iterdir()})
        self.assertTrue(complete_lake_match(self.folder))

    def test_successful_replacement_updates_both_tables_together(self):
        self.save()
        self.save([self.rows[0]])
        self.assertEqual(len(read_lake_match(99, output_dir=self.output)), 1)
        self.assertEqual(read_json(self.folder / "status.json")["row_counts"]["match_player"], 1)
        self.assertEqual([path.name for path in self.root.iterdir()], ["99"])

    def test_invalid_tables_or_integer_coercion_cannot_replace_match(self):
        self.save()
        before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
        with self.assertRaises(ValueError):
            save_lake_match(self.root, 99, {"match_player": []}, self.provenance)
        with self.assertRaises(ValueError):
            self.save([dict(self.rows[0], account_id=float(2**63))])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.folder.iterdir()})

    def test_integer_inference_has_no_float_intermediate(self):
        rows = [{"large": 2**64 - 1, "negative": -(2**63)}, {"large": None, "negative": None}]
        save_lake_match(self.root, 100, {"match_player": rows, "match_salts": []}, {})
        frame = read_lake_match(100, output_dir=self.output)
        self.assertEqual(int(frame.loc[0, "large"]), 2**64 - 1)
        self.assertEqual(int(frame.loc[0, "negative"]), -(2**63))
        self.assertTrue(frame["large"].isna().iloc[1])

    def test_truncated_manifest_is_treated_as_incomplete(self):
        self.save()
        path = self.folder / "status.json.gz"
        path.write_bytes(path.read_bytes()[:20])
        self.assertFalse(complete_lake_match(self.folder))
        self.assertEqual(lake_match_ids(self.output), [])
        with self.assertRaisesRegex(ValueError, "incomplete or damaged"):
            read_lake_match(99, output_dir=self.output)


if __name__ == "__main__":
    unittest.main()
