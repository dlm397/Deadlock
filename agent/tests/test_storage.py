import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent.io import GROUPS, group_for, load_table, save_match


EXPECTED_GROUPS = {
    "summaries": (
        "matches", "players", "game_versions", "metadata_teams", "metadata_damage",
        "metadata_custom_stats", "metadata_gold_sources",
    ),
    "states": ("player_state", "team_state", "metadata_player_stats", "metadata_player_paths"),
    "purchases": ("purchases", "metadata_item_records"),
    "events": ("events", "metadata_objectives", "metadata_midboss", "legacy_events", "replay_event_counts"),
}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "raw" / "99"
        self.manifest = {"match_id": 99, "sample_interval": 10}

    def rows(self):
        return [
            {"match_id": 99, "player_id": 2**53 + 1, "game_time": 0,
             "value": 0.0, "label": "雪", "nested": {"ids": [1, 2], "missing": None}},
            {"match_id": 99, "player_id": None, "game_time": 10,
             "value": None, "label": None, "nested": None},
        ]

    def test_every_table_has_one_explicit_group(self):
        self.assertEqual(set(GROUPS), set(EXPECTED_GROUPS))
        actual = [name for names in GROUPS.values() for name in names]
        self.assertEqual(len(actual), len(set(actual)))
        for group, names in EXPECTED_GROUPS.items():
            self.assertEqual(set(GROUPS[group]), set(names))
            for name in names:
                self.assertEqual(group_for(name), group)
        with self.assertRaises((KeyError, ValueError)):
            group_for("unrecognized_new_table")

    def test_four_archives_preserve_all_tables_and_values(self):
        tables = {name: self.rows() for names in EXPECTED_GROUPS.values() for name in names}
        save_match(self.folder, tables, self.manifest)
        self.assertEqual(
            {path.name for path in self.folder.glob("*.zip")},
            {f"{group}.zip" for group in EXPECTED_GROUPS},
        )
        self.assertFalse(list(self.folder.rglob("*.csv")))
        self.assertFalse(list(self.folder.rglob("*.csv.gz")))
        for group, names in EXPECTED_GROUPS.items():
            with zipfile.ZipFile(self.folder / f"{group}.zip") as archive:
                self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))
                self.assertEqual(
                    {Path(name).stem for name in archive.namelist() if name.endswith(".parquet")},
                    set(names),
                )
                if group == "summaries":
                    manifest = json.loads(archive.read("manifest.json"))
                    self.assertEqual(manifest["match_id"], 99)
            for name in names:
                with self.subTest(table=name):
                    frame = load_table(self.folder, name)
                    self.assertEqual(len(frame), 2)
                    self.assertEqual(int(frame.loc[0, "player_id"]), 2**53 + 1)
                    self.assertTrue(frame["player_id"].isna().iloc[1])
                    self.assertEqual(frame.loc[0, "label"], "雪")
                    self.assertEqual(frame.loc[0, "value"], 0)
                    self.assertTrue(frame["value"].isna().iloc[1])
                    self.assertEqual(json.loads(frame.loc[0, "nested"]), self.rows()[0]["nested"])
                    self.assertTrue(frame["nested"].isna().iloc[1])

    def test_columns_can_be_read_without_loading_other_fields(self):
        save_match(self.folder, {"player_state": self.rows()}, self.manifest)
        frame = load_table(self.folder, "player_state", columns=["match_id", "game_time"])
        self.assertEqual(list(frame.columns), ["match_id", "game_time"])
        self.assertEqual(frame["game_time"].tolist(), [0, 10])

    def test_empty_tables_remain_readable(self):
        save_match(self.folder, {"events": []}, self.manifest)
        self.assertEqual(len(load_table(self.folder, "events")), 0)

    def test_unknown_table_cannot_replace_committed_archives(self):
        save_match(self.folder, {"matches": self.rows()}, self.manifest)
        before = {path.name: path.read_bytes() for path in self.folder.glob("*.zip")}
        with self.assertRaises((KeyError, ValueError)):
            save_match(self.folder, {"matches": [], "unrecognized_new_table": []}, self.manifest)
        after = {path.name: path.read_bytes() for path in self.folder.glob("*.zip")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
