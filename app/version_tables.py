"""Small, versioned CSV catalogs for asset IDs and normalized event types."""

import csv
from pathlib import Path


ITEM_COLUMNS = ("client_version", "item_id", "name", "class_name", "type",
                "item_slot_type", "item_tier", "cost")
EVENT_COLUMNS = ("client_version", "event_type", "event_name", "description",
                 "participant_roles", "notes")

# These labels describe this pipeline's normalized events, not Valve event IDs.
EVENT_DEFINITIONS = (
    ("hero_kill", "Hero kill", "A hero death with known combat participants.",
     "victim; killer; assister",
     "Participants share event_id; count distinct event IDs. Death duration and time to kill "
     "belong to the victim row. Missing killer or assister information is not inferred."),
    ("item_acquired", "Item acquired", "A shop upgrade was acquired by a player.", "player",
     "Item acquisition does not establish the actual amount paid; catalog cost is a definition price."),
    ("item_removed", "Item removed", "An item was removed from a player's records or inventory.", "player",
     "Removal alone does not prove a sale; it can accompany an upgrade. Refund amount is unknown."),
    ("item_upgrade", "Item upgraded", "A replay notification identifies an upgrade to a shop item.", "player",
     "An upgrade notification does not establish the actual payment or refund."),
    ("item_record", "Unclassified item record", "An asset record could not be classified as an ability or shop upgrade.", "player",
     "Definitions can be unavailable or the asset can have another type. Do not assume a purchase."),
    ("ability_progression", "Ability progression", "A metadata record identifies ability progression.", "player",
     "The specific unlock or upgrade action is unverified. upgrade_id and upgrade_info are raw protocol "
     "codes, not decoded ability levels."),
    ("ability_unlock", "Ability unlocked", "A verified replay notification identifies an ability unlock.", "player",
     "This describes progression, not an ability cast."),
    ("ability_upgrade", "Ability upgraded", "A verified replay notification identifies an ability upgrade.", "player",
     "This describes progression, not an ability cast or a verified numeric level."),
    ("ability_removed", "Ability removed", "An ability removal is recorded in metadata or a replay notification.", "player",
     "Removal alone does not establish a sale or refund."),
    ("objective_first_damage", "Objective first damaged", "The first recorded damage time for a team objective.", "",
     "objective_team_id is the owner, not the attacker. objective_id is a team-local protocol ID, "
     "not an item catalog ID; individual attackers are not identified."),
    ("objective_destroyed", "Objective destroyed", "The destruction time for a team objective.", "",
     "objective_team_id is the owner, not the attacker. objective_id is a team-local protocol ID, "
     "not an item catalog ID; individual attackers are not identified."),
    ("midboss_killed", "Midboss killed", "A midboss death with the killing team when known.", "killing_team",
     "A team event does not identify an individual killer. The death time is not the rejuvenator claim time."),
    ("rejuvenator_claim", "Rejuvenator claimed", "Metadata identifies the team that claimed a rejuvenator.", "claiming_team",
     "Claim time is unverified and remains missing. related_event_id links the associated midboss death; "
     "its time must not be substituted for claim time."),
)


def _exact_id(value, label):
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or not 0 < value < 2**32:
        raise ValueError(f"{label} must be an exact positive uint32 integer")
    return value


def _write_csv(path, columns, rows):
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_version_tables(supplemental_dir, client_version, items):
    """Write small CSV catalogs; return whether this version's item catalog exists.

    Catalogs live in supplemental_dir/version, outside either match dataset.
    `items` must be the exact-version API definitions. The shared catalog labels
    item_id, ability_id and imbued_ability_id; absent scalar values stay blank.
    Unknown versions produce no files. Empty definitions never erase a catalog.
    """
    if client_version is None:
        return False
    version = _exact_id(client_version, "client_version")
    folder = Path(supplemental_dir) / "version"
    folder.mkdir(parents=True, exist_ok=True)
    event_rows = [dict(zip(EVENT_COLUMNS, (version, *definition)))
                  for definition in EVENT_DEFINITIONS]
    _write_csv(folder / f"{version}_event.csv", EVENT_COLUMNS, event_rows)
    item_path = folder / f"{version}_item.csv"
    if items:
        item_rows = [{"client_version": version, "item_id": _exact_id(item["id"], "item_id"),
                      **{name: item.get(name) for name in ITEM_COLUMNS[2:]}} for item in items]
        if len({row["item_id"] for row in item_rows}) != len(item_rows):
            raise ValueError("versioned item definitions contain duplicate IDs")
        item_rows.sort(key=lambda row: row["item_id"])
        _write_csv(item_path, ITEM_COLUMNS, item_rows)
    return item_path.is_file()
