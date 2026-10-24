"""Storage regressions: surrogate-ID collisions must not corrupt natural keys."""

from __future__ import annotations

import sqlite3

import pytest
from scripts.reconstruct_core import merge_day


def make_source(path, race_key: str, venue: str = "V") -> None:
    """Minimal storage fixture; no model or probability is generated."""
    with sqlite3.connect(path) as c:
        c.executescript("""
            CREATE TABLE wt_races(race_key TEXT PRIMARY KEY);
            CREATE TABLE wt_entries(id INTEGER PRIMARY KEY, race_key TEXT,
                frame_no INTEGER, UNIQUE(race_key,frame_no));
            CREATE TABLE venue_info(venue_code TEXT PRIMARY KEY, name TEXT);
        """)
        c.execute("INSERT INTO wt_races VALUES (?)", (race_key,))
        c.execute("INSERT INTO wt_entries VALUES (1,?,1)", (race_key,))
        c.execute("INSERT INTO venue_info VALUES ('01',?)", (venue,))


def make_destination() -> sqlite3.Connection:
    """Create matching isolated storage without source records."""
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE wt_races(race_key TEXT PRIMARY KEY);
        CREATE TABLE wt_entries(id INTEGER PRIMARY KEY, race_key TEXT,
            frame_no INTEGER, UNIQUE(race_key,frame_no));
        CREATE TABLE venue_info(venue_code TEXT PRIMARY KEY,name TEXT);
    """)
    return c


def test_daily_integer_ids_do_not_overwrite_other_races(tmp_path):
    """Daily id=1 must yield two distinct actual race/entry keys after merge."""
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    make_source(a, "20230101_01_01")
    make_source(b, "20230102_01_01")
    with make_destination() as dst:
        merge_day(a, dst)
        merge_day(b, dst)
        assert dst.execute(
            "SELECT race_key,frame_no FROM wt_entries ORDER BY race_key"
        ).fetchall() == [("20230101_01_01", 1), ("20230102_01_01", 1)]


def test_repeated_race_key_is_rejected_instead_of_overwritten(tmp_path):
    """A cross-day duplicated race is an audit failure, never last-write-wins."""
    a = tmp_path / "a.db"
    make_source(a, "20230101_01_01")
    with make_destination() as dst:
        merge_day(a, dst)
        with pytest.raises(sqlite3.IntegrityError):
            merge_day(a, dst)


def test_conflicting_venue_metadata_is_rejected(tmp_path):
    """Different source venue definitions cannot be silently substituted."""
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    make_source(a, "20230101_01_01", "old")
    make_source(b, "20230102_01_01", "different")
    with make_destination() as dst:
        merge_day(a, dst)
        with pytest.raises(RuntimeError, match="conflicting_venue_metadata"):
            merge_day(b, dst)


def test_checkpoint_includes_committed_uncheckpointed_wal(tmp_path, monkeypatch):
    """A physical .db copy loses WAL-only rows; SQLite backup must retain them."""
    import shutil

    import scripts.reconstruct_core as core

    path = tmp_path / "source.db"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.executescript(
        "CREATE TABLE wt_races(race_key TEXT,race_date TEXT); CREATE TABLE wt_entries(race_key TEXT);"
    )
    writer.execute("INSERT INTO wt_races VALUES ('20261023_01_01','2026-10-23')")
    writer.execute("INSERT INTO wt_entries VALUES ('20261023_01_01')")
    writer.commit()
    monkeypatch.setattr(
        core, "crypt", lambda src, dst, **kwargs: shutil.copyfile(src, dst)
    )
    report = {"days": [{"rows": {"wt_races": 1, "wt_entries": 1}}]}
    output = tmp_path / "checkpoint.gpg"
    core.save_database_checkpoint(path, output, report)
    with sqlite3.connect(output) as restored:
        assert restored.execute("SELECT race_date FROM wt_races").fetchall() == [
            ("2026-10-23",)
        ]
        assert restored.execute("SELECT COUNT(*) FROM wt_entries").fetchone()[0] == 1
    assert report["verified_checkpoint_counts"] == {"wt_races": 1, "wt_entries": 1}
    writer.close()


def test_original_odds_loaders_read_real_sqlite_values_unchanged(tmp_path, monkeypatch):
    """Exercise the two original SQL loaders through the storage-only bridge."""
    from scripts.reconstruct_odds import ReadConnection
    from src import database

    from scripts import train_odds_prediction_tf as original

    path = tmp_path / "odds.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()
    database.migrate_db()
    with sqlite3.connect(path) as c:
        c.execute(
            "INSERT INTO wt_entries(race_key,frame_no,race_point,prediction_mark,player_class,style,line_group,line_size,line_pos,is_line_leader,first_rate,second_rate,third_rate) VALUES ('20250101_01_01',1,90.5,1,'A1','逃',1,3,1,1,20,30,40)"
        )
        c.executemany(
            "INSERT INTO wt_odds(race_key,bet_type,combination,odds_value) VALUES ('20250101_01_01','trifecta',?,?)",
            [("1-2-3", 25.7), ("1-3-2", 9999.9)],
        )
    monkeypatch.setattr(original, "_connect", lambda: ReadConnection(path))
    monkeypatch.setattr(original, "N_CAR", 7)
    entries = original._load_entries(["20250101_01_01"])
    boards = original._load_final_boards(["20250101_01_01"])
    assert entries["20250101_01_01"][1]["race_point"] == 90.5
    assert entries["20250101_01_01"][1]["mark"] == 1
    assert boards == {"20250101_01_01": {(1, 2, 3): 25.7}}
