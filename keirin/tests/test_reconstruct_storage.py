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
