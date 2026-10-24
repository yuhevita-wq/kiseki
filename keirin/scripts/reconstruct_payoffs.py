"""Recover missing official payouts with Nyanta's unchanged collector only."""

from __future__ import annotations

import argparse
import calendar
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_core import crypt


def main() -> None:
    """Collect absent payout records, save encrypted resumable daily progress."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--first-day", type=int, default=1)
    args = ap.parse_args()
    from src import database, official_payoffs

    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(args.checkpoint, database.DB_PATH, decrypt=True)
    y, m = map(int, args.month.split("-"))
    first = date(y, m, args.first_day)
    last = min(date(y, m, calendar.monthrange(y, m)[1]), date(2026, 10, 23))
    if y != 2026 or first > last:
        raise RuntimeError("invalid_holdout_month")
    args.output.mkdir(parents=True, exist_ok=True)
    plaintext = args.output / "payoffs.sqlite"
    report = {
        "month": args.month,
        "from": first.isoformat(),
        "through": last.isoformat(),
        "collector": "src.official_payoffs.collect (unchanged)",
        "days": [],
    }
    day = first
    while day <= last:
        with database.get_connection() as c:
            keys = [
                r[0]
                for r in c.execute(
                    "SELECT race_key FROM wt_races WHERE race_date=? ORDER BY race_key",
                    (day.isoformat(),),
                )
            ]
        stats = official_payoffs.collect(keys)
        report["days"].append({"date": day.isoformat(), **stats})
        # Snapshot only missing-field repairs, not the whole original corpus.
        plaintext.unlink(missing_ok=True)
        with (
            sqlite3.connect(database.DB_PATH) as src,
            sqlite3.connect(plaintext) as dst,
        ):
            schema = src.execute(
                "SELECT sql FROM sqlite_master WHERE name='wt_payoffs'"
            ).fetchone()[0]
            dst.execute(schema)
            cols = [
                r[1]
                for r in src.execute("PRAGMA table_info(wt_payoffs)")
                if r[1] != "id"
            ]
            names = ",".join('"' + c + '"' for c in cols)
            dst.executemany(
                f"INSERT INTO wt_payoffs ({names}) VALUES ({','.join('?' for _ in cols)})",
                src.execute(f"SELECT {names} FROM wt_payoffs"),
            )
            dst.commit()
            report["official_payout_rows"] = dst.execute(
                "SELECT COUNT(*) FROM wt_payoffs"
            ).fetchone()[0]
        crypt(plaintext, args.output / f"official-payoffs-{args.month}.sqlite.gpg")
        plaintext.unlink()
        (args.output / f"official-payoffs-{args.month}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report["days"][-1]), flush=True)
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
