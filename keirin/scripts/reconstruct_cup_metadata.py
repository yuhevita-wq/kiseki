"""Retrieve actual missing cup grade/name through the original WINTICKET parser.

Cup grade affects the original daily-cap exemptions. It cannot be inferred from
S/A/L class. Fetch once per cup, retaining existing entries/results/odds intact.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_core import crypt


def main() -> None:
    """Read actual source cup metadata; never manufacture a missing grade."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--shard", type=int, choices=range(4), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from src import database
    from src.scraper.winticket import WinticketScraper

    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(args.checkpoint, database.DB_PATH, decrypt=True)
    with database.get_connection() as c:
        races = c.execute(
            "SELECT cup_id,venue_id,race_date,race_no,day_index,race_key FROM wt_races WHERE race_date BETWEEN '2026-01-01' AND '2026-10-23' ORDER BY cup_id,race_date,race_no"
        ).fetchall()
    cups = {}
    for row in races:
        cups.setdefault((row[0], row[1]), []).append(tuple(row))
    keys = [k for i, k in enumerate(sorted(cups)) if i % 4 == args.shard]
    scraper = WinticketScraper()
    records = []
    failed = []
    args.output.mkdir(parents=True, exist_ok=True)
    for i, key in enumerate(keys):
        found = None
        for cup, venue, day, no, day_index, rk in cups[key][:3]:
            got = scraper.fetch_race_data(
                str(venue), str(day), int(no), cup_id=str(cup), day_index=int(day_index)
            )
            if got is not None and got.get("cup_grade") is not None:
                found = {
                    "cup_id": cup,
                    "venue_id": venue,
                    "cup_grade": got["cup_grade"],
                    "cup_name": got.get("cup_name"),
                    "source_race_key": rk,
                    "fetched_at": datetime.now(UTC).isoformat(),
                }
                break
        if found is None:
            failed.append(
                {
                    "cup_id": key[0],
                    "venue_id": key[1],
                    "reason": "original_parser_did_not_return_cup_grade",
                }
            )
        else:
            records.append(found)
        if (i + 1) % 20 == 0 or i + 1 == len(keys):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "metadata.json"
                p.write_text(
                    json.dumps(
                        {"records": records, "missing": failed}, ensure_ascii=False
                    )
                )
                crypt(p, args.output / f"cup-metadata-{args.shard}.json.gpg")
            summary = {
                "shard": args.shard,
                "requested_cups": len(keys),
                "processed_cups": i + 1,
                "retrieved_cups": len(records),
                "missing_cups": len(failed),
                "source_function": "WinticketScraper.fetch_race_data unchanged",
                "entries_results_odds_modified": False,
                "complete": i + 1 == len(keys) and not failed,
            }
            (args.output / f"cup-metadata-{args.shard}.summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(summary), flush=True)
    if not records:
        raise RuntimeError("no_actual_cup_metadata_retrieved")


if __name__ == "__main__":
    main()
