"""Compare real archived card fields with the captured original public cards."""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_core import crypt


def main() -> None:
    """Measure input agreement without interpreting capture time as issuance."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    reference = json.loads(
        gzip.decompress(
            (ROOT / "reconstruction/original_reference_20261023.json.gz").read_bytes()
        )
    )
    fields = ("line_group", "line_pos", "style", "race_point", "prediction_mark")
    report = {f: {"compared": 0, "equal": 0, "missing": 0} for f in fields}
    missing_entries = 0
    n_entries = 0
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "source.sqlite"
        crypt(args.checkpoint, db, decrypt=True)
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            actual = {
                (r["race_key"], r["frame_no"]): dict(r)
                for r in conn.execute(
                    "SELECT race_key,frame_no,line_group,line_pos,style,race_point,prediction_mark FROM wt_entries WHERE race_key LIKE '20261023_%'"
                )
            }
        for race in reference["data"]["displayed"]:
            for e in race["entries"]:
                n_entries += 1
                a = actual.get((race["race_key"], int(e["frame_no"])))
                if a is None:
                    missing_entries += 1
                    continue
                for f in fields:
                    if e.get(f) is None or a.get(f) is None:
                        report[f]["missing"] += 1
                        continue
                    report[f]["compared"] += 1
                    report[f]["equal"] += int(a[f] == e[f])
    output = {
        "date": "2026-10-23",
        "reference_entries": n_entries,
        "missing_entries": missing_entries,
        "fields": report,
        "reference_captured_at": reference["captured_at"],
        "original_morning_asof_verified": False,
        "behind_comparison": "Not available in captured original card fields; cannot prove morning extended-stat equality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
