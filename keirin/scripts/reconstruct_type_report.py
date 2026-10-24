"""Summarize actual adopted stakes and official settlement by original A-F type."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reconstruct_core import crypt


def summarize(folder: Path) -> dict:
    """Decrypt inside Actions only and reconcile type sums to native summary."""
    summary = json.loads((folder / "runtime-summary.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "rows.json"
        crypt(folder / "runtime-rows.json.gpg", p, decrypt=True)
        accepted = json.loads(p.read_text())["accepted"]
    if len({r["race_key"] for r in accepted}) != len(accepted):
        raise RuntimeError("duplicate_adopted_race")
    types = {
        t: {
            "adopted": 0,
            "settled": 0,
            "pending": 0,
            "hits": 0,
            "purchase_total": 0,
            "purchase_settled": 0,
            "official_return_settled": 0,
        }
        for t in "ABCDEF"
    }
    for row in accepted:
        b = types[row["type_label"]]
        amount = sum(int(l["stake"]) for l in row["legs"])
        b["adopted"] += 1
        b["purchase_total"] += amount
        s = row["settlement"]
        if s["settled_at"] is None:
            b["pending"] += 1
            continue
        b["settled"] += 1
        b["hits"] += int(bool(s["hit"]))
        b["purchase_settled"] += amount
        b["official_return_settled"] += (s["payout"] or 0) + (s["void_refund"] or 0)
    for b in types.values():
        b["net_settled"] = b["official_return_settled"] - b["purchase_settled"]
        b["roi_settled"] = (
            b["official_return_settled"] / b["purchase_settled"]
            if b["purchase_settled"]
            else None
        )
    fields = {
        "adopted": "adopted_races",
        "settled": "settled_adopted_races",
        "pending": "pending_adopted_races",
        "purchase_settled": "stake_settled",
        "official_return_settled": "official_return_settled",
    }
    for field, expected in fields.items():
        if sum(b[field] for b in types.values()) != summary[expected]:
            raise RuntimeError("type_totals_do_not_reconcile_to_native_summary")
    return {
        "period": summary["period"],
        "types": types,
        "native_summary": {k: v for k, v in summary.items() if k != "odds_model_meta"},
        "settled_only": summary["pending_adopted_races"] > 0,
        "original_issuance_asof_verified": False,
    }


def main() -> None:
    """Print aggregates only; never publish raw encrypted rows or the key."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {f.name: summarize(f) for f in sorted(args.inputs.iterdir()) if f.is_dir()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
