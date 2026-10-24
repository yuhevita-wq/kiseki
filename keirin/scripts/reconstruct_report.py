"""Aggregate actual native replay artifacts without treating pending as losses."""

from __future__ import annotations

import argparse
import calendar
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    """Require ten non-overlapping monthly results and report their real totals."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    files = sorted(args.results.rglob("runtime-summary.json"))
    if len(files) != 10:
        raise RuntimeError("all_ten_actual_monthly_replay_summaries_required")
    reports = [json.loads(p.read_text()) for p in files]
    periods = {tuple(r["period"]) for r in reports}
    expected = {
        (
            f"2026-{m:02d}-01",
            f"2026-{m:02d}-{23 if m == 10 else calendar.monthrange(2026, m)[1]:02d}",
        )
        for m in range(1, 11)
    }
    if periods != expected:
        raise RuntimeError("monthly_period_gap_or_overlap")
    if any(r["original_issuance_asof_verified"] for r in reports):
        raise RuntimeError("unexpected_issuance_asof_claim")
    types = Counter()
    for r in reports:
        types.update(r["type_counts"])
    totals = {
        key: sum(r[key] for r in reports)
        for key in (
            "candidate_rows",
            "candidate_races",
            "adopted_races",
            "settled_adopted_races",
            "pending_adopted_races",
            "stake_settled",
            "official_return_settled",
        )
    }
    hits = sum(
        round(r["hit_rate_settled"] * r["settled_adopted_races"])
        for r in reports
        if r["settled_adopted_races"]
    )
    settled = totals["settled_adopted_races"]
    amount = totals["stake_settled"]
    output = {
        "period": ["2026-01-01", "2026-10-23"],
        "months": 10,
        "actual_totals": totals,
        "type_counts": dict(types),
        "settled_hits": hits,
        "hit_rate_settled_only": hits / settled if settled else None,
        "roi_settled_only": totals["official_return_settled"] / amount
        if amount
        else None,
        "all_adopted_rows_settled": totals["pending_adopted_races"] == 0,
        "pending_warning": "Pending payouts/results are excluded from the settled-only denominator. Missing payouts can be biased toward winners; do not label partial ROI final.",
        "original_issuance_asof_verified": False,
        "system_rules_were_researched_on_2026_results": True,
        "formal_point_in_time_system_roi_proven": False,
        "months_detail": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in output.items() if k != "months_detail"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
