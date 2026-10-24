"""Audit encrypted source shards without publishing plaintext or secrets.

This module changes no prediction rules. Run on the data repository's Actions
runner, where COLLECTOR_PASSPHRASE already exists. Only aggregate JSON is public.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

SOURCE_REPO = "yuhevita-wq/keirin"
LAST_DAY = date(2026, 10, 23)


def api(path: str) -> Any:
    """Read GitHub JSON, never include credentials in exceptions or output."""
    p = subprocess.run(["gh", "api", path], capture_output=True, check=False)
    if p.returncode:
        raise RuntimeError("github_read_failed")
    return json.loads(p.stdout)


def stats(conn: sqlite3.Connection, day: str) -> dict[str, Any]:
    """Inspect real schema, completeness, uniqueness and acquisition times."""
    out: dict[str, Any] = {"date": day}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    out["tables"] = sorted(tables)
    out["integrity"] = conn.execute("PRAGMA quick_check").fetchone()[0]
    for table in ("wt_races", "wt_entries", "wt_odds", "wt_payoffs", "venue_info", "wt_odds_snapshot"):
        if table not in tables:
            out[table] = {"exists": False}
            continue
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
        item: dict[str, Any] = {"exists": True, "rows": conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0], "columns": cols}
        item["nulls"] = {c: conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{c}" IS NULL').fetchone()[0] for c in cols if c not in ("id", "comment", "name")}
        if "race_key" in cols:
            item["race_count"] = conn.execute(f'SELECT COUNT(DISTINCT race_key) FROM "{table}"').fetchone()[0]
        keys = {"wt_races": "race_key", "wt_entries": "race_key,frame_no", "wt_odds": "race_key,bet_type,combination"}.get(table)
        if keys:
            item["duplicate_keys"] = conn.execute(f'SELECT COUNT(*) FROM (SELECT {keys} FROM "{table}" GROUP BY {keys} HAVING COUNT(*)>1)').fetchone()[0]
        for col in ("created_at", "collected_at", "snapshot_at"):
            if col in cols:
                item[col + "_range"] = list(conn.execute(f'SELECT MIN("{col}"), MAX("{col}") FROM "{table}"').fetchone())
        if table == "wt_entries":
            item["results_positive"] = conn.execute("SELECT COUNT(*) FROM wt_entries WHERE finish_order>=1").fetchone()[0]
            item["dnf_zero"] = conn.execute("SELECT COUNT(*) FROM wt_entries WHERE finish_order=0").fetchone()[0]
            item["zero_scores"] = conn.execute("SELECT COUNT(*) FROM wt_entries WHERE race_point=0").fetchone()[0]
            item["styles"] = dict(conn.execute("SELECT style,COUNT(*) FROM wt_entries GROUP BY style"))
            item["usable_core_races"] = conn.execute("""SELECT COUNT(*) FROM (
              SELECT e.race_key FROM wt_entries e JOIN wt_races r USING(race_key)
              GROUP BY e.race_key HAVING COUNT(*)>=3
              AND SUM(e.finish_order IS NULL OR e.race_point IS NULL OR e.gear_ratio IS NULL
                OR e.style IS NULL OR e.player_id IS NULL OR e.first_rate IS NULL
                OR e.third_rate IS NULL OR e.line_group IS NULL OR e.line_pos IS NULL
                OR e.prediction_mark IS NULL)=0)""").fetchone()[0]
            item["complete_type_input_races"] = conn.execute("""SELECT COUNT(*) FROM (
              SELECT e.race_key FROM wt_entries e JOIN wt_races r USING(race_key)
              WHERE COALESCE(r.cancel,0)=0 GROUP BY e.race_key HAVING COUNT(*) IN (7,9)
              AND SUM(e.ex_left_behind_pct IS NULL OR e.race_point IS NULL OR e.style IS NULL
                  OR e.line_group IS NULL OR e.line_pos IS NULL OR r.day_index IS NULL)=0
              AND COUNT(DISTINCT e.line_group)>1 AND MAX(e.prediction_mark)>0)""").fetchone()[0]
        if table == "wt_odds":
            item["by_bet_type"] = dict(conn.execute("SELECT bet_type,COUNT(*) FROM wt_odds GROUP BY bet_type"))
            item["invalid_values"] = conn.execute("SELECT COUNT(*) FROM wt_odds WHERE odds_value IS NULL OR odds_value<=0").fetchone()[0]
        out[table] = item
    out["race_dates"] = [r[0] for r in conn.execute("SELECT DISTINCT race_date FROM wt_races ORDER BY race_date")]
    out["races_without_entries"] = conn.execute("SELECT COUNT(*) FROM wt_races r WHERE NOT EXISTS(SELECT 1 FROM wt_entries e WHERE e.race_key=r.race_key)").fetchone()[0]
    out["entry_count_mismatches"] = conn.execute("SELECT COUNT(*) FROM wt_races r WHERE r.n_entries<>(SELECT COUNT(*) FROM wt_entries e WHERE e.race_key=r.race_key)").fetchone()[0]
    out["wrong_day_races"] = conn.execute("SELECT COUNT(*) FROM wt_races WHERE race_date<>?", (day,)).fetchone()[0]
    # Entry fields have no snapshot timestamp. created_at proves acquisition only,
    # never that mutable extended stats were valid before the original race.
    out["pre_race_entry_snapshot_proven"] = False
    out["behind_asof_original_morning_proven"] = False
    return out


def month_audit(month: str, output: Path) -> None:
    """Decrypt every available day of a month inside an ephemeral runner."""
    secret = os.environ.get("COLLECTOR_PASSPHRASE", "")
    if len(secret) < 32:
        raise RuntimeError("existing_actions_secret_unavailable")
    y, m = map(int, month.split("-"))
    days = [date(y, m, d).isoformat() for d in range(1, calendar.monthrange(y, m)[1]+1) if date(y,m,d)<=LAST_DAY]
    out: dict[str, Any] = {"month": month, "expected_days": len(days), "days": [], "errors": []}
    try:
        rel = api(f"repos/{SOURCE_REPO}/releases/tags/training-data-{month}")
        assets = {a["name"]: a for a in rel["assets"]}
    except RuntimeError:
        assets = {}
        out["release_lookup_failed"] = True
    with tempfile.TemporaryDirectory(prefix="nyanta-audit-") as tmp:
        for day in days:
            a = assets.get(f"winticket-{day}.sqlite.gpg")
            if not a:
                out["errors"].append({"date": day, "error": "release_asset_missing"})
                continue
            enc, db = Path(tmp)/"day.gpg", Path(tmp)/"day.sqlite"
            try:
                urllib.request.urlretrieve(a["browser_download_url"], enc)
                encrypted_sha = hashlib.sha256(enc.read_bytes()).hexdigest()
                p = subprocess.run(["gpg", "--batch", "--yes", "--pinentry-mode", "loopback", "--passphrase-fd", "0", "--output", str(db), "--decrypt", str(enc)], input=secret.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if p.returncode:
                    out["errors"].append({"date": day, "error": "decryption_failed"})
                    continue
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                    item = stats(conn, day)
                item["encrypted_sha256"] = encrypted_sha
                item["release_asset_id"] = a["id"]
                out["days"].append(item)
            except Exception as exc:
                out["errors"].append({"date": day, "error": type(exc).__name__})
            finally:
                enc.unlink(missing_ok=True)
                db.unlink(missing_ok=True)
            print(f"audited {day}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"month": month, "audited_days": len(out["days"]), "errors": len(out["errors"])}))


def main() -> None:
    """CLI for isolated monthly audit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    month_audit(args.month, args.output)


if __name__ == "__main__":
    main()
