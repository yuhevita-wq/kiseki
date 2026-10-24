"""Run unchanged Nyanta generation, adoption dry-run and official settlement.

Only storage, encrypted artifact restoration, and an explicit replay clock are
adapted. No model probability, purchase rule, stake or grade is fabricated.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_core import crypt
from scripts.reconstruct_odds import unpack


def restore_models(core: Path, odds: Path) -> dict:
    """Restore genuine learned artifacts and retain their truthful fit dates."""
    from src.models.trainer import load_model, save_model

    models = ROOT / "data/models"
    for name in ("lgbm_wt_eval", "lgbm_wt_win"):
        found = list(core.rglob(name + ".tar.gpg"))
        if len(found) != 1:
            raise RuntimeError("core_artifact_missing_or_ambiguous")
        unpack(found[0], models, {name + ".pkl", name + ".meta.json"})
        meta = json.loads((models / (name + ".meta.json")).read_text())
        valid = (
            meta.get("full_refit") and meta.get("to") == "2025-12-31"
            if name.endswith("win")
            else not meta.get("full_refit") and meta.get("test_from") == "2026-01-01"
        )
        if not valid:
            raise RuntimeError("model_is_not_frozen_before_2026")
        frozen = name + "_m2601"
        save_model(load_model(name), frozen)
        (models / (frozen + ".meta.json")).write_text(json.dumps(meta))
    parts = []
    for n in (7, 9):
        found = list(odds.rglob(f"odds-tf-n{n}.tar.gpg"))
        if len(found) != 1:
            raise RuntimeError("odds_artifact_missing_or_ambiguous")
        unpack(found[0], models, {f"odds_tf_n{n}.txt", "odds_tf_meta.json"})
        meta = json.loads((models / "odds_tf_meta.json").read_text())
        if meta["per_n_car"][str(n)]["train_end"] != "2025-12-31":
            raise RuntimeError("odds_model_fit_end_is_not_2025_end")
        parts.append(meta)
    if parts[0]["feature_names"] != parts[1]["feature_names"]:
        raise RuntimeError("odds_feature_definition_mismatch")
    # Same union and top-level n9 values as sequential original n7 -> n9 main.
    meta = dict(parts[1])
    for field in ("target_sum", "per_n_car"):
        meta[field] = {**parts[0][field], **parts[1][field]}
    meta["train_end"] = max(p["train_end"] for p in parts)
    (models / "odds_tf_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def migrate_replay_storage(path: Path) -> None:
    """Apply original Alembic columns to an isolated SQLite replay database."""
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    engine = sa.create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))

        class LocalSchema:
            """Translate PostgreSQL schema qualification; skip existing columns."""

            def add_column(self, table, column, schema=None):
                cols = {c["name"] for c in sa.inspect(conn).get_columns(table)}
                if column.name not in cols:
                    operations.add_column(table, column)

            def create_index(self, name, table, columns, schema=None):
                names = {i["name"] for i in sa.inspect(conn).get_indexes(table)}
                if name not in names:
                    operations.create_index(name, table, columns)

        for filename in (
            "202608141000_keirin_add_cup_grade.py",
            "202608272200_keirin_type_lab_outcome.py",
        ):
            source = ROOT.parent / "backend/alembic/versions" / filename
            spec = importlib.util.spec_from_file_location("replay_migration", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.op = LocalSchema()
            module.upgrade()
    engine.dispose()


def load_auxiliary(path: Path, cups: Path, payoffs: Path) -> None:
    """Load actual recovered grades and official payouts into isolated storage."""
    with sqlite3.connect(path) as dest:
        for encrypted in sorted(cups.rglob("cup-metadata-*.json.gpg")):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "cup.json"
                crypt(encrypted, p, decrypt=True)
                data = json.loads(p.read_text())
            if data["missing"]:
                raise RuntimeError("cup_metadata_incomplete")
            for r in data["records"]:
                dest.execute(
                    "UPDATE wt_races SET cup_grade=?,cup_name=? WHERE cup_id=? AND venue_id=?",
                    (r["cup_grade"], r["cup_name"], r["cup_id"], r["venue_id"]),
                )
        for encrypted in sorted(payoffs.rglob("official-payoffs-*.sqlite.gpg")):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "payoffs.sqlite"
                crypt(encrypted, p, decrypt=True)
                with sqlite3.connect(p) as source:
                    dest.executemany(
                        "INSERT INTO wt_payoffs(race_key,bet_type,combination,payoff) VALUES (?,?,?,?) ON CONFLICT(race_key,bet_type,combination) DO UPDATE SET payoff=excluded.payoff",
                        source.execute(
                            "SELECT race_key,bet_type,combination,payoff FROM wt_payoffs"
                        ),
                    )
        # Captured real runtime switches; no rank is enabled by this bridge.
        flags = json.loads(
            (ROOT / "config/nyanta_original_runtime_flags.json").read_text()
        )
        for r in flags["settings"]:
            dest.execute(
                "INSERT INTO netkeirin_settings(rank_key,enabled,axis_gate_enabled) VALUES (?,?,?) ON CONFLICT(rank_key) DO UPDATE SET enabled=excluded.enabled,axis_gate_enabled=excluded.axis_gate_enabled",
                (r["rank_key"], r["enabled"], r["axis_gate_enabled"]),
            )


def main() -> None:
    """Execute genuine model inference and native purchase code, never publish."""
    ap = argparse.ArgumentParser()
    for name in (
        "checkpoint",
        "core-models",
        "odds-models",
        "cups",
        "payoffs",
        "output",
    ):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    args = ap.parse_args()
    if not ("2026-01-01" <= args.start <= args.end <= "2026-10-23"):
        raise RuntimeError("invalid_heldout_period")
    from src import database

    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(args.checkpoint, database.DB_PATH, decrypt=True)
    database.init_db()
    database.migrate_db()
    migrate_replay_storage(database.DB_PATH)
    load_auxiliary(database.DB_PATH, args.cups, args.payoffs)
    odds_meta = restore_models(args.core_models, args.odds_models)

    from src.preprocessing.feature_wt import prepare_X

    from scripts import build_type_lab_picks as build
    from scripts import netkeirin_submit_type_lab as adoption
    from scripts import netkeirin_submit_wt as shared
    from scripts import settle_type_lab_picks as settle

    original_connection = database.get_connection
    replay_clock = [
        datetime.fromisoformat(args.start).replace(
            tzinfo=timezone(timedelta(hours=9)), hour=5
        )
    ]

    @contextlib.contextmanager
    def replay_connection():
        with original_connection() as conn:
            conn.create_function("NOW", 0, lambda: replay_clock[0].isoformat())
            yield conn

    class ReplayTime(datetime):
        """Explicit hypothetical 05:00 JST replay clock, not original asof proof."""

        @classmethod
        def now(cls, tz=None):
            return replay_clock[0].astimezone(tz) if tz else replay_clock[0]

    # Preserve the native close-window function and pass it an explicit clock.
    shared.datetime = ReplayTime
    build.get_connection = replay_connection
    candidates, accepted, skipped = [], [], []
    original_submit = adoption.submit_row
    original_skip = adoption._make_skip

    def capture_submit(row, session, client, dry_run, **kwargs):
        if not dry_run or client is not None:
            raise RuntimeError("reconstruction_is_dry_run_only")
        result = original_submit(row, session, client, dry_run, **kwargs)
        if result[0]:
            accepted.append(dict(row))
        return result

    def capture_skip(dry_run):
        native = original_skip(dry_run)

        def record(*a, **kw):
            skipped.append(
                {"race_key": a[0], "plan_key": a[1], "reason": a[3], "detail": a[4]}
            )
            return native(*a, **kw)

        return record

    adoption.submit_row = capture_submit
    adoption._make_skip = capture_skip
    # Construct this exact period using the original function. Populate its own
    # per-day cache only after checking all raw terms exist (the remaining
    # period-wide median fill could otherwise differ across inference ranges).
    feats, _ = build.build_day_features(args.start, args.end)
    if feats is None or feats.term.isna().any():
        raise RuntimeError("period_cache_cannot_preserve_original_term_fill")
    day = datetime.fromisoformat(args.start)
    while day.isoformat()[:10] <= args.end:
        text_day = day.isoformat()[:10]
        replay_clock[0] = day.replace(tzinfo=timezone(timedelta(hours=9)), hour=5)
        f = feats[feats.race_date.astype(str) == text_day].copy()
        build._FEATURE_CACHE[(text_day, None)] = (f, prepare_X(f))
        with database.get_connection() as c:
            if c.execute(
                "SELECT COUNT(*) FROM wt_races WHERE race_date=? AND cup_grade IS NULL",
                (text_day,),
            ).fetchone()[0]:
                raise RuntimeError("actual_cup_grade_missing_for_replay_day")
        for n in (7, 9):
            build.N_ENTRIES, build.MODE_TAG = n, "" if n == 7 else "9"
            rows = build.run_live(text_day, "lgbm_wt_eval_m2601", "lgbm_wt_win_m2601")
            build.save(rows)
            candidates.extend(rows)
        adoption.run(text_day, "morning", True, None, False)
        day += timedelta(days=1)
    # Use the actual native official-payout/tie/void settlement logic. --no-fetch
    # prevents an independent recollection while a payout collection is active.
    sys.argv = [
        "settle_type_lab_picks.py",
        "--from",
        args.start,
        "--to",
        args.end,
        "--no-fetch",
    ]
    settle.main()
    with database.get_connection() as c:
        settled = {
            (r["race_key"], r["plan_key"], r["mode"]): dict(r)
            for r in c.execute(
                "SELECT race_key,plan_key,mode,settled_at,hit,payout,void_refund FROM type_lab_picks WHERE race_date BETWEEN ? AND ?",
                (args.start, args.end),
            )
        }
    for row in accepted:
        row["settlement"] = settled[(row["race_key"], row["plan_key"], row["mode"])]
    done = [r for r in accepted if r["settlement"]["settled_at"] is not None]
    amount = sum(sum(int(l["stake"]) for l in r["legs"]) for r in done)
    returned = sum(
        (r["settlement"]["payout"] or 0) + (r["settlement"]["void_refund"] or 0)
        for r in done
    )
    summary = {
        "period": [args.start, args.end],
        "candidate_rows": len(candidates),
        "candidate_races": len({r["race_key"] for r in candidates}),
        "type_counts": dict(
            Counter({r["race_key"]: r["type_label"] for r in candidates}.values())
        ),
        "adopted_races": len(accepted),
        "settled_adopted_races": len(done),
        "pending_adopted_races": len(accepted) - len(done),
        "hit_rate_settled": sum(bool(r["settlement"]["hit"]) for r in done) / len(done)
        if done
        else None,
        "stake_settled": amount,
        "official_return_settled": returned,
        "roi_settled": returned / amount if amount else None,
        "replay_clock": "05:00 JST each historical day; hypothetical single morning wave",
        "original_issuance_asof_verified": False,
        "system_rules_were_researched_on_2026_results": True,
        "performance_claim": "Retrospective reconstruction replay only; not a proven point-in-time original-output reproduction or fully unseen system ROI",
        "odds_model_meta": odds_meta,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runtime-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "runtime.json"
        p.write_text(
            json.dumps(
                {"candidates": candidates, "accepted": accepted, "skips": skipped},
                ensure_ascii=False,
                default=str,
            )
        )
        crypt(p, args.output / "runtime-rows.json.gpg")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "odds_model_meta"},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
