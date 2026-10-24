"""Connect encrypted real data to the unchanged original 70-feature trainer.

Only storage adaptation, input validation and measurement live here. Model,
feature engineering and parameters are imported from the pinned original code.
No odds model, purchase rule or provisional prediction is invented here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_audit import SOURCE_REPO, api

_GPG_LOCK = threading.Lock()


def crypt(source: Path, target: Path, decrypt: bool = False) -> None:
    """Use the existing secret over stdin; never put it in arguments/logs."""
    secret = os.environ.get("COLLECTOR_PASSPHRASE", "")
    if len(secret) < 32:
        raise RuntimeError("existing_actions_secret_unavailable")
    cmd = [
        "gpg",
        "--batch",
        "--yes",
        "--pinentry-mode",
        "loopback",
        "--passphrase-fd",
        "0",
        "--no-symkey-cache",
        "--output",
        str(target),
    ]
    cmd += ["--decrypt"] if decrypt else ["--symmetric", "--cipher-algo", "AES256"]
    # gpg-agent initializes shared state; concurrent first-use can race even
    # though archive names differ. Parallelize downloads, serialize crypto.
    with _GPG_LOCK:
        for attempt in range(3):
            p = subprocess.run(
                cmd + [str(source)],
                input=secret.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if p.returncode == 0:
                return
            time.sleep(attempt + 1)
    raise RuntimeError("decryption_failed" if decrypt else "encryption_failed")


def merge_day(source: Path, dest: sqlite3.Connection) -> dict[str, Any]:
    """Merge with natural keys; daily integer surrogate IDs must not collide."""
    counts: dict[str, Any] = {}
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
        for table in ("wt_races", "wt_entries", "venue_info"):
            cols = [
                r[1]
                for r in src.execute(f'PRAGMA table_info("{table}")')
                if r[1] != "id"
            ]
            if not cols:
                raise RuntimeError("required_source_table_missing")
            names = ",".join('"' + c + '"' for c in cols)
            rows = src.execute(f'SELECT {names} FROM "{table}"').fetchall()
            # Only static venue metadata repeats. A conflicting repeated venue is
            # an error; race/entry duplicates always error rather than overwrite.
            if table == "venue_info":
                for row in rows:
                    old = dest.execute(
                        f"SELECT {names} FROM venue_info WHERE venue_code=?",
                        (row[cols.index("venue_code")],),
                    ).fetchone()
                    if old is not None:
                        if tuple(old) != tuple(row):
                            raise RuntimeError("conflicting_venue_metadata")
                        continue
                    dest.execute(
                        f'INSERT INTO "{table}" ({names}) VALUES ({",".join("?" for _ in cols)})',
                        row,
                    )
            else:
                dest.executemany(
                    f'INSERT INTO "{table}" ({names}) VALUES ({",".join("?" for _ in cols)})',
                    rows,
                )
            counts[table] = len(rows)
    dest.commit()
    return counts


def save_database_checkpoint(db: Path, output: Path, report: dict[str, Any]) -> None:
    """Capture committed WAL frames with SQLite backup, verify, then encrypt."""
    expected = {
        t: sum(d["rows"][t] for d in report["days"]) for t in ("wt_races", "wt_entries")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nyanta-sealed-") as tmp:
        snapshot = Path(tmp) / "snapshot.sqlite"
        src, dst = sqlite3.connect(db), sqlite3.connect(snapshot)
        try:
            src.backup(dst)
            actual = {
                t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in expected
            }
            if actual != expected:
                raise RuntimeError("checkpoint_row_counts_do_not_match_source_manifest")
            report["verified_checkpoint_counts"] = actual
            report["verified_checkpoint_date_range"] = list(
                dst.execute(
                    "SELECT MIN(race_date),MAX(race_date) FROM wt_races"
                ).fetchone()
            )
        finally:
            src.close()
            dst.close()
        encrypted_tmp = output.with_suffix(".gpg.tmp")
        crypt(snapshot, encrypted_tmp)
        restored = Path(tmp) / "restored.sqlite"
        crypt(encrypted_tmp, restored, decrypt=True)
        verifier = sqlite3.connect(restored)
        try:
            verified = {
                t: verifier.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in expected
            }
            if verified != expected:
                raise RuntimeError("encrypted_checkpoint_roundtrip_mismatch")
            report["encrypted_roundtrip_verified"] = True
        finally:
            verifier.close()
        encrypted_tmp.replace(output)
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def prepare(output: Path) -> None:
    """Download existing Releases only, merge all 1,392 days, encrypt checkpoint."""
    from src import database

    database.DB_PATH = ROOT / "data" / "keirin.db"
    resumed = None
    manifest = output.with_suffix(".manifest.json")
    if output.exists() and manifest.exists():
        resumed = json.loads(manifest.read_text())
        if resumed.get("source_repository") != SOURCE_REPO:
            raise RuntimeError("checkpoint_source_mismatch")
        database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        crypt(output, database.DB_PATH, decrypt=True)
    else:
        database.init_db()
        database.migrate_db()
        # Do not substitute current venue metadata for source metadata.
        with sqlite3.connect(database.DB_PATH) as dst:
            dst.execute("DELETE FROM venue_info")
    assets, missing = [], []
    d = date(2023, 1, 1)
    while d <= date(2026, 10, 23):
        month = d.strftime("%Y-%m")
        try:
            rel = api(f"repos/{SOURCE_REPO}/releases/tags/training-data-{month}")
            available = {a["name"]: a for a in rel["assets"]}
        except RuntimeError:
            available = {}
        while d <= date(2026, 10, 23) and d.strftime("%Y-%m") == month:
            a = available.get(f"winticket-{d}.sqlite.gpg")
            if a:
                assets.append((d.isoformat(), a))
            else:
                missing.append(d.isoformat())
            d += timedelta(days=1)
    if missing:
        raise RuntimeError(f"missing_release_days:{len(missing)}")
    report: dict[str, Any] = {
        "source_repository": SOURCE_REPO,
        "expected_days": 1392,
        "days": [],
        "surrogate_ids": "regenerated; race_key/frame_no preserved",
        "odds": "not copied in core checkpoint; retained in original Releases",
    }
    if resumed is not None:
        report = resumed
        done = {item["date"]: item for item in report["days"]}
        with sqlite3.connect(database.DB_PATH) as check:
            actual_days = {
                d: (nr, ne)
                for d, nr, ne in check.execute(
                    "SELECT r.race_date,COUNT(DISTINCT r.race_key),COUNT(e.race_key) FROM wt_races r LEFT JOIN wt_entries e USING(race_key) GROUP BY r.race_date"
                )
            }
            invalid = {
                d
                for d, item in done.items()
                if actual_days.get(d, (0, 0))
                != (item["rows"]["wt_races"], item["rows"]["wt_entries"])
            }
            for day in invalid:
                check.execute(
                    "DELETE FROM wt_entries WHERE substr(race_key,1,8)=?",
                    (day.replace("-", ""),),
                )
                check.execute("DELETE FROM wt_races WHERE race_date=?", (day,))
            check.commit()
        if invalid:
            report["repaired_incomplete_checkpoint_days"] = sorted(invalid)
            report["days"] = [
                item for item in report["days"] if item["date"] not in invalid
            ]
            done = {item["date"]: item for item in report["days"]}
        for day, asset in assets:
            if day in done and asset["id"] != done[day]["release_asset_id"]:
                raise RuntimeError("checkpoint_release_asset_changed")
        assets = [(day, a) for day, a in assets if day not in done]
        print(f"resumed {len(done)}/1392 days", flush=True)
    with tempfile.TemporaryDirectory(prefix="nyanta-core-") as tmp:

        def download(item: tuple[str, dict]) -> tuple[str, Path, dict]:
            day, a = item
            enc, db = Path(tmp) / (day + ".gpg"), Path(tmp) / (day + ".sqlite")
            urllib.request.urlretrieve(a["browser_download_url"], enc)
            checksum = hashlib.sha256(enc.read_bytes()).hexdigest()
            crypt(enc, db, decrypt=True)
            enc.unlink()
            return (
                day,
                db,
                {
                    "date": day,
                    "encrypted_sha256": checksum,
                    "release_asset_id": a["id"],
                },
            )

        # Bound batches, so futures cannot fill runner storage with a full corpus.
        for i in range(0, len(assets), 16):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for day, db, item in pool.map(download, assets[i : i + 16]):
                    with sqlite3.connect(database.DB_PATH) as dst:
                        item["rows"] = merge_day(db, dst)
                    db.unlink()
                    report["days"].append(item)
                    print(f"merged {day} ({len(report['days'])}/1392)", flush=True)
            # Persist a resumable encrypted checkpoint every 3 months, and finish.
            if (i + 16) % 96 == 0 or i + 16 >= len(assets):
                save_database_checkpoint(database.DB_PATH, output, report)
    if not assets:
        save_database_checkpoint(database.DB_PATH, output, report)
    print(
        json.dumps(
            {"merged_days": len(report["days"]), "encrypted_checkpoint": str(output)}
        )
    )


def train(
    target: str, checkpoint: Path, report_path: Path, do_fit: bool = True
) -> None:
    """Run the original CLI training callback; report independent 2026 metrics."""
    from src import database

    database.DB_PATH = ROOT / "data" / "keirin.db"
    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(checkpoint, database.DB_PATH, decrypt=True)
    import pandas as pd
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from src.cli.main import train_wt
    from src.models.trainer import load_model
    from src.preprocessing import feature_wt as f

    assert len(f.FEATURE_COLS_WT) == 70
    original_loader = f.load_raw_data_wt
    # User confirmed that ORIGINAL builtin imputation is part of reproduction.
    # Audit missing values, but do not add a new race-exclusion policy here.
    raw = original_loader("2023-01-01", "2026-10-23")
    required = [
        "race_point",
        "gear_ratio",
        "first_rate",
        "third_rate",
        "style",
        "player_class",
        "frame_no",
        "bank_length",
        "is_indoor",
        "grade",
        "term",
        "player_prefecture",
        "venue_prefecture",
        "player_id",
        "line_group",
        "line_size",
        "line_pos",
        "is_line_leader",
        "n_lines",
        "s_count",
        "h_count",
        "b_count",
        "prediction_mark",
        "race_type",
    ]
    null_bad = raw[required].isna().any(axis=1) | raw["race_point"].eq(0)
    bad_keys = set(raw.loc[null_bad, "race_key"])
    clean = raw.copy()

    def validated_loader(
        min_date: str = "2025-01-01", max_date: str | None = None
    ) -> pd.DataFrame:
        mask = clean.race_date.ge(min_date)
        if max_date:
            mask &= clean.race_date.le(max_date)
        return clean.loc[mask].copy()

    f.load_raw_data_wt = validated_loader
    model_name = "lgbm_wt_eval" if target == "top3" else "lgbm_wt_win"
    if not do_fit:
        metadata = json.loads(
            (ROOT / "data/models" / (model_name + ".meta.json")).read_text()
        )
        expected_fit_rows = int(
            ((clean.race_date < "2026-01-01") & clean.finish_order.notna()).sum()
        )
        if int(metadata["fit_rows"]) != expected_fit_rows:
            raise RuntimeError(
                "saved_model_training_rows_do_not_match_verified_training_corpus"
            )
    # win full-refits only <=2025-12-31, preserving the original full_refit mode
    # while never absorbing the external 2026 holdout into the saved model.
    if do_fit:
        train_wt.callback(
            from_date="2023-01-01",
            to_date="2025-12-31" if target == "win" else None,
            test_from=None if target == "win" else "2026-01-01",
            test_to=None if target == "win" else "2026-10-23",
            save_as=model_name,
            full_refit=target == "win",
            promote=False,
            target_kind=target,
            force_overwrite_vintage=False,
            dnf_weight=1.0,
        )
    feats = f.build_features_wt(validated_loader("2026-01-01", "2026-10-23"))
    feats = feats[feats.finish_order.notna()].copy()
    model = load_model(model_name)
    pred = model.predict_proba(f.prepare_X(feats))[:, 1]
    tcol = "top3_flag" if target == "top3" else "win_flag"
    truth = feats[tcol].to_numpy()
    bins = []
    for i in range(10):
        sel = (pred >= i / 10) & (pred < (i + 1) / 10 if i < 9 else pred <= 1)
        if sel.any():
            bins.append(
                {
                    "lower": i / 10,
                    "n": int(sel.sum()),
                    "pred": float(pred[sel].mean()),
                    "actual": float(truth[sel].mean()),
                }
            )
    fit_keys = set(
        clean.loc[
            (clean.race_date < "2026-01-01") & clean.finish_order.notna(), "race_key"
        ]
    )
    hold_keys = set(feats.race_key)
    assert not fit_keys & hold_keys
    report = {
        "model": model_name,
        "original_feature_count": 70,
        "original_feature_names": f.FEATURE_COLS_WT,
        "raw_races": int(raw.race_key.nunique()),
        "usable_races": int(
            clean.loc[clean.finish_order.notna(), "race_key"].nunique()
        ),
        "excluded_missing_input_races": 0,
        "raw_input_races_with_null_or_zero_score": len(bad_keys),
        "input_policy": "original builtin imputation unchanged; confirmed by user",
        "missing_raw_fields": {c: int(raw[c].isna().sum()) for c in required},
        "train_races": len(fit_keys),
        "holdout_races": len(hold_keys),
        "overlap_races": 0,
        "train_period": ["2023-01-01", "2025-12-31"],
        "holdout_period": ["2026-01-01", "2026-10-23"],
        "actual_holdout_date_range": [
            str(feats.race_date.min()),
            str(feats.race_date.max()),
        ],
        "model_refit_in_this_job": do_fit,
        "model_sha256": hashlib.sha256(
            (ROOT / "data/models" / (model_name + ".pkl")).read_bytes()
        ).hexdigest(),
        "auc": float(roc_auc_score(truth, pred)),
        "log_loss": float(log_loss(truth, pred)),
        "brier": float(brier_score_loss(truth, pred)),
        "calibration_bins": bins,
        "original_model_output_agreement": None,
        "source_start_difference": "2022-12 missing; source begins 2023-01-01",
        "provenance_limit": "Retrospective cards; original morning extended-stat snapshots absent. No original result field is directly added to the 70 inputs; original point-in-time history functions are used unchanged.",
        "odds_models": "not trained by core job",
        "purchase_outputs": "not generated by core job",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Models and all row-level derived data are encrypted before artifact upload.
    import tarfile

    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "models.tar"
        with tarfile.open(tar, "w") as archive:
            for path in (ROOT / "data/models").glob(model_name + ".*"):
                archive.add(path, arcname=path.name)
        crypt(tar, report_path.parent / (model_name + ".tar.gpg"))
    print(
        json.dumps(
            {
                "model": model_name,
                "train_races": len(fit_keys),
                "holdout_races": len(hold_keys),
                "auc": report["auc"],
            }
        )
    )


def main() -> None:
    """Explicit preparation/training stages; never publish a provisional model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "evaluate"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--target", choices=["top3", "win"], default="top3")
    ap.add_argument("--model-artifact", type=Path)
    ap.add_argument(
        "--report", type=Path, default=ROOT / "data/reconstruction/core-report.json"
    )
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.checkpoint)
    elif args.stage == "train":
        train(args.target, args.checkpoint, args.report)
    else:
        if args.model_artifact is None:
            raise RuntimeError("actual_trained_model_artifact_required")
        import tarfile

        model_name = "lgbm_wt_eval" if args.target == "top3" else "lgbm_wt_win"
        model_dir = ROOT / "data/models"
        model_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tar = Path(tmp) / "model.tar"
            crypt(args.model_artifact, tar, decrypt=True)
            with tarfile.open(tar) as archive:
                for member in archive.getmembers():
                    if (
                        member.name
                        not in (model_name + ".pkl", model_name + ".meta.json")
                        or not member.isfile()
                    ):
                        raise RuntimeError("unexpected_model_archive_member")
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("invalid_model_archive")
                    (model_dir / member.name).write_bytes(source.read())
        train(args.target, args.checkpoint, args.report, do_fit=False)


if __name__ == "__main__":
    main()
