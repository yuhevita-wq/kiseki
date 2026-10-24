"""Storage-only bridge to the original Nyanta trifecta odds trainer.

Original feature builders, five-seed inputs, regression parameters, board filters,
sampling, normalization and metadata creation remain in their original modules.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pickle
import sqlite3
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_audit import SOURCE_REPO, api
from scripts.reconstruct_core import crypt, save_database_checkpoint


class ReadCursor:
    """Translate the two original read-only PostgreSQL queries for real SQLite."""

    def __init__(self, cursor):
        self.raw = cursor

    def execute(self, sql, params=None):
        sql = sql.replace("keirin.wt_entries", "wt_entries").replace(
            "keirin.wt_odds", "wt_odds"
        )
        # wt_odds has one row per race/market/combination (audited natural key).
        # DISTINCT ON therefore changes no selection on these source tables.
        sql = sql.replace("SELECT DISTINCT ON (race_key, combination)", "SELECT")
        if "ANY(%s)" not in sql or params is None or len(params) != 1:
            raise RuntimeError("unexpected_original_odds_query")
        keys = list(params[0])
        sql = sql.replace("= ANY(%s)", "IN (" + ",".join("?" for _ in keys) + ")")
        self.raw.execute(sql, keys)
        return self

    def __getattr__(self, name):
        return getattr(self.raw, name)


class ReadConnection:
    """DB-API facade; never changes values, rows, features or model parameters."""

    def __init__(self, path: Path):
        self.raw = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def cursor(self):
        return ReadCursor(self.raw.cursor())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.raw.close()

    def __getattr__(self, name):
        return getattr(self.raw, name)


def unpack(encrypted: Path, dest: Path, allowed: set[str]) -> None:
    """Restore only explicitly expected files from our encrypted job artifacts."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "archive.tar"
        crypt(encrypted, path, decrypt=True)
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.name not in allowed or not member.isfile():
                    raise RuntimeError("unexpected_artifact_member")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError("invalid_artifact_member")
                (dest / member.name).write_bytes(source.read())


def prepare(args) -> None:
    """Combine genuine WF/frozen predictions and copy only real requested odds."""
    import pandas as pd
    from src import database
    from src.models.trainer import load_model
    from src.preprocessing.feature_wt import FEATURE_COLS_WT, prepare_X

    from scripts import train_odds_prediction_tf as original

    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(args.checkpoint, database.DB_PATH, decrypt=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.pkl"
        crypt(args.features, path, decrypt=True)
        features = pd.read_pickle(path)
    assert len(FEATURE_COLS_WT) == 70
    models = ROOT / "data/models"
    for name in ("lgbm_wt_eval", "lgbm_wt_win"):
        found = list(args.core_models.rglob(name + ".tar.gpg"))
        if len(found) != 1:
            raise RuntimeError("original_input_core_model_missing")
        unpack(found[0], models, {name + ".pkl", name + ".meta.json"})
        meta = json.loads((models / (name + ".meta.json")).read_text())
        expected = int(
            ((features.race_date < "2026-01-01") & features.finish_order.notna()).sum()
        )
        if meta["fit_rows"] != expected:
            raise RuntimeError("core_model_not_fit_on_confirmed_original_input_corpus")
        if name.endswith("win"):
            valid = meta["full_refit"] and meta["to"] == "2025-12-31"
        else:
            valid = not meta["full_refit"] and meta["test_from"] == "2026-01-01"
        if not valid:
            raise RuntimeError("core_model_not_frozen_before_2026")
    frames = []
    caches = sorted(args.wf_caches.rglob("wf-n*.pkl.gpg"))
    if len(caches) != 12:
        raise RuntimeError("all_twelve_original_walkforward_windows_required")
    for encrypted in caches:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wf.pkl"
            crypt(encrypted, p, decrypt=True)
            f = pd.read_pickle(p)
        if str(f.race_date.max()) > "2025-12-31":
            raise RuntimeError("unexpected_2026_walkforward_fit_input")
        frames.append(f)
    # 2026 is evaluated using the genuinely frozen <=2025 core models. It is
    # never used to fit another core model for the odds validation inputs.
    hold = features[features.race_date >= "2026-01-01"].copy()
    scores = hold[["race_key", "frame_no", "race_date"]].copy()
    X = prepare_X(hold)
    scores["pp3"] = load_model("lgbm_wt_eval").predict_proba(X)[:, 1]
    scores["ppw"] = load_model("lgbm_wt_win").predict_proba(X)[:, 1]
    frames.append(scores)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(["race_key", "frame_no"]).any():
        raise RuntimeError("overlapping_walkforward_prediction_rows")
    sizes = dict(zip(features.race_key, features._n_entries))
    exp = ROOT / "data/exp_cache"
    exp.mkdir(parents=True, exist_ok=True)
    # Match the original axis_detail_7car cache eligibility before the native
    # trainer performs its unchanged max-races selection. Results are used only
    # for historical cache eligibility, never as probability features.
    finish = features[features.finish_order.between(1, 3)]
    top3 = finish.groupby("race_key").frame_no.agg(lambda x: frozenset(x.astype(int)))
    details = []
    for rk, g in predictions.groupby("race_key", sort=True):
        if sizes.get(rk) != 7 or len(g) != 7 or len(top3.get(rk, ())) != 3:
            continue
        details.append(
            {
                "rk": str(rk),
                "date": str(g.race_date.iloc[0]),
                "p3": dict(zip(g.frame_no.astype(int), g.pp3.astype(float))),
                "pw": dict(zip(g.frame_no.astype(int), g.ppw.astype(float))),
            }
        )
    (exp / "axis_detail_7car.pkl").write_bytes(pickle.dumps(details))
    nine = predictions[predictions.race_key.map(sizes) == 9].copy()
    nine.to_pickle(exp / "wf_preds9_reconstruction.pkl")
    selected = set()
    for n in (7, 9):
        original.N_CAR = n
        keys = sorted(original._load_wf_preds())
        # Fetch all genuine candidate boards first. Sampling is left entirely
        # to the unchanged trainer, after original seven-car cache eligibility.
        selected.update(keys)
    by_day = {}
    for rk in sorted(selected):
        compact = rk[:8]
        day = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
        by_day.setdefault(day, []).append(rk)
    assets = {}
    for month in sorted({d[:7] for d in by_day}):
        rel = api(f"repos/{SOURCE_REPO}/releases/tags/training-data-{month}")
        assets.update({a["name"]: a for a in rel["assets"]})
    with sqlite3.connect(database.DB_PATH) as c:
        if c.execute("SELECT COUNT(*) FROM wt_odds").fetchone()[0]:
            raise RuntimeError("odds_preparation_requires_empty_original_odds_table")
    copied = 0
    winning_trio_present = set()
    days = sorted(by_day)
    with tempfile.TemporaryDirectory() as tmp:

        def fetch(day):
            a = assets[f"winticket-{day}.sqlite.gpg"]
            encrypted, db = Path(tmp) / (day + ".gpg"), Path(tmp) / (day + ".sqlite")
            urllib.request.urlretrieve(a["browser_download_url"], encrypted)
            crypt(encrypted, db, decrypt=True)
            encrypted.unlink()
            return day, db

        for i in range(0, len(days), 16):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for day, db in pool.map(fetch, days[i : i + 16]):
                    keys = by_day[day]
                    with (
                        sqlite3.connect(db) as source,
                        sqlite3.connect(database.DB_PATH) as dest,
                    ):
                        rows = source.execute(
                            "SELECT race_key,bet_type,combination,odds_value,collected_at FROM wt_odds WHERE bet_type IN ('trifecta','trio') AND race_key IN ("
                            + ",".join("?" for _ in keys)
                            + ")",
                            keys,
                        ).fetchall()
                        dest.executemany(
                            "INSERT INTO wt_odds(race_key,bet_type,combination,odds_value,collected_at) VALUES (?,?,?,?,?)",
                            rows,
                        )
                    # Reuse the original parser/filter, including its exact
                    # 0 < odds < 9999 criterion and combination parsing.
                    from scripts.exp_7s7a_payout_structure import load_trio

                    old_pg = load_trio.__globals__["_pg"]
                    load_trio.__globals__["_pg"] = lambda: ReadConnection(
                        database.DB_PATH
                    )
                    try:
                        boards = load_trio(keys)
                    finally:
                        load_trio.__globals__["_pg"] = old_pg
                    winning_trio_present.update(
                        rk for rk, board in boards.items() if top3.get(rk) in board
                    )
                    copied += len(rows)
                    db.unlink()
                    print(f"real odds {day}: {len(rows)} rows", flush=True)
    eligible_details = [d for d in details if d["rk"] in winning_trio_present]
    (exp / "axis_detail_7car.pkl").write_bytes(pickle.dumps(eligible_details))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    manifest["real_trifecta_and_trio_odds_rows_copied"] = copied
    manifest["odds_requested_races"] = len(selected)
    save_database_checkpoint(
        database.DB_PATH, args.output / "odds-input.sqlite.gpg", manifest
    )
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "predictions.tar"
        with tarfile.open(tar, "w") as archive:
            for name in ("axis_detail_7car.pkl", "wf_preds9_reconstruction.pkl"):
                archive.add(exp / name, arcname=name)
        crypt(tar, args.output / "original-wf-inputs.tar.gpg")
    (args.output / "odds-input-summary.json").write_text(
        json.dumps(
            {
                "original_regression_trainer": "scripts.train_odds_prediction_tf.main",
                "requested_races": len(selected),
                "real_odds_rows": copied,
                "default_max_races_per_car_count": 12000,
                "seven_car_original_cache_candidates": len(details),
                "seven_car_original_cache_eligible": len(eligible_details),
                "cache_eligibility_source": "exp_7s7a_axis_and_band.build_detail: seven rows, exactly three top3 cars, winning trio in original load_trio board",
                "sampling": "only unchanged original build_dataset performs sampling",
                "validation_extension": "real archive through 2026-10-23; original legacy cache ended 2026-08-04",
                "2026_core_fit": False,
                "original_feature_count": 70,
                "odds_feature_count": len(original.FEATURE_NAMES),
            },
            indent=2,
        )
        + "\n"
    )


def train(args) -> None:
    """Execute the unchanged original odds trainer with real source data."""
    from src import database

    from scripts import train_odds_prediction_tf as original

    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(args.checkpoint, database.DB_PATH, decrypt=True)
    unpack(
        args.predictions,
        ROOT / "data/exp_cache",
        {"axis_detail_7car.pkl", "wf_preds9_reconstruction.pkl"},
    )
    original._connect = lambda: ReadConnection(database.DB_PATH)
    sys.argv = [
        "train_odds_prediction_tf.py",
        "--n-car",
        str(args.n_car),
        "--train-end",
        "2025-12-31",
    ]
    original.main()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(original.META_PATH.read_text())
    (args.output / f"odds-tf-n{args.n_car}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "odds.tar"
        with tarfile.open(tar, "w") as archive:
            for p in (
                original.MODEL_DIR / f"odds_tf_n{args.n_car}.txt",
                original.META_PATH,
            ):
                archive.add(p, arcname=p.name)
        crypt(tar, args.output / f"odds-tf-n{args.n_car}.tar.gpg")


def main() -> None:
    """CLI for real-data preparation and the original n7/n9 trainers."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train"])
    for name in (
        "checkpoint",
        "manifest",
        "features",
        "core-models",
        "wf-caches",
        "predictions",
    ):
        ap.add_argument("--" + name, type=Path)
    ap.add_argument("--n-car", type=int, choices=[7, 9], default=7)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
