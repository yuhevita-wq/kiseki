"""Prepare real 70-feature data and original walk-forward odds-training inputs.

The learning primitive, seeds, feature list and historical window definitions
are imported from the original repository. No purchase rule is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_core import crypt


def prepare(checkpoint: Path, output: Path) -> None:
    """Build the original feature matrix once and preserve an encrypted cache."""
    from src import database
    from src.preprocessing.feature_wt import (
        FEATURE_COLS_WT,
        build_features_wt,
        load_raw_data_wt,
    )

    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(checkpoint, database.DB_PATH, decrypt=True)
    df = build_features_wt(load_raw_data_wt("2023-01-01", "2026-10-23"))
    with database.get_connection() as conn:
        sizes = dict(conn.execute("SELECT race_key,n_entries FROM wt_races"))
    # Storage metadata only; this is NEVER added to the 70 model input columns.
    df["_n_entries"] = df.race_key.map(sizes)
    assert len(FEATURE_COLS_WT) == 70
    if df[FEATURE_COLS_WT].isna().any().any():
        raise RuntimeError("original_feature_builder_left_missing_features")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.pkl"
        df.to_pickle(path)
        crypt(path, output / "original-features.pkl.gpg")
    report = {
        "races": int(df.race_key.nunique()),
        "rows": len(df),
        "feature_names": FEATURE_COLS_WT,
        "source_period": [str(df.race_date.min()), str(df.race_date.max())],
        "input_policy": "original builtin imputation unchanged; user confirmed",
        "fit_has_not_run_in_this_stage": True,
        "history": "unchanged original point-in-time rolling and previous-day meeting-form functions",
    }
    (output / "original-features.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "feature_rows": len(df),
                "races": report["races"],
                "feature_count": len(FEATURE_COLS_WT),
            }
        ),
        flush=True,
    )


def window(features: Path, n_car: int, index: int, output: Path) -> None:
    """Call original fit_predict with its original five seeds, on past data only."""
    import pandas as pd
    from scripts.exp_7s7a_payout_structure import PRED_FILES
    from scripts.exp_axis_rule_decomposition import SEEDS, fit_predict
    from src.preprocessing.feature_wt import FEATURE_COLS_WT

    specs = [s for s in PRED_FILES if s[1] <= "2025-12-31"]
    start, end, source_name = specs[index]
    # The original 7-car cache deliberately uses two different history starts.
    # The original 9-car generator starts at 2022-12; actual data begin 2023-01.
    history_start = (
        source_name.rsplit("_", 1)[-1].removesuffix(".pkl")
        if n_car == 7
        else "2022-12-01"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.pkl"
        crypt(features, path, decrypt=True)
        df = pd.read_pickle(path)
    train = df[
        (df.race_date >= history_start)
        & (df.race_date < start)
        & df.finish_order.notna()
    ].copy()
    if n_car == 7:
        train = train[train._n_entries == 7].copy()
    test = df[
        (df.race_date >= start) & (df.race_date <= end) & (df._n_entries == n_car)
    ].copy()
    if train.empty or test.empty:
        raise RuntimeError("original_walkforward_window_empty")
    assert str(train.race_date.max()) < str(test.race_date.min())
    assert not set(train.race_key) & set(test.race_key)
    result = test[["race_key", "frame_no", "race_date"]].copy()
    result["pp3"] = fit_predict(train, test, "top3_flag")
    result["ppw"] = fit_predict(train, test, "win_flag")
    output.mkdir(parents=True, exist_ok=True)
    name = f"wf-n{n_car}-{start}-{end}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.pkl"
        result.to_pickle(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        crypt(path, output / (name + ".pkl.gpg"))
    report = {
        "n_car": n_car,
        "window_index": index,
        "prediction_from": start,
        "prediction_to": end,
        "original_history_start": history_start,
        "actual_fit_from": str(train.race_date.min()),
        "actual_fit_through": str(train.race_date.max()),
        "fit_races": int(train.race_key.nunique()),
        "predicted_races": int(test.race_key.nunique()),
        "predicted_entries": len(test),
        "features": FEATURE_COLS_WT,
        "seeds": SEEDS,
        "prediction_sha256": sha,
        "original_function": "scripts.exp_axis_rule_decomposition.fit_predict",
        "labels": "Only confirmed original finish_order rows fit; no missing result is fabricated.",
        "period_difference": "Original 2022-12 data unavailable; actual source begins 2023-01-01.",
        "no_2026_training": True,
    }
    (output / (name + ".json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    """CLI for source-preserving feature preparation and independent windows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "window"])
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--features", type=Path)
    ap.add_argument("--n-car", type=int, choices=[7, 9], default=7)
    ap.add_argument("--window-index", type=int, choices=range(6), default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.stage == "prepare":
        if args.checkpoint is None:
            raise RuntimeError("verified_checkpoint_required")
        prepare(args.checkpoint, args.output)
    else:
        if args.features is None:
            raise RuntimeError("original_feature_cache_required")
        window(args.features, args.n_car, args.window_index, args.output)


if __name__ == "__main__":
    main()
