"""Measure saved reconstructed models against captured original display values.

This is evaluation only. It does not manufacture odds or purchase decisions.
Original displayed probabilities may have been backfilled after issuance, so
agreement here must not be described as agreement with original issuance values.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reconstruct_core import crypt


def main() -> None:
    """Load encrypted actual trained models, call original inference, compare."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from src import database

    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    crypt(args.checkpoint, database.DB_PATH, decrypt=True)
    model_dir = ROOT / "data/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("lgbm_wt_eval", "lgbm_wt_win"):
        files = list(args.models.rglob(name + ".tar.gpg"))
        if len(files) != 1:
            raise RuntimeError("trained_model_artifact_missing_or_ambiguous")
        with tempfile.TemporaryDirectory() as tmp:
            tar = Path(tmp) / "models.tar"
            crypt(files[0], tar, decrypt=True)
            with tarfile.open(tar) as archive:
                for member in archive.getmembers():
                    if (
                        member.name not in (name + ".pkl", name + ".meta.json")
                        or not member.isfile()
                    ):
                        raise RuntimeError("unexpected_model_archive_member")
                    data = archive.extractfile(member)
                    if data is None:
                        raise RuntimeError("invalid_model_archive")
                    (model_dir / member.name).write_bytes(data.read())

    # The original historical inference path requires an actual frozen vintage.
    # These models were fit only through Dec 2025, so m2601 is their truthful
    # vintage (never label them m2610 or disable the original past-date guard).
    from src.models.trainer import load_model, save_model

    for name in ("lgbm_wt_eval", "lgbm_wt_win"):
        meta = json.loads((model_dir / (name + ".meta.json")).read_text())
        if name.endswith("_win"):
            valid = bool(meta.get("full_refit")) and meta.get("to") == "2025-12-31"
        else:
            valid = not meta.get("full_refit") and meta.get("test_from") == "2026-01-01"
        if not valid:
            raise RuntimeError("saved_model_is_not_a_frozen_2025_end_model")
        frozen = name + "_m2601"
        save_model(load_model(name), frozen)
        (model_dir / (frozen + ".meta.json")).write_text(
            json.dumps(
                {**meta, "model_name": frozen, "frozen_fit_end": "2025-12-31"}, indent=2
            )
        )

    from scripts.build_type_lab_picks import predict_p3_pw

    p3, pw = predict_p3_pw(
        "2026-10-23", eval_model="lgbm_wt_eval_m2601", win_model="lgbm_wt_win_m2601"
    )
    reference_path = ROOT / "reconstruction/original_reference_20261023.json.gz"
    reference = json.loads(gzip.decompress(reference_path.read_bytes()))
    rows = []
    missing = 0
    for race in reference["data"]["displayed"]:
        rk = race["race_key"]
        for entry in race["entries"]:
            c = int(entry["frame_no"])
            for kind, predictions, field in (
                ("top3", p3, "pred_top3_pct"),
                ("win", pw, "pred_win_pct"),
            ):
                expected = entry.get(field)
                actual = predictions.get(rk, {}).get(c)
                if expected is None or actual is None:
                    missing += 1
                    continue
                delta = float(actual) * 100.0 - float(expected)
                rows.append(
                    {
                        "race_key": rk,
                        "frame_no": c,
                        "kind": kind,
                        "reference_pct": expected,
                        "reconstructed_pct": float(actual) * 100.0,
                        "delta_pp": delta,
                        "equal_at_display_precision": round(float(actual) * 100.0, 1)
                        == float(expected),
                    }
                )
    summary = {}
    for kind in ("win", "top3"):
        subset = [r for r in rows if r["kind"] == kind]
        n = len(subset)
        summary[kind] = {
            "n": n,
            "n_equal_at_display_precision": sum(
                r["equal_at_display_precision"] for r in subset
            ),
            "mean_absolute_delta_pp": sum(abs(r["delta_pp"]) for r in subset) / n
            if n
            else None,
        }
    # Classify using reconstructed probabilities plus actual collected inputs.
    # This is distinct from the original-input rule replay already recorded.
    from scripts.build_type_lab_picks import _load_entries, _load_race_meta
    from src.type_lab import race_shape

    expected_types = {}
    for row in reference["data"]["type_lab"]["picks"]:
        expected_types.setdefault(str(row["race_key"]), set()).add(row["type_label"])
    keys = sorted(expected_types)
    entries, metas = _load_entries(keys), _load_race_meta(keys)
    type_rows = []
    for rk in keys:
        ent, meta = entries.get(rk), metas.get(rk)
        if not ent or not meta or set(ent) != set(p3.get(rk, {})):
            continue
        shape = race_shape(
            p3[rk],
            {c: v["line_group"] for c, v in ent.items()},
            {c: v["line_pos"] for c, v in ent.items()},
            {c: v["style"] for c, v in ent.items()},
            {c: v["race_point"] for c, v in ent.items()},
            {c: v["behind"] for c, v in ent.items()},
            meta.get("day_index") or 0,
            pw.get(rk),
        )
        if shape is not None:
            type_rows.append(
                {
                    "race_key": rk,
                    "type": shape.type_label,
                    "reference_types": sorted(expected_types[rk]),
                    "equal": expected_types[rk] == {shape.type_label},
                }
            )
    type_summary = {
        "n": len(type_rows),
        "n_equal": sum(r["equal"] for r in type_rows),
        "missing": len(keys) - len(type_rows),
        "input_asof_verified": False,
    }
    out = {
        "date": "2026-10-23",
        "reference_captured_at": reference["captured_at"],
        "summary": summary,
        "missing_probabilities": missing,
        "model_probability_scale": "raw; no additional calibration",
        "original_issuance_probabilities_verified": False,
        "note": "Compares current original public display values at capture time; not guaranteed original issuance-time probabilities. No full model equality is claimed.",
        "reconstructed_type_agreement": type_summary,
        "plan_agreement": None,
        "purchase_agreement": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "comparison.json"
        path.write_text(
            json.dumps({"probabilities": rows, "types": type_rows}, ensure_ascii=False)
        )
        crypt(path, args.output.with_suffix(".rows.json.gpg"))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
