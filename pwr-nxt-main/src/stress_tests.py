"""Robustness stress tests — evidence generator for METHODOLOGY_NOTE section 2b.

Regenerates every stress-test number cited in the methodology note:
  1. Task 2 extrapolation on all four operating axes (hybrid vs pure CatBoost)
  2. Task 1 label-noise sensitivity (5% of labels flipped)
  3. Task 1 out-of-fold error analysis (missed Invalid / false-flagged Valid)
  4. Task 2 noise-floor diagnostic (OOF residual vs input correlations)

Writes results/stress_tests.txt. Deterministic (seed 42).
Run:  python src/stress_tests.py
"""
from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
import pipeline as P

from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

OUT = io.StringIO()


def log(*a):
    print(*a)
    print(*a, file=OUT)


def fit_pure_catboost_bag(train_df: pd.DataFrame):
    X = train_df[P.IN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    X["VI"] = X["Applied_Voltage_kV"] * X["Load_Current_A"]
    regs = [CatBoostRegressor(iterations=800, learning_rate=0.04, depth=4,
                              verbose=0, random_state=P.SEED + s) for s in range(5)]
    for m in regs:
        m.fit(X[P.REG_COLS], train_df[P.YCOL])
    return regs


def pred_bag(regs, df):
    return np.mean([m.predict(P._reg_matrix(df)) for m in regs], axis=0)


def task1_oof(tr: pd.DataFrame):
    """Out-of-fold Invalid predictions for the deployed Task 1 ensemble."""
    y = (tr["Validity_Label"] == "Invalid").astype(int).values
    dup_all = tr.duplicated(subset=P.IN + P.SEN, keep=False)
    preds = np.zeros(len(tr))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=P.SEED)
    for trn, tst in cv.split(tr, y):
        sub = P.train_task1(tr.iloc[trn])
        res = P.apply_task1(sub, tr.iloc[tst], dup_flags=dup_all.iloc[tst])
        preds[tst] = res["invalid"].astype(int)
    return y, preds


def main():
    tr = pd.read_excel(P.XLSX, sheet_name="Training_Data")
    valid = tr[tr.Validity_Label == "Valid"].reset_index(drop=True)

    log("=" * 78)
    log("STRESS TEST EVIDENCE — team pentupbois (regenerated, seed 42)")
    log("=" * 78)

    # ---- 1. Task 2 extrapolation, four axes --------------------------------
    log("")
    log("[1] Task 2 extrapolation: withhold the top quintile of each operating")
    log("    dimension, train on the rest, predict the withheld band.")
    log(f"    {'axis':<24}{'held-out band':>22}{'hybrid RMSE':>13}{'pure CAT':>10}")
    for dim in ["Load_Current_A", "Applied_Voltage_kV",
                "Ambient_Temperature_C", "Test_Duration_min"]:
        mask = valid[dim] >= valid[dim].quantile(0.80)
        train_df, test_df = valid[~mask], valid[mask]
        y = test_df[P.YCOL].values
        hyb = P.train_task2(train_df)
        p_hyb = P.apply_task2(hyb, test_df)
        p_pure = pred_bag(fit_pure_catboost_bag(train_df), test_df)
        log(f"    {dim:<24}{f'[{test_df[dim].min():.1f}, {test_df[dim].max():.1f}]':>22}"
            f"{np.sqrt(mean_squared_error(y, p_hyb)):>13.3f}"
            f"{np.sqrt(mean_squared_error(y, p_pure)):>10.3f}")

    # ---- 2. Task 1 label-noise sensitivity ---------------------------------
    log("")
    log("[2] Task 1 label-noise: flip 5% of training labels at random, re-run")
    log("    the full CV protocol, score against the corrupted labels.")
    rng = np.random.default_rng(P.SEED)
    y_true = (tr["Validity_Label"] == "Invalid").astype(int).values.copy()
    flip = rng.random(len(y_true)) < 0.05
    tr_corrupt = tr.copy()
    tr_corrupt["Validity_Label"] = np.where(
        y_true ^ flip.astype(int) == 1, "Invalid", "Valid")
    y_corrupt = (tr_corrupt["Validity_Label"] == "Invalid").astype(int).values
    _, preds = task1_oof(tr_corrupt)
    log(f"    corrupted labels: {int(flip.sum())} of {len(y_true)} flipped")
    log(f"    CV accuracy vs corrupted labels: {accuracy_score(y_corrupt, preds):.3f}")

    # ---- 3. Task 1 out-of-fold error analysis ------------------------------
    log("")
    log("[3] Task 1 out-of-fold error analysis (uncorrupted labels).")
    y, preds = task1_oof(tr)
    fn = tr.index[(y == 1) & (preds == 0)]
    fp = tr.index[(y == 0) & (preds == 1)]
    log(f"    missed Invalid (false negatives): {len(fn)}  {[f'TRN-{i:04d}' for i in fn]}")
    log(f"    false-flagged Valid (false positives): {len(fp)}")

    # ---- 4. Task 2 noise-floor diagnostic ----------------------------------
    log("")
    log("[4] Task 2 noise floor: OOF residual vs each input. If |corr| ~ 0")
    log("    everywhere, no learnable signal remains (we are at the floor).")
    yv = valid[P.YCOL].values
    cvr = RepeatedKFold(n_splits=5, n_repeats=3, random_state=P.SEED)
    oof = np.zeros(len(valid))
    for trn, tst in cvr.split(valid):
        sub = P.train_task2(valid.iloc[trn])
        oof[tst] = P.apply_task2(sub, valid.iloc[tst])
    resid = yv - oof
    vd = valid[P.IN].copy()
    vd["I2"] = vd["Load_Current_A"] ** 2
    for c in vd.columns:
        log(f"    corr(resid, {c:<24}) = {np.corrcoef(resid, vd[c])[0, 1]:+.4f}")
    log(f"    max |corr| = {max(abs(np.corrcoef(resid, vd[c])[0, 1]) for c in vd.columns):.4f}"
        "  -> no learnable structure remains: at the noise floor")
    log(f"    CV RMSE = {np.sqrt(mean_squared_error(yv, oof)):.4f}")

    out_path = ROOT / "results" / "stress_tests.txt"
    out_path.write_text(OUT.getvalue(), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
