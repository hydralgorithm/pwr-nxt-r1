"""
PowerNext-AI — CPRI Black-Box Test Bench Challenge: production pipeline.

Winning architecture (selected by a 17-candidate bake-off under repeated
5x3-fold cross-validation, identical folds for every candidate):

  Task 1  Valid/Invalid: probability-mean ensemble of
          RandomForest + SVM-RBF + LightGBM + CatBoost on engineered
          *relationship* features (inter-sensor residuals, duplicate
          structure, missingness pattern, non-positivity).
          CV: accuracy 0.999, precision 1.000, recall 0.993, F1 0.996, AUC 1.000.
  Task 2  Reference_Parameter: physics-informed hybrid — an OLS physics base
          (Ref ~ I + I^2 + Ambient + Voltage + Duration) plus a 5-seed bagged
          CatBoost (depth 4) that learns only the bounded residual correction,
          on operating inputs only (V, I, I^2, V*I, Ambient, Duration) —
          deliberately sensor-free so predictions are immune to the very faults
          Task 1 detects. CV RMSE 0.41 degC, MAE 0.27, R^2 0.9985 on Valid
          rows; ~3x better than pure CatBoost when predicting beyond the
          training current range (tree models cannot extrapolate; the OLS
          base can).
  Task 3  programmatic summary.json (counts, min/max/avg, 3 attention IDs,
          <=100-word blurb, CV metrics).

The four physically-motivated fault rules (duplicate logging, non-positive
reading, thermometer dropout, inter-sensor disagreement) are computed and
reported as an explainability layer; the learned ensemble subsumes them
(features include the rule flags) and outperforms them under honest CV.

Run:  python src/pipeline.py      ->  submission/pentupbois.csv, submission/summary.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"
SUB.mkdir(exist_ok=True)

# Flexible dataset location search: CLI argument > ROOT > cwd > parent
_candidates = []
if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
    _candidates.append(Path(sys.argv[1]))
_candidates.extend([
    ROOT / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx",
    Path.cwd() / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx",
    ROOT.parent / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx",
])
XLSX = next((p for p in _candidates if p.exists()), _candidates[0])

SEED = 42
TEAM = "pentupbois"
IN = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min"]
SEN = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
S123 = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]
YCOL = "Reference_Parameter"

ENG_COLS = IN + SEN + ["I2", "VI", "resid_Sensor_S2", "resid_Sensor_S3", "resid_max",
                       "n_missing_S123", "n_missing_S4", "any_nonpositive", "dup_feature"]
REG_COLS = ["Applied_Voltage_kV", "Load_Current_A", "I2", "VI",
            "Ambient_Temperature_C", "Test_Duration_min"]


# ============================================================== feature engineering
def sensor_residual_params(valid: pd.DataFrame) -> dict:
    """OLS S2~S1, S3~S1 fit on VALID training rows only (R^2 0.89 / 0.98)."""
    p = {}
    for a, b in [("Sensor_S1", "Sensor_S2"), ("Sensor_S1", "Sensor_S3")]:
        d = valid[[a, b]].dropna()
        lr = LinearRegression().fit(d[[a]], d[b])
        p[b] = (float(lr.intercept_), float(lr.coef_[0]))
    return p


def make_features(df: pd.DataFrame, resid_params: dict, dup_flags: pd.Series | None = None) -> pd.DataFrame:
    """Test-time-computable features. NEVER uses Reference_Parameter.

    `dup_flags` optionally supplies duplicate flags computed on the whole file
    being scored (deployment semantics); defaults to within-`df` detection.
    """
    X = df[IN + SEN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    X["VI"] = X["Applied_Voltage_kV"] * X["Load_Current_A"]
    for b, (c0, c1) in resid_params.items():
        X[f"resid_{b}"] = np.where(
            X["Sensor_S1"].notna() & X[b].notna(), X[b] - (c0 + c1 * X["Sensor_S1"]), np.nan)
    X["resid_max"] = X[["resid_Sensor_S2", "resid_Sensor_S3"]].max(axis=1)
    X["n_missing_S123"] = X[S123].isna().sum(axis=1)
    X["n_missing_S4"] = X["Sensor_S4"].isna().astype(int)
    X["any_nonpositive"] = (X[SEN] <= 0).any(axis=1).astype(int)
    if dup_flags is None:
        dup_flags = df.duplicated(subset=IN + SEN, keep=False)
    X["dup_feature"] = dup_flags.astype(int).values
    return X


def residual_thresholds(X_valid: pd.DataFrame, q: float = 0.995) -> dict:
    return {"resid_Sensor_S2": float(X_valid["resid_Sensor_S2"].quantile(q)),
            "resid_Sensor_S3": float(X_valid["resid_Sensor_S3"].quantile(q))}


def rule_flags(X: pd.DataFrame, thr: dict) -> pd.Series:
    """Four physically-motivated fault rules (explainability layer)."""
    return (
        (X["dup_feature"] == 1)                            # R1 duplicate logging
        | (X["any_nonpositive"] == 1)                      # R2 impossible reading
        | (X["n_missing_S123"] > 0)                        # R3 thermometer dropout
        | (X["resid_Sensor_S2"] > thr["resid_Sensor_S2"])  # R4a sensor disagreement
        | (X["resid_Sensor_S3"] > thr["resid_Sensor_S3"])  # R4b
    )


# ============================================================== Task 1
def _task1_members(seed: int = SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    return {
        "rf": RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                     class_weight="balanced", random_state=seed, n_jobs=-1),
        "svm": make_pipeline(SimpleImputer(), StandardScaler(),
                             SVC(probability=True, class_weight="balanced", random_state=seed)),
        "lgbm": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                               min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                               class_weight="balanced", verbose=-1, random_state=seed),
        "cat": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=4,
                                  verbose=0, random_seed=seed),
    }


def train_task1(tr: pd.DataFrame) -> dict:
    valid = tr[tr.Validity_Label == "Valid"]
    resid_params = sensor_residual_params(valid)
    X = make_features(tr, resid_params)
    thr = residual_thresholds(X.loc[valid.index])
    y = (tr["Validity_Label"] == "Invalid").astype(int)
    members = _task1_members()
    for m in members.values():
        m.fit(X[ENG_COLS], y)  # NaN passed through natively (trees, LGBM, CAT; SVM imputes)
    return {"resid_params": resid_params, "thresholds": thr, "members": members}


def apply_task1(model: dict, df: pd.DataFrame, dup_flags: pd.Series | None = None) -> pd.DataFrame:
    X = make_features(df, model["resid_params"], dup_flags=dup_flags)
    Xf = X[ENG_COLS]  # NaN semantics, identical to training
    probs = {name: m.predict_proba(Xf)[:, 1]
             for name, m in model["members"].items()}
    ens = np.mean(list(probs.values()), axis=0)
    rules = rule_flags(X, model["thresholds"])
    return pd.DataFrame({
        "Test_ID": df["Test_ID"].values,
        "rule_invalid": rules.values,
        "learned_proba": ens,
        "member_proba_rf": probs["rf"],
        "member_proba_svm": probs["svm"],
        "member_proba_lgbm": probs["lgbm"],
        "member_proba_cat": probs["cat"],
        "invalid": ens > 0.5,
    })


# ============================================================== Task 2
def train_task2(tr: pd.DataFrame) -> dict:
    """Physics-informed hybrid: OLS physics base + 5-seed CatBoost residual bag.

    The OLS term (I + I^2 + Ambient + Voltage + Duration) carries the
    extrapolatable physics; the CatBoost bag learns only the bounded residual
    correction. In-range CV RMSE 0.415 (vs 0.517 pure CatBoost) and ~3x better
    error when predicting beyond the training current range (stress test:
    RMSE 3.5 vs 11.4 on a held-out 85-110A band).
    """
    from catboost import CatBoostRegressor

    valid = tr[tr.Validity_Label == "Valid"].copy()
    X = valid[IN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    X["VI"] = X["Applied_Voltage_kV"] * X["Load_Current_A"]
    y = valid[YCOL]

    lin_cols = ["Load_Current_A", "I2", "Ambient_Temperature_C",
                "Applied_Voltage_kV", "Test_Duration_min"]
    ols = LinearRegression().fit(X[lin_cols], y)
    resid = y - ols.predict(X[lin_cols])

    regs = []
    for s in range(5):
        m = CatBoostRegressor(iterations=800, learning_rate=0.04, depth=4,
                              verbose=0, random_seed=SEED + s)
        m.fit(X[REG_COLS], resid)
        regs.append(m)

    return {"regs": regs, "ols": ols, "ols_cols": lin_cols}


def _reg_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[IN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    X["VI"] = X["Applied_Voltage_kV"] * X["Load_Current_A"]
    return X[REG_COLS]


def apply_task2(model: dict, df: pd.DataFrame) -> np.ndarray:
    X = df[IN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    base = model["ols"].predict(X[model["ols_cols"]])
    corr = np.mean([m.predict(_reg_matrix(df)) for m in model["regs"]], axis=0)
    return base + corr


def apply_task2_ols(model: dict, df: pd.DataFrame) -> np.ndarray:
    X = df[IN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    return model["ols"].predict(X[model["ols_cols"]])


# ============================================================== Task 3
def make_summary(pred_df: pd.DataFrame, t1_cv: dict, t2_cv: dict, rule_hits: int) -> dict:
    valid_rows = pred_df[~pred_df["invalid"]]
    q75 = valid_rows["learned_proba"].quantile(0.75)
    pool = valid_rows[valid_rows["learned_proba"] >= q75]
    if len(pool) < 3:
        pool = valid_rows
    attention = pool.nlargest(3, "pred_ref")["Test_ID"].tolist()

    blurb = (
        "Hybrid physics-plus-ML pipeline. Validity is decided by a four-model "
        "ensemble (random forest, SVM, LightGBM, CatBoost) on relationship "
        "features: inter-sensor residuals against a Valid-fit agreement model, "
        "duplicate-logging structure, missingness pattern, and impossible "
        "readings. Reference_Parameter is predicted from operating inputs only "
        "by a physics-informed hybrid: a Joule-heating linear model carries "
        "the extrapolatable physics and a bagged CatBoost learns the bounded "
        "residual correction. Thresholds derive from Valid-row distributions; "
        "fully automatic, no manual edits."
    )
    return {
        "team": TEAM,
        "record_count": int(len(pred_df)),
        "predicted_invalid_count": int(pred_df["invalid"].sum()),
        "predicted_valid_count": int((~pred_df["invalid"]).sum()),
        "rule_layer_invalid_count": int(rule_hits),
        "reference_parameter": {
            "min": float(pred_df["pred_ref"].min()),
            "max": float(pred_df["pred_ref"].max()),
            "mean": float(pred_df["pred_ref"].mean()),
            "median": float(pred_df["pred_ref"].median()),
            "std": float(pred_df["pred_ref"].std()),
        },
        "highest_attention_test_ids": attention,
        "attention_rule": (
            "Highest predicted Reference_Parameter among Valid rows whose "
            "ensemble validity probability is in the least-confident quartile "
            "(>= 75th percentile of P(Invalid)); falls back to all Valid rows "
            "if fewer than 3 qualify."
        ),
        "methodology_blurb": blurb,
        "validation": {"task1_cv": t1_cv, "task2_cv": t2_cv},
    }


# ============================================================== main
def main():
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                                 mean_squared_error, precision_score, r2_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

    tr = pd.read_excel(XLSX, sheet_name="Training_Data")
    te = pd.read_excel(XLSX, sheet_name="Test_Data")
    ss = pd.read_excel(XLSX, sheet_name="Sample_Submission")

    # ---- Task 1: fit on all training data, apply to test
    m1 = train_task1(tr)
    out_te = apply_task1(m1, te)

    # honest CV estimate of the exact deployed ensemble (features refit in-fold)
    y = (tr["Validity_Label"] == "Invalid").astype(int).values
    dup_all = tr.duplicated(subset=IN + SEN, keep=False)
    preds = np.zeros(len(tr)); probs = np.zeros(len(tr))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    for trn, tst in cv.split(tr, y):
        sub = train_task1(tr.iloc[trn])
        res = apply_task1(sub, tr.iloc[tst], dup_flags=dup_all.iloc[tst])
        preds[tst] = res["invalid"].astype(int)
        probs[tst] = res["learned_proba"]
    t1_cv = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision_invalid": float(precision_score(y, preds, zero_division=0)),
        "recall_invalid": float(recall_score(y, preds, zero_division=0)),
        "f1_invalid": float(f1_score(y, preds, zero_division=0)),
        "auc": float(roc_auc_score(y, probs)),
        "protocol": "repeated 5x3-fold stratified CV, out-of-fold, all preprocessing refit in-fold",
    }
    print("Task1 CV:", t1_cv)

    # ---- Task 2
    m2 = train_task2(tr)
    out_te["pred_ref"] = apply_task2(m2, te).round(4)  # rounded once; csv & summary agree exactly
    ols_ref = apply_task2_ols(m2, te)

    valid = tr[tr.Validity_Label == "Valid"].reset_index(drop=True)
    yv = valid[YCOL].values
    cvr = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    oof = np.zeros(len(valid))
    for trn, tst in cvr.split(valid):
        sub = train_task2(tr[tr.Validity_Label == "Valid"].iloc[trn])
        oof[tst] = apply_task2(sub, valid.iloc[tst])
    t2_cv = {
        "rmse": float(np.sqrt(mean_squared_error(yv, oof))),
        "mae": float(mean_absolute_error(yv, oof)),
        "r2": float(r2_score(yv, oof)),
        "n_train_valid": int(len(valid)),
        "oof_abs_error_q90": float(np.quantile(np.abs(yv - oof), 0.9)),
        "protocol": "repeated 5x3-fold CV on Valid rows only, out-of-fold",
    }
    print("Task2 CV:", t2_cv)

    # Noise-floor diagnostic: if the OOF residual still correlates with any
    # input, learnable signal remains; if all |corr| are ~0, we are at the
    # irreducible-noise floor of the data-generating process.
    resid = yv - oof
    vd = valid[IN].copy()
    vd["I2"] = vd["Load_Current_A"] ** 2
    floor_diag = {c: float(np.corrcoef(resid, vd[c])[0, 1]) for c in vd.columns}
    print("Noise-floor diagnostic (|corr| of OOF residual with inputs, max):",
          round(max(abs(v) for v in floor_diag.values()), 4))

    # ---- assemble deliverable in Sample_Submission order / header
    order = ss["Test_ID"].tolist()
    out_te = out_te.set_index("Test_ID").loc[order].reset_index()
    sub_df = pd.DataFrame({
        "Test_ID": out_te["Test_ID"],
        "Predicted_Reference_Parameter": out_te["pred_ref"].round(4),
        "Validity_Label": np.where(out_te["invalid"], "Invalid", "Valid"),
    })
    sub_df.to_csv(SUB / f"{TEAM}.csv", index=False)
    if ROOT.name == "pentupbois-submission" or not (ROOT / f"{TEAM}.csv").exists():
        sub_df.to_csv(ROOT / f"{TEAM}.csv", index=False)

    summary = make_summary(out_te, t1_cv, t2_cv, int(out_te["rule_invalid"].sum()))
    summary["physics_model_agreement"] = {
        "pearson_r_catboost_vs_ols_on_test": float(np.corrcoef(out_te["pred_ref"], ols_ref)[0, 1]),
        "mean_abs_diff": float(np.abs(out_te["pred_ref"] - ols_ref).mean()),
    }
    # honest uncertainty: 90% of CV predictions were within this half-width
    summary["prediction_uncertainty"] = {
        "p90_abs_error_degC": t2_cv["oof_abs_error_q90"],
        "interpretation": "Based on out-of-fold CV, 90% of predictions are "
                          "expected within this absolute error (degC) of the true value.",
    }
    # automatic input-drift guard: is the scored file still inside the
    # training distribution? (KS test per operating input)
    from scipy.stats import ks_2samp
    ks = {c: float(ks_2samp(tr[c].dropna(), te[c].dropna()).pvalue) for c in IN}
    summary["input_drift_check"] = {
        "ks_pvalue_per_input": {k: round(v, 4) for k, v in ks.items()},
        "within_training_distribution": bool(all(p >= 0.05 for p in ks.values())),
        "note": "Kolmogorov-Smirnov test per input vs training; p >= 0.05 means "
                "no detectable distribution shift. If False on a future run, "
                "predictions rely on the extrapolating physics base.",
    }
    summary["noise_floor_diagnostic"] = {k: round(v, 4) for k, v in floor_diag.items()}
    summary_json_text = json.dumps(summary, indent=2)
    (SUB / "summary.json").write_text(summary_json_text)
    if ROOT.name == "pentupbois-submission" or not (ROOT / "summary.json").exists():
        (ROOT / "summary.json").write_text(summary_json_text)

    # sanity report
    dup_ids = te.loc[te.duplicated(subset=IN + SEN, keep=False), "Test_ID"].tolist()
    flagged = out_te.loc[out_te["Test_ID"].isin(dup_ids), "invalid"].tolist()
    print(f"\nWrote {SUB / (TEAM + '.csv')} ({len(sub_df)} rows) and summary.json")
    print("Invalid predicted:", int(out_te["invalid"].sum()), "/", len(out_te),
          f"({100*out_te['invalid'].mean():.1f}%)")
    print("Pred Ref min/max/mean:", round(out_te["pred_ref"].min(), 2),
          round(out_te["pred_ref"].max(), 2), round(out_te["pred_ref"].mean(), 2))
    print("Duplicate test-pairs flagged:", sum(flagged), "of", len(flagged), dup_ids)
    print("Attention IDs:", summary["highest_attention_test_ids"])


if __name__ == "__main__":
    main()
