"""
Model bake-off for the CPRI PowerNext-AI Black-Box Test Bench Challenge.

Compares candidate models for:
  Task 1  Valid/Invalid classification  (repeated stratified 5-fold CV)
  Task 2  Reference_Parameter regression on Valid rows (repeated 5-fold CV)

All candidates are evaluated with identical folds and seeds. Rule-engine and
sensor-residual features are refit INSIDE each fold on training data only
(no leakage). Results are written to results/bakeoff_results.md and .json.

Run:  python src/bakeoff.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
IN = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min"]
SEN = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
S123 = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]


# ----------------------------------------------------------------------------- data
def load():
    tr = pd.read_excel(XLSX, sheet_name="Training_Data")
    te = pd.read_excel(XLSX, sheet_name="Test_Data")
    return tr, te


# ----------------------------------------------------------------------------- features
def sensor_residual_params(train_valid: pd.DataFrame):
    """Fit S2~S1 and S3~S1 OLS on VALID training rows only."""
    p = {}
    for a, b in [("Sensor_S1", "Sensor_S2"), ("Sensor_S1", "Sensor_S3")]:
        d = train_valid[[a, b]].dropna()
        m = sm.OLS(d[b], sm.add_constant(d[a])).fit()
        p[b] = (m.params.iloc[0], m.params.iloc[1])
    return p


def make_features(df: pd.DataFrame, resid_params: dict, dup_flags: pd.Series | None = None):
    """Engineered features computable at TEST TIME (no Reference_Parameter use).

    `dup_flags` overrides duplicate detection with deployment semantics:
    at scoring time duplicates are found within the file being scored, so in CV
    we precompute flags on the full file once (features only, no label leakage).
    """
    X = df[IN + SEN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    X["VI"] = X["Applied_Voltage_kV"] * X["Load_Current_A"]

    # sensor mutual-consistency residuals
    for b, (c0, c1) in resid_params.items():
        X[f"resid_{b}"] = np.where(
            X["Sensor_S1"].notna() & X[b].notna(), X[b] - (c0 + c1 * X["Sensor_S1"]), np.nan
        )
    X["resid_max"] = X[["resid_Sensor_S2", "resid_Sensor_S3"]].max(axis=1)

    # NaN structure
    X["n_missing_S123"] = X[S123].isna().sum(axis=1)
    X["n_missing_S4"] = X["Sensor_S4"].isna().astype(int)
    X["any_nonpositive"] = (X[SEN] <= 0).any(axis=1).astype(int)

    if dup_flags is None:
        dup_flags = df.duplicated(subset=IN + SEN, keep=False)
    X["dup_feature"] = dup_flags.astype(int).values
    return X


def residual_thresholds(train_valid_X: pd.DataFrame, q: float = 0.995):
    return {
        "resid_Sensor_S2": float(train_valid_X["resid_Sensor_S2"].quantile(q)),
        "resid_Sensor_S3": float(train_valid_X["resid_Sensor_S3"].quantile(q)),
    }


def rule_flags(X: pd.DataFrame, thr: dict) -> pd.Series:
    """The four physically-motivated fault rules. Test-time applicable."""
    return (
        (X["dup_feature"] == 1)
        | (X["any_nonpositive"] == 1)
        | (X["n_missing_S123"] > 0)
        | (X["resid_Sensor_S2"] > thr["resid_Sensor_S2"])
        | (X["resid_Sensor_S3"] > thr["resid_Sensor_S3"])
    )


# ----------------------------------------------------------------------------- candidates
def get_classifiers():
    cand = {
        "Dummy(majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic(raw)": make_pipeline(
            SimpleImputer(), StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced")),
        "Logistic(eng)": make_pipeline(
            SimpleImputer(), StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced")),
        "kNN(eng)": make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsClassifier(15)),
        "SVM-RBF(eng)": make_pipeline(
            SimpleImputer(), StandardScaler(), SVC(probability=True, class_weight="balanced")),
        "RandomForest(eng)": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=SEED),
        "ExtraTrees(eng)": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=SEED),
        "HistGB(eng)": HistGradientBoostingClassifier(max_depth=3, random_state=SEED),
        "LightGBM(eng)": None,   # filled lazily if importable
        "XGBoost(eng)": None,
        "CatBoost(eng)": None,
        "TabPFN(eng)": None,
    }
    try:
        from lightgbm import LGBMClassifier
        cand["LightGBM(eng)"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced", verbose=-1, random_state=SEED)
    except ImportError:
        del cand["LightGBM(eng)"]
    try:
        from xgboost import XGBClassifier
        cand["XGBoost(eng)"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
            eval_metric="logloss", random_state=SEED)
    except ImportError:
        del cand["XGBoost(eng)"]
    try:
        from catboost import CatBoostClassifier
        cand["CatBoost(eng)"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=4, verbose=0, random_seed=SEED)
    except ImportError:
        del cand["CatBoost(eng)"]
    try:
        from tabpfn import TabPFNClassifier
        cand["TabPFN(eng)"] = TabPFNClassifier(random_state=SEED)
    except Exception:
        del cand["TabPFN(eng)"]
    return cand


def get_regressors():
    cand = {
        "Dummy(mean)": DummyRegressor(),
        "OLS-physics(I,I2,Amb,V,t)": None,  # special-cased
        "Ridge(inputs+I2)": make_pipeline(
            StandardScaler(), Ridge(alpha=1.0)),
        "ElasticNet(inputs+I2)": make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=10000)),
        "kNN(inputs)": make_pipeline(StandardScaler(), KNeighborsRegressor(15)),
        "SVR-RBF(inputs)": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.5)),
        "RandomForest(inputs)": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=SEED),
        "ExtraTrees(inputs)": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=SEED),
        "HistGB(inputs)": HistGradientBoostingRegressor(random_state=SEED),
        "LightGBM(inputs)": None,
        "XGBoost(inputs)": None,
        "CatBoost(inputs)": None,
        "TabPFN(inputs)": None,
        "HistGB(inputs+S123)": HistGradientBoostingRegressor(random_state=SEED),
        "LightGBM(inputs+S123)": None,
    }
    try:
        from lightgbm import LGBMRegressor
        cand["LightGBM(inputs)"] = LGBMRegressor(
            n_estimators=600, learning_rate=0.04, num_leaves=15, min_child_samples=15,
            subsample=0.9, colsample_bytree=0.9, verbose=-1, random_state=SEED)
        cand["LightGBM(inputs+S123)"] = LGBMRegressor(
            n_estimators=600, learning_rate=0.04, num_leaves=15, min_child_samples=15,
            subsample=0.9, colsample_bytree=0.9, verbose=-1, random_state=SEED)
    except ImportError:
        del cand["LightGBM(inputs)"], cand["LightGBM(inputs+S123)"]
    try:
        from xgboost import XGBRegressor
        cand["XGBoost(inputs)"] = XGBRegressor(
            n_estimators=600, learning_rate=0.04, max_depth=3, subsample=0.9,
            colsample_bytree=0.9, min_child_weight=5, random_state=SEED)
    except ImportError:
        del cand["XGBoost(inputs)"]
    try:
        from catboost import CatBoostRegressor
        cand["CatBoost(inputs)"] = CatBoostRegressor(
            iterations=800, learning_rate=0.04, depth=4, verbose=0, random_seed=SEED)
    except ImportError:
        del cand["CatBoost(inputs)"]
    try:
        from tabpfn import TabPFNRegressor
        cand["TabPFN(inputs)"] = TabPFNRegressor(random_state=SEED)
    except Exception:
        del cand["TabPFN(inputs)"]
    return cand


# ----------------------------------------------------------------------------- evaluation
def eval_task1(tr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = (tr["Validity_Label"] == "Invalid").astype(int).values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows, oof_store = [], {}
    # deployment semantics: duplicate detection runs on the whole scored file
    dup_all = tr.duplicated(subset=IN + SEN, keep=False)

    for name, clf in get_classifiers().items():
        oof = np.zeros(len(tr))
        ok = True
        for trn, tst in cv.split(tr, y):
            tr_fold, te_fold = tr.iloc[trn], tr.iloc[tst]
            vp = sensor_residual_params(tr_fold[tr_fold.Validity_Label == "Valid"])
            Xtr = make_features(tr_fold, vp, dup_flags=dup_all.iloc[trn])
            Xte = make_features(te_fold, vp, dup_flags=dup_all.iloc[tst])
            use_raw = "raw" in name
            if use_raw:
                Xtr_f, Xte_f = Xtr[IN + SEN], Xte[IN + SEN]
            else:
                Xtr_f, Xte_f = Xtr, Xte
            try:
                import copy
                m = copy.deepcopy(clf)
                m.fit(Xtr_f, y[trn])
                oof[tst] = m.predict_proba(Xte_f)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xte_f)
            except Exception as e:
                print(f"  [skip] {name}: {type(e).__name__} {str(e)[:90]}")
                ok = False
                break
        if not ok:
            continue
        oof_store[name] = oof
        rows.append(_t1_metrics(name, y, oof))

    # rule engine (leakage-free: thresholds fit per fold on train-valid rows)
    oof_rule = np.zeros(len(tr), dtype=bool)
    for trn, tst in cv.split(tr, y):
        tr_fold, te_fold = tr.iloc[trn], tr.iloc[tst]
        vp = sensor_residual_params(tr_fold[tr_fold.Validity_Label == "Valid"])
        Xtr = make_features(tr_fold, vp, dup_flags=dup_all.iloc[trn])
        thr = residual_thresholds(Xtr[tr_fold.Validity_Label == "Valid"])
        Xte = make_features(te_fold, vp, dup_flags=dup_all.iloc[tst])
        oof_rule[tst] = rule_flags(Xte, thr).values
    oof_store["RULE-ENGINE"] = oof_rule.astype(float)
    rows.append(_t1_metrics("RULE-ENGINE", y, oof_rule.astype(float), binary=True))

    # hybrids: rule OR model, for the top learned models
    for name in ["HistGB(eng)", "LightGBM(eng)", "XGBoost(eng)", "CatBoost(eng)"]:
        if name not in oof_store:
            continue
        for thr_p in (0.35, 0.5):
            hybrid = oof_rule | (oof_store[name] > thr_p)
            rows.append(_t1_metrics(f"HYBRID rules|{name}@{thr_p}", y, hybrid.astype(float), binary=True))

    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df, oof_store


def _t1_metrics(name, y, score, binary=False):
    pred = score.astype(bool) if binary else (score > 0.5)
    try:
        auc = roc_auc_score(y, score)
    except Exception:
        auc = np.nan
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": auc,
    }


def eval_task2(tr: pd.DataFrame) -> pd.DataFrame:
    valid = tr[tr.Validity_Label == "Valid"].reset_index(drop=True)
    y = valid["Reference_Parameter"].values
    Xbase = valid[IN].copy()
    Xbase["I2"] = Xbase["Load_Current_A"] ** 2
    Xbase["VI"] = Xbase["Applied_Voltage_kV"] * Xbase["Load_Current_A"]
    Xsens = pd.concat([Xbase, valid[S123]], axis=1)

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    for name, reg in get_regressors().items():
        if name == "OLS-physics(I,I2,Amb,V,t)":
            cols = ["Load_Current_A", "I2", "Ambient_Temperature_C", "Applied_Voltage_kV", "Test_Duration_min"]
            preds = np.zeros(len(valid))
            for trn, tst in cv.split(Xbase):
                m = sm.OLS(y[trn], sm.add_constant(Xbase.iloc[trn][cols])).fit()
                preds[tst] = m.predict(sm.add_constant(Xbase.iloc[tst][cols]))
        else:
            Xd = Xsens if "S123" in name else Xbase
            preds = np.zeros(len(valid))
            try:
                import copy
                for trn, tst in cv.split(Xd):
                    m = copy.deepcopy(reg)
                    m.fit(Xd.iloc[trn], y[trn])
                    preds[tst] = m.predict(Xd.iloc[tst])
            except Exception as e:
                print(f"  [skip] {name}: {type(e).__name__} {str(e)[:90]}")
                continue
        rows.append({
            "model": name,
            "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
            "MAE": float(mean_absolute_error(y, preds)),
            "R2": float(r2_score(y, preds)),
        })

    # seed-bagged GBT (5 seeds averaged) - the cheap ensemble
    try:
        from lightgbm import LGBMRegressor
        oof = np.zeros(len(valid))
        Xd = Xbase
        for trn, tst in cv.split(Xd):
            ps = np.zeros(len(tst))
            for s in range(5):
                m = LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=15,
                                  min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                                  verbose=-1, random_state=SEED + s)
                m.fit(Xd.iloc[trn], y[trn])
                ps += m.predict(Xd.iloc[tst])
            oof[tst] = ps / 5
        rows.append({"model": "LightGBM-5seed-bag(inputs)", **_r2m(y, oof)})
    except ImportError:
        pass

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    # blending: search convex weights over top-3 diverse models via OOF
    return df


def _r2m(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p))}


# ----------------------------------------------------------------------------- main
def main():
    tr, te = load()
    print("Task 1 bake-off (repeated 5x3 stratified CV, out-of-fold):")
    t1, _ = eval_task1(tr)
    print(t1.to_string(index=False))

    print("\nTask 2 bake-off (repeated 5x3 CV, Valid rows only):")
    t2 = eval_task2(tr)
    print(t2.to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    t1.to_csv(RESULTS / "bakeoff_task1.csv", index=False)
    t2.to_csv(RESULTS / "bakeoff_task2.csv", index=False)
    (RESULTS / "bakeoff.json").write_text(
        json.dumps({"task1": t1.to_dict("records"), "task2": t2.to_dict("records")}, indent=2))
    print(f"\nSaved -> {RESULTS/'bakeoff_task1.csv'}, {RESULTS/'bakeoff_task2.csv'}, bakeoff.json")


if __name__ == "__main__":
    main()
