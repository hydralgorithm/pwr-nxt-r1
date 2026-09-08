"""Round 2: ensemble and tuning experiments on top of the round-1 winners.

Task 1: probability-averaging ensembles and seed-bagged RandomForest.
Task 2: CatBoost seed-bags, blends with XGBoost/LightGBM, depth tuning.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
RESULTS = ROOT / "results"
SEED = 42
IN = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min"]
SEN = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
S123 = SEN[:3]

import importlib, sys
sys.path.insert(0, str(ROOT / "src"))
import bakeoff as B  # reuse feature engineering


def t1_metrics(name, y, score):
    pred = score > 0.5
    return {"model": name, "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, score)}


def task1(tr):
    y = (tr["Validity_Label"] == "Invalid").astype(int).values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    dup_all = tr.duplicated(subset=IN + SEN, keep=False)

    def build(trn, tst):
        tr_fold, te_fold = tr.iloc[trn], tr.iloc[tst]
        vp = B.sensor_residual_params(tr_fold[tr_fold.Validity_Label == "Valid"])
        Xtr = B.make_features(tr_fold, vp, dup_flags=dup_all.iloc[trn])
        Xte = B.make_features(te_fold, vp, dup_flags=dup_all.iloc[tst])
        return Xtr, Xte

    def rf(seed):
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed,
                                      n_jobs=-1)

    def svm():
        return make_pipeline(SimpleImputer(), StandardScaler(),
                             SVC(probability=True, class_weight="balanced", random_state=SEED))

    def lgb():
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              class_weight="balanced", verbose=-1, random_state=SEED)

    def cat():
        return CatBoostClassifier(iterations=500, learning_rate=0.05, depth=4,
                                  verbose=0, random_seed=SEED)

    runs = {
        "RF(eng)": lambda Xtr, Xte, ytr: _fit_pred(rf(SEED), Xtr, Xte, ytr),
        "RF-5seed-bag(eng)": lambda Xtr, Xte, ytr: _bag(rf, 5, Xtr, Xte, ytr),
        "SVM-RBF(eng)": lambda Xtr, Xte, ytr: _fit_pred(svm(), Xtr, Xte, ytr),
        "ENS mean(RF+SVM+LGBM+CAT)": None,
        "ENS mean(RF+SVM)": None,
        "ENS mean(RF+LGBM+CAT)": None,
    }
    probs = {k: np.zeros(len(tr)) for k in runs}
    for trn, tst in cv.split(tr, y):
        Xtr, Xte = build(trn, tst)
        p = {}
        p["RF"] = _fit_pred(rf(SEED), Xtr, Xte, y[trn])
        p["RFbag"] = _bag(rf, 5, Xtr, Xte, y[trn])
        p["SVM"] = _fit_pred(svm(), Xtr, Xte, y[trn])
        p["LGBM"] = _fit_pred(lgb(), Xtr, Xte, y[trn])
        p["CAT"] = _fit_pred(cat(), Xtr, Xte, y[trn])
        probs["RF(eng)"][tst] = p["RF"]
        probs["RF-5seed-bag(eng)"][tst] = p["RFbag"]
        probs["SVM-RBF(eng)"][tst] = p["SVM"]
        probs["ENS mean(RF+SVM+LGBM+CAT)"][tst] = (p["RF"] + p["SVM"] + p["LGBM"] + p["CAT"]) / 4
        probs["ENS mean(RF+SVM)"][tst] = (p["RF"] + p["SVM"]) / 2
        probs["ENS mean(RF+LGBM+CAT)"][tst] = (p["RF"] + p["LGBM"] + p["CAT"]) / 3

    return pd.DataFrame([t1_metrics(k, y, v) for k, v in probs.items()]).sort_values("F1", ascending=False)


def _fit_pred(model, Xtr, Xte, ytr):
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    return model.predict(Xte)


def _bag(factory, n, Xtr, Xte, ytr):
    out = np.zeros(len(Xte))
    for s in range(n):
        out += _fit_pred(factory(SEED + s), Xtr, Xte, ytr)
    return out / n


def task2(tr):
    valid = tr[tr.Validity_Label == "Valid"].reset_index(drop=True)
    y = valid["Reference_Parameter"].values
    X = valid[IN].copy()
    X["I2"] = X["Load_Current_A"] ** 2
    X["VI"] = X["Applied_Voltage_kV"] * X["Load_Current_A"]
    cols = list(X.columns)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    def cat(depth=4, it=800, seed=SEED):
        return CatBoostRegressor(iterations=it, learning_rate=0.04, depth=depth,
                                 verbose=0, random_seed=seed)

    def xgb():
        return XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=3,
                            subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                            random_state=SEED)

    def lgb():
        return LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=15,
                             min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                             verbose=-1, random_state=SEED)

    oof = {k: np.zeros(len(valid)) for k in [
        "CatBoost-d4", "CatBoost-d5", "CatBoost-d6", "CatBoost-5seed-bag-d4",
        "XGBoost", "LightGBM", "blend CAT+XGB", "blend CAT+XGB+LGBM",
        "blend 0.5*CAT+0.5*XGB", "blend 0.7*CAT+0.3*XGB"]}

    for trn, tst in cv.split(X):
        Xtr, Xte, ytr = X.iloc[trn], X.iloc[tst], y[trn]
        pcat4 = _fit_pred(cat(4), Xtr, Xte, ytr)
        pcat5 = _fit_pred(cat(5), Xtr, Xte, ytr)
        pcat6 = _fit_pred(cat(6), Xtr, Xte, ytr)
        pcatbag = np.mean([_fit_pred(cat(4, seed=SEED + s), Xtr, Xte, ytr) for s in range(5)], axis=0)
        pxgb = _fit_pred(xgb(), Xtr, Xte, ytr)
        plgb = _fit_pred(lgb(), Xtr, Xte, ytr)
        oof["CatBoost-d4"][tst] = pcat4
        oof["CatBoost-d5"][tst] = pcat5
        oof["CatBoost-d6"][tst] = pcat6
        oof["CatBoost-5seed-bag-d4"][tst] = pcatbag
        oof["XGBoost"][tst] = pxgb
        oof["LightGBM"][tst] = plgb
        oof["blend CAT+XGB"][tst] = 0.5 * pcat4 + 0.5 * pxgb
        oof["blend CAT+XGB+LGBM"][tst] = (pcat4 + pxgb + plgb) / 3
        oof["blend 0.5*CAT+0.5*XGB"][tst] = 0.5 * pcat4 + 0.5 * pxgb
        oof["blend 0.7*CAT+0.3*XGB"][tst] = 0.7 * pcat4 + 0.3 * pxgb

    rows = [{"model": k, "RMSE": float(np.sqrt(mean_squared_error(y, v))),
             "MAE": float(mean_absolute_error(y, v)), "R2": float(r2_score(y, v))}
            for k, v in oof.items()]
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def main():
    tr = pd.read_excel(XLSX, sheet_name="Training_Data")
    print("ROUND 2 — Task 1 ensembles:")
    t1 = task1(tr)
    print(t1.to_string(index=False))
    print("\nROUND 2 — Task 2 CatBoost/ensembles:")
    t2 = task2(tr)
    print(t2.to_string(index=False))
    t1.to_csv(RESULTS / "bakeoff2_task1.csv", index=False)
    t2.to_csv(RESULTS / "bakeoff2_task2.csv", index=False)
    (RESULTS / "bakeoff2.json").write_text(json.dumps(
        {"task1": t1.to_dict("records"), "task2": t2.to_dict("records")}, indent=2))
    print("\nSaved round-2 results.")


if __name__ == "__main__":
    main()
