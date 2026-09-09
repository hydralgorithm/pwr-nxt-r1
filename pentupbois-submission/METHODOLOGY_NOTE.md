# PowerNext-AI — Methodology Note
**Team:** pentupbois · **Challenge:** CPRI State-Level Hackathon, Screening Round — Black-Box Test Bench Challenge

## 1. Approach

Records are invalid because their internal story contradicts itself, not
because any single value looks unusual: logistic regression on raw sensor
readings achieves AUC 0.502 (chance) for validity, while relationship
features exceed 0.999 — the organizers' doctrine ("an unusual value is
not necessarily an invalid value") made quantitative; the design never
uses univariate outlier thresholds.

**Task 1 (fault detection).** Each record is described by engineered
consistency features: residuals of S2 and S3 against an S1-agreement model
fitted on Valid rows only, duplicate-logging structure, missingness counts
per sensor, non-positive readings, and the raw operating conditions. Four
diverse learners — RandomForest, RBF-SVM, LightGBM, CatBoost — are trained
on these features and their Invalid-probabilities averaged. Under repeated
5×3-fold stratified cross-validation (all preprocessing refit in-fold) the
ensemble scores **accuracy 0.999, precision 1.000, recall 0.993, F1 0.996,
AUC ≈ 1.000**, missing exactly 1 of 134 Invalid rows (a borderline
miscalibration) with zero false flags.
Four physically-motivated rules are computed as an explainability layer and
as ensemble features:

| Rule | Physical meaning |
|---|---|
| R1: duplicated feature-vector (inputs + all sensors) | same test logged twice with conflicting certified answers (24/24 Invalid) |
| R2: any sensor reading ≤ 0 | physically impossible temperature rise (4/4 Invalid) |
| R3: NaN in S1/S2/S3 | real thermometer dropped out (15/15 Invalid; NaN in S4 alone: 29/29 Valid) |
| R4: inter-sensor residual > 99.5th-pct of Valid | thermometers disagree with each other |

**Task 2 (Reference_Parameter regression).** The target is almost
deterministic in the inputs: a cubic polynomial in the four operating
inputs (Joule heating) reaches R² = 0.987 using **no sensors at all**.
Sensors are exactly the corrupted channels in Invalid rows, so the model
deliberately uses operating inputs only — every prediction (Valid or
Invalid) comes from one corruption-immune mechanism. The deployed
model is a **physics-informed hybrid**: an additive cubic Ridge base (each
input to powers 1–3) carries the extrapolatable physics, and a 5-seed
bagged CatBoost (depth 4) learns only the bounded residual correction. CV
on Valid rows: **RMSE 0.37 °C, MAE 0.25, R² 0.9988** — better than pure
CatBoost in range (0.52) and far safer beyond it. The out-of-fold residual
correlates at most |r| = 0.08 with any input: the score sits at the data's
irreducible noise floor.

**Task 3 (auto-summary).** Counts and statistics are computed
programmatically from the prediction table. The three attention Test IDs
are the highest predicted Reference_Parameter among Valid rows whose
ensemble validity probability falls in the least-confident quartile — the
thermally riskiest tests the model is least sure about.

## 2. Model selection

A bake-off under an identical repeated-CV protocol compared 17 candidates
per task. Task 1: ensemble 0.999 accuracy / F1 0.996 > LightGBM 0.996 >
CatBoost 0.997 > rules alone 0.952 > raw-feature logistic 0.562 (chance).
Task 2: hybrid 0.415 RMSE > CatBoost-bag 0.517 > XGBoost 0.790 >
OLS-physics 1.54 > kNN 2.22; sensor-including variants were tested (1.08)
and rejected — no gain, added corruption exposure. Upgrading the base to
additive-cubic ridge then cut the hybrid to 0.371 RMSE and halved
current-axis extrapolation error.

## 3. Robustness stress tests (hidden-dataset insurance)

- **Extrapolation** (train without the top quintile of each operating
  axis): the hybrid beats pure CatBoost on current (1.40 vs 9.36 — tree
  models plateau at the training boundary), voltage (0.35 vs 0.76),
  ambient (2.28 vs 3.51) and duration (0.52 vs 0.65).
- **Sensor invariance:** blanking all four sensors changes no Task 2
  prediction (bit-identical across all 350 rows).
- **Label noise:** 5% of training labels randomly flipped: accuracy
  degrades gracefully to 0.946.
- **Self-monitoring:** the pipeline emits a 90% prediction error band
  (±0.51 °C) and an automatic input-drift check (Kolmogorov–Smirnov per
  operating input) in summary.json; a run on shifted data announces itself.

## 4. Important parameters and why

- **Residual thresholds at the 99.5th percentile of the Valid residual
  distribution:** the tightest threshold that flags ≲5 of 866 Valid rows.
- **Ensemble decision threshold 0.5** on the mean probability — an F1-max
  threshold search was tested and rejected (equal F1, adds a false
  positive, flips zero test labels); the 13.4% class imbalance is handled
  by class weights inside each member.
- **CatBoost depth 4 / lr 0.04 / 800 iterations, 5-seed bag:** best of the
  depth sweep; shallow trees generalize on ~850 rows.
- **RNG seed 42 everywhere**; the pipeline is bit-for-bit deterministic
  (verified by re-run hash comparison).

## 5. Method used for detecting abnormal data

Relationship-based, never value-based. Concretely: fit S2~S1 and S3~S1 on
Valid rows only, then measure how far each row's actual sensors sit from
that mutual-agreement manifold. Genuine regime shifts move all sensors
together (small residuals — a genuinely hot test is never flagged); a
faulty sensor breaks the correlation and is caught.
Duplicate logging and dropped-out channels are structural checks. Missing
values are not imputed for Task 1 (the missingness itself is the signal);
Task 2 never touches the sensors, so sensor missingness cannot affect a
prediction.

## 6. Assumptions

1. The Reference_Parameter of Invalid rows is untrustworthy (duplicate
   pairs share features but conflict on the target), so regression trains
   on Valid rows only.
2. Hidden-dataset conditions stay within (or near) the training envelope;
   the physics base covers near-range extension.
3. Sensor_S4 is noise with respect to both tasks (|r| < 0.03 with the
   target) and is excluded from modeling.
4. Engineers' Valid/Invalid labels are ground truth.

## 7. Digital-twin automation steps

1. **Ingest:** the pipeline is a pure function of an xlsx/csv dump — point
   it at a scheduled export from the lab's data historian (SCADA/LIMS).
2. **Score:** records are scored in milliseconds, per batch or streaming.
3. **Alert:** R1/R2 hits (duplicate logging, impossible values) page an
   engineer immediately; R3/R4 and high ensemble probability enter a
   review queue ordered by P(Invalid).
4. **Monitor drift:** track the Valid-row residual distribution over
   time; a shift means sensor degradation or a genuine regime change —
   thresholds recompute per rolling window automatically.
5. **Retrain:** engineer adjudications feed back as labels; the fixed CV
   protocol is the deployment gate (no model ships if CV regresses).
6. **Audit:** every run writes machine-readable counts, extremes,
   attention IDs and validation metrics for dashboards.

## 8. Reproducibility

`src/pipeline.py` is the complete solution. Place the participant workbook
(`CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx`) next to `src/` and
run `python src/pipeline.py` — it regenerates both deliverable files in
place (requires pandas, numpy, scikit-learn, lightgbm, catboost, scipy,
openpyxl); no manual edits, thresholds derive from the data at run time;
fixed seeds; outputs verified identical on re-run.
