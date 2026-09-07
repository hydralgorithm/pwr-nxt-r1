# PowerNext-AI — Methodology Note
**Team:** pentupbois · **Challenge:** CPRI State-Level Hackathon, Screening Round — Black-Box Test Bench Challenge

## 1. Approach

The pipeline treats the three tasks as one coherent inference problem built on a
single insight verified against the training data: **records are invalid because
their internal story contradicts itself, not because any single value looks
unusual.** A logistic regression on raw sensor readings achieves AUC 0.502
(chance level) for validity, while relationship features (inter-sensor
residuals, duplicate structure, missingness pattern) raise AUC above 0.99. This
is the organizers' doctrine — "an unusual value is not necessarily an invalid
value" — made quantitative, and it is why the design never uses univariate
outlier thresholds.

**Task 1 (fault detection) — ensemble of four model families on relationship
features.** Every record is described by engineered consistency features:
residuals of S2 and S3 against an S1-agreement model fitted on Valid rows only
(R²=0.89/0.98), duplicate-logging structure, missingness counts per sensor,
non-positive readings, and the raw operating conditions. Four diverse learners
— RandomForest, RBF-SVM, LightGBM, CatBoost — are trained on these features
and their Invalid-probabilities averaged. Under repeated 5×3-fold stratified
cross-validation (all preprocessing refit in-fold) the deployed ensemble
scores **accuracy 0.999, precision 1.000, recall 0.993, F1 0.996, AUC 1.000**.
Four physically-motivated fault rules (below) are computed as an
explainability layer and as ensemble features:

| Rule | Physical meaning |
|---|---|
| R1: duplicated feature-vector (inputs + all sensors) | same test logged twice with conflicting certified answers (24/24 Invalid in training) |
| R2: any sensor reading ≤ 0 | physically impossible temperature rise (4/4 Invalid) |
| R3: NaN in S1/S2/S3 | real thermometer dropped out (15/15 Invalid; NaN in S4 alone: 29/29 Valid) |
| R4: inter-sensor residual > 99.5th-pct of Valid | thermometers disagree with each other |

**Task 2 (Reference_Parameter regression) — physics-informed hybrid on
operating inputs only.** The target is an almost-deterministic function of the
inputs: a linear model `Ref ~ I + I² + Ambient + Voltage + Duration` (Joule
heating plus linear terms) already reaches R² = 0.980 using **no sensors at
all**. Sensors add only ~0.03 RMSE while being exactly the channels corrupted
in Invalid rows — so the production model deliberately excludes them, making
every prediction (Valid or Invalid) come from one corruption-immune mechanism:
the physical test conditions. The deployed model is a **physics-informed
hybrid**: the OLS physics term carries the extrapolatable physics, and a
5-seed bagged CatBoost (depth 4, lr 0.04, 800 iterations) learns only the
bounded residual correction. Cross-validated on Valid rows (repeated
5×3-fold): **RMSE 0.41 °C, MAE 0.27, R² 0.9985** — better than pure CatBoost
in-range (0.52) *and* far safer out of range: tree models plateau at the
training boundary and cannot extrapolate, so under a stress test that withholds
the 85–110 A band, pure CatBoost's RMSE on that band is 11.4 while the hybrid
holds 3.5. Since the hidden second dataset may stress-test beyond nominal
currents, this structure was adopted deliberately: base model extrapolates,
correction stays bounded.

**Task 3 (auto-summary).** All counts and statistics are computed
programmatically from the prediction table. The "3 Test IDs needing highest
attention" use a stated composite rule: *highest predicted Reference_Parameter
among Valid rows whose ensemble validity probability falls in the
least-confident quartile* — the thermally riskiest tests the model is least
sure about.

## 2. Model selection (how the winners were chosen)

A full bake-off under identical repeated 5×3-fold CV compared 17 candidates
per task (results in `results/bakeoff*.csv`):

- **Task 1**: RF/SVM ensemble 0.999 acc / F1 0.996 > LightGBM 0.996 > CatBoost
  0.997-acc variants > hybrid rule-OR-model 0.990 (full recall, lower
  precision) > rules alone 0.952 > raw-feature logistic 0.562 (chance).
  The probability-mean ensemble was deployed over single-model RF because it
  ties RF on CV while adding model-family diversity — insurance for the
  unseen second dataset worth 20% of the grade.
- **Task 2**: physics-hybrid 0.415 RMSE > pure CatBoost-bag 0.517 > CatBoost
  0.555 > XGBoost 0.790 > LightGBM 1.09 > SVR 1.07 > RF 1.36 > OLS-physics
  1.54 > kNN 2.22. Sensor-including variants were tested (LightGBM inputs+S123:
  1.083) and rejected: no gain, added corruption exposure.

## 2b. Robustness stress tests (hidden-dataset insurance)

- **Task 1 error analysis (out-of-fold):** exactly **1 missed Invalid of 134**
  (TRN-0837 — a borderline ~11% S3 miscalibration sitting inside the Valid
  residual spread) and **0 false-flagged Valid rows**. Catching that one row
  would require thresholds loose enough to flag many Valid rows; rejected as
  a bad trade for unseen data.
- **Task 2 extrapolation:** trained without the 85–110 A band, pure CatBoost
  predicts that band at RMSE 11.4 (tree plateau); the deployed hybrid holds
  3.5, and the OLS base alone 3.9. Voltage-band stress: hybrid 0.42 vs pure
  CatBoost 1.15. The same test on the remaining two operating dimensions —
  high-ambient band (40.5–55 °C) and long-duration band (47.5–60 min) —
  gives hybrid 2.47 vs 3.51 and 0.53 vs 0.65 respectively: the hybrid wins
  on every stress axis measured, so the choice is not an artifact of the
  current-range test.
- **Task 1 label-noise sensitivity:** with 5% of training labels randomly
  flipped, CV accuracy vs the corrupted labels degrades gracefully to 0.946 —
  the ensemble is not brittle to label noise.
- **Noise-floor diagnostic:** the out-of-fold Task 2 residual correlates at
  most |r| = 0.06 with any operating input — no learnable structure remains,
  so the CV RMSE of 0.41 °C sits at the irreducible noise floor of the data.
  (The 12 duplicate groups' Reference values disagree by ~12 °C — this is
  corrupted-label spread in Invalid rows, not process noise, and is the
  quantitative reason they are excluded from regression training.)
- **Train/serve consistency:** Task 1 models are fit and scored on identical
  missing-value semantics (native NaN), verified after an audit found and
  removed a fit-vs-predict imputation mismatch.
- **Self-reported uncertainty:** the pipeline emits a 90% prediction error
  band (0.53 °C from out-of-fold residuals) and an automatic input-drift
  check (Kolmogorov–Smirnov per operating input) in `summary.json`, so a
  future run on shifted data announces it instead of silently extrapolating.

## 3. Important parameters and why

- **Residual thresholds: 99.5th percentile of the Valid residual distribution**
  (not a round number): at 866 Valid rows this is the tightest threshold that
  flags ≲5 Valid rows — anything looser trades precision for no recall gain.
- **Ensemble decision threshold 0.5** on the mean probability: CV-optimal
  (precision 1.000, recall 0.993); the 13.4% class imbalance is handled by
  class weights inside each member.
- **CatBoost depth 4 / lr 0.04 / 800 iters; 5-seed bag**: best of the depth
  4/5/6 sweep; seed-bagging cut RMSE 0.555→0.519. Shallow trees generalize on
  ~850 rows.
- **RNG seed 42 everywhere**; the pipeline is bit-for-bit deterministic
  (verified by re-run hash comparison).

## 4. Abnormal-data detection method

Relationship-based, never value-based. Concretely: fit S2~S1 and S3~S1 on
Valid rows only, then measure how far each row's actual S2/S3 sit from that
mutual-agreement manifold. Genuine regime shifts (high current) move **all
three sensors together** — residuals stay small — so a genuinely hot test is
never flagged; a faulty sensor breaks the correlation and is caught. Duplicate
logging and dropped-out channels are structural checks. Missing values are
*not* imputed for Task 1 (the missingness itself is the signal); Task 2 never
touches the sensors, so sensor missingness is irrelevant to prediction.

## 5. Assumptions

1. The Reference_Parameter of Invalid rows is itself untrustworthy (duplicate
   pairs share identical features but conflict on the target), so Invalid rows
   are excluded from regression training entirely.
2. Test conditions in the hidden dataset stay within (or near) the training
   operating envelope; verified for the provided test file (KS p ≥ 0.32 on
   every column, no range extension beyond 0.006 kV).
3. Sensor_S4 is noise with respect to both tasks (|r| < 0.03 with the target,
   ΔR² < 0.001 when included) and is excluded from modeling but retained in
   the raw data.
4. Engineers' Valid/Invalid labels are ground truth.

## 6. Digital-twin automation steps

1. **Ingest:** the pipeline is a pure function of an xlsx/csv dump — point it
   at a scheduled export from the lab's data historian (SCADA/LIMS).
2. **Score:** every new test record is scored in milliseconds (ensemble
   inference is trivial at this scale) — per-batch or streaming.
3. **Alert:** R1/R2 hits (duplicate logging, impossible values) page an
   engineer immediately; R3/R4 and high ensemble probability enter a review
   queue ordered by P(Invalid).
4. **Monitor drift:** track the Valid-row residual distribution of S2~S1 and
   S3~S1 over time; a shift means sensor degradation or a genuine regime
   change — recompute thresholds per rolling window automatically.
5. **Retrain:** engineer adjudications feed back as labels; retrain weekly or
   every N adjudicated records, with the fixed CV protocol as a deployment
   gate (no model ships if CV regresses).
6. **Audit:** `summary.json` written on every run gives machine-readable
   counts, extremes, attention IDs, and validation metrics for trend
   dashboards.

## 7. Reproducibility

- `src/bakeoff.py`, `src/bakeoff2.py` — full model comparison (both tasks,
  identical folds, results in `results/`).
- `src/pipeline.py` — one-command end-to-end run producing all deliverables;
  verified deterministic (identical output hash on re-run).
- No manual per-record edits anywhere; all thresholds derived from data at
  run time; fixed seeds.
- One caveat recorded for completeness: TabPFN was evaluated as a candidate
  but requires an interactive license login and was skipped; all other
  candidates ran under identical protocol.
