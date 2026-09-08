# PowerNext-AI — Section 2 Verification Report & Implementation Plan
**Date:** 2026-09-07 · **Data:** `CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx` (sheets: README, Training_Data 1000×11, Test_Data 350×9, Sample_Submission 350×3)
**Team name:** pentupbois · All numbers below re-derived from the actual workbook this session.

> **Status note (final):** This is the *early-phase* verification and planning
> document (2026-09-07). Parts C–D record the initial recommendations made
> before the full model bake-off ran; the final deployed choices — the
> probability-mean four-model ensemble for Task 1 and the physics-informed
> hybrid for Task 2 — supersede them and are documented in
> `METHODOLOGY_NOTE.md` with evidence in `results/`. The Part A–B verification
> findings remain fully valid. Task 3's attention rule (Part E) was resolved:
> highest predicted Reference among least-confident Valid rows.

---

## PART A — Verification of Section 2 Claims

### Claim 1 — Duplicate-condition conflicts (~24 rows, all Invalid) — **CONFIRMED, with a crucial correction**
- 12 condition-keys (V, I, Amb, t exact match) have 2 rows each = **24 rows, all labeled Invalid**. ✅
- **Correction:** the pairs are identical in **inputs AND all four sensors** — they disagree **only on Reference_Parameter** (relative spread 13%–91%). This is not "sensors disagree with each other"; it is "same test logged twice with two different certified answers" — at least one is fabricated. The correct rule is therefore **duplicate feature-vector (inputs+sensors)**, which is even cleaner than the handoff's condition-key version.
- On Test_Data: 8 rows form 4 exact-duplicate pairs (identical in every observed column). In training, every such duplicate was Invalid → flag all 8.

### Claim 2 — Sensor mutual agreement separates Valid/Invalid — **CONFIRMED almost exactly**
- Valid (n=866) S1/S2/S3 pairwise Pearson: S1–S2 **0.943**, S1–S3 **0.991**, S2–S3 **0.894** (claimed 0.89–0.99 ✅).
- Invalid (n=134): S1–S2 **0.371**, S1–S3 **0.419**, S2–S3 **0.424** (claimed 0.36–0.42 ✅).
- OLS S3~S1 fit on Valid: R²=0.983; mean |residual| = **0.551 Valid vs 5.17 Invalid** (claimed 0.55 vs 5.24 ✅).
- Bonus: a **logistic regression on raw sensor values has AUC 0.502** — raw values carry *zero* validity signal. The signal is entirely in sensor *relationships*. This empirically confirms the organizers' "unusual value ≠ invalid" doctrine.

### Claim 3 — Feature relevance ranking — **CONFIRMED (Valid rows)**
| Feature | Pearson (Valid) | Spearman (Valid) | Claimed |
|---|---|---|---|
| Load_Current_A | **0.877** | 0.898 | 0.88 ✅ |
| Sensor_S2 | 0.795 | 0.852 | 0.80 ✅ |
| Sensor_S1 | 0.591 | 0.648 | 0.59 ✅ |
| Sensor_S3 | 0.501 | 0.565 | 0.50 ✅ |
| Applied_Voltage_kV | 0.267 | 0.334 | 0.27 ✅ |
| Ambient_Temperature_C | 0.238 | 0.217 | 0.24 ✅ |
| Test_Duration_min | −0.004 | 0.005 | ≈0 ✅ |
| Sensor_S4 | −0.009 | −0.032 | ≈0 ✅ |
- **But the ranking is misleading** (see new findings): Load_Current**²** alone correlates 0.927, and a model with Voltage included makes sensors nearly redundant.

### Claim 4 — Linear model R²≈0.87 with U-shaped residual — **CONFIRMED as stated, REFUTED as interpretation**
- OLS Ref ~ V+I+Amb+S1+S2+S3 on Valid: **R²=0.865** ✅; residual-by-current-decile means run +4.9, +2.2, −0.0, −2.6, −4.4, −3.3, −3.6, −1.6, +1.6, +6.8 — a clear U ✅.
- **Refuted interpretation:** the U is *model misspecification*, not a regime shift. Adding Load_Current² alone lifts R² to 0.980 and flattens every decile residual to |mean| < 1.1. The physics is simply Joule heating (I²) plus linear terms; there is no evidence of a breakpoint regime change in the *Valid* data. The README's "genuine operating-regime change" is expressed as smooth curvature, not a discontinuity — so models must be non-linear, but not piecewise.

### Claim 5 — Housekeeping — **CONFIRMED exactly**
- Missing: S4 29 train / 11 test ✅; S1 6/1, S2 2/3, S3 7/2; **zero** missing in inputs, Reference_Parameter, or labels.
- Label balance: **866 Valid / 134 Invalid (13.4%)** ✅. No ID duplicates, no exact duplicate rows (train), no label whitespace variants.

---

## PART B — New Findings Section 2 Missed

1. **The generating physics is nearly solved.** `Ref ~ I + I² + Ambient + Voltage + Duration` on Valid rows: **R² = 0.980, RMSE 1.52** — using *inputs only, no sensors*. Adding all sensors improves it to R² 0.9801 (negligible). Reference_Parameter is a smooth deterministic function of the four inputs plus ~1.5 units of noise. Coefficients (all p<0.001): `5.70 − 0.340·I + 0.00535·I² + 0.271·Amb + 0.346·V + 0.0218·t`.
2. **Sensor_S4 is provably dead weight**: ≈0 correlation with everything, adds +0.0007 R², and its NaNs never mark a fault (next point). Drop it — this resolves Section 5 fork #2 with data, not judgment.
3. **NaN location is a perfect fault signature**: NaN in S1, S2, or S3 → **Invalid 15/15 times**. NaN in S4 only → **Valid 29/29 times**. Missingness in the real thermometers is a fault; missingness in the junk sensor is benign noise.
4. **Four fault archetypes fully explain all 134 Invalid rows** (each rule test-time-applicable — none uses Reference_Parameter):
   | # | Rule (fit on Valid only) | Train Invalid caught | Train Valid caught |
   |---|---|---|---|
   | R1 | Duplicate feature-vector (inputs+sensors) | 24/24 | 0 |
   | R2 | Any sensor ≤ 0 (negative/exact-zero reading) | 4/4 | 0 |
   | R3 | \|S2−S1-predicted\| or \|S3−S1-predicted\| > 99.5th-pct threshold | ~142 flags | 10 |
   | R4 | NaN in S1/S2/S3 | 15/15 | 0 |
   | Union | | **134/134 (100%)** | **10/866 (1.2%)** |
   → Pure rule-based Task 1 scores ~99.0% train accuracy with total explainability. An HGB classifier on engineered features reaches only 97.7% CV accuracy (F1 0.92) — **the rules beat the learner on this data**.
5. **Reference_Parameter of Invalid rows is itself corrupted** (duplicate pairs share features but conflict on Ref) → never train Task 2 on Invalid rows, and for test rows predicted Invalid, trust the inputs-only model, not the sensors.
6. **Train→test distribution shift: none.** KS p-values 0.32–0.93 on every column; test stays within training range (Voltage marginally, by 0.006 kV). Extrapolation risk is negligible.
7. **Deliverable column name correction:** Sample_Submission header is `Test_ID, Predicted_Reference_Parameter, Validity_Label` — the handoff's "Valid/Invalid" column name is wrong. Sample order equals Test_Data row order.
8. Naive |z|>3 outlier flagging on raw sensors catches 27 Invalid but also 2 Valid — and notably does *not* discriminate against high-current Valid rows (the regime-shift trap barely bites in this file, but the architecture must still avoid it: z-score rules are not in our rule set for exactly this reason).
9. Invalid rows are uniformly scattered (no row-order/ID autocorrelation, r=0.004) — no indexing leakage.

---

## PART C — Model Choices (recommendation, data-driven)

**Task 1 (Valid/Invalid) — Hybrid: deterministic physics rules + learned backstop.**
- Primary: the 4 rules above (R1–R4), thresholds fit only on Valid rows.
- Backstop: HistGradientBoostingClassifier on engineered features (sensor residuals, dup flag, NaN counts) — CV AUC 0.964 — firing only on rows the rules pass, to catch unseen fault modes in the hidden 20% dataset.
- Why: rules are 100%-recall/98.8%-precision and physically explainable; the classifier adds generalization insurance without costing explainability (it only ever adds flags, never removes them).

**Task 2 (Reference_Parameter) — HistGradientBoostingRegressor on INPUTS ONLY (V, I, I², Amb, t).**
- CV RMSE **1.10** vs 1.52 for the interpretable OLS, vs 1.08 with sensors included.
- Sensors add ~0.03 RMSE but bring corruption risk on exactly the rows we're least sure about; inputs-only is immune to sensor faults and predicts every row (Valid or Invalid) with one consistent mechanism.
- The OLS physics model is kept as a documented cross-check (its coefficients go in the methodology note as the engineering explanation).

**Task 3 — programmatic summary; attention rule pending user choice** (composite recommended: highest predicted Ref among low-confidence Valid rows).

---

## PART D — Step-by-Step Implementation Plan

### Step 0 — Ingest & validate (both technical + plain)
- **Tech:** Load workbook values-only via pandas/openpyxl (no formulas/macros exist — verified). Assert shapes (1000×11, 350×9), dtypes, label set {Valid, Invalid}, sample-submission ID alignment. Fixed RNG seed (42) everywhere.
- **Plain:** Open the spreadsheet, check it's the size and shape the README promises, and lock in the exact answer-format template.

### Step 1 — Feature engineering (single shared module)
- **Tech:** `features.py` — compute per-row: `I2`, `S2_residual`, `S3_residual` (OLS fit on Valid training rows only, stored with the model), `dup_feature_vector` flag (computed within the file being scored), `n_missing_S123`, `any_nonpositive`, `S4_z`. Fit NOTHING on test data.
- **Plain:** For every test, compute "do the thermometers agree with each other?" and "has this exact test been logged twice?" — using only lessons learned from trusted historical tests.

### Step 2 — Task 1 classifier
- **Tech:** Apply R1–R4 → `rule_flag`. Train HGB (max_depth=3, 5-fold stratified CV) on engineered features; `learned_flag = proba > 0.35` (best CV F1 threshold). Final: Invalid if `rule_flag OR learned_flag`. Report CV confusion matrix, precision/recall/F1 on Invalid class.
- **Plain:** First apply four common-sense checks; then let a pattern-learner double-check anything the checks passed. A test is flagged if either says so.

### Step 3 — Task 2 regressor
- **Tech:** Train HGB regressor on Valid rows, features [V, I, I², Amb, t]. 5-fold CV RMSE reported. Also fit the OLS physics model for the note. Predict on ALL 350 test rows (inputs only — uniform mechanism).
- **Plain:** Learn the true-temperature formula only from trusted tests, using the dials the lab sets (not the thermometers), then apply it to every new test.

### Step 4 — Task 3 auto-summary
- **Tech:** `summary.json` written by the pipeline: record count, invalid count, min/max/avg predicted Ref, 3 attention Test IDs per chosen rule, ≤100-word methodology blurb, CV metrics, rule hit-counts.
- **Plain:** The program writes its own report card — nothing hand-typed.

### Step 5 — Deliverables & packaging
- `pentupbois.csv` — columns exactly `Test_ID, Predicted_Reference_Parameter, Validity_Label`, row order matching Sample_Submission.
- `summary.json`; `pipeline.py` (+ optional notebook) running end-to-end with zero manual edits; ≤2-page methodology note (approach, parameters + why, fault-detection method, assumptions, digital-twin automation: retrain cadence, drift monitoring on sensor-residual distribution, alerting on rule hits, human-adjudication queue).
- Zip as `powernext-ai-submission.zip`… correction: `pentupbois-submission.zip` per team name. Verify against the Section 6 checklist.

### Verification section (how we'll know it works)
- Re-run pipeline from clean state → identical outputs (seeded).
- Assert output CSV: 350 rows, exact header, ID set/order == Sample_Submission, labels ∈ {Valid, Invalid}, predictions finite & within physical range (~[10, 65]).
- Cross-check Task 1 flags vs the 46 test rows the forensic rules fire on; eyeball the 8 duplicate-pair rows are flagged.
- CV metrics reported in summary.json must match the methodology note.

---

## PART E — Open item for the user
Task 3 "3 Test IDs needing highest attention" — rule choice pending (composite recommended).
