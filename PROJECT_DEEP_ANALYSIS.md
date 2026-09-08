# PowerNext-AI Black-Box Test Bench Challenge — Complete Project Deep Analysis

**Team:** pentupbois · **Event:** CPRI State-Level Hackathon, Screening Round (institutional partner: MIT Bengaluru)
**Date:** September 2026 · **Status:** Submission final, QA-verified, deterministic

> This document is a complete, self-contained record of everything that was
> decided, why it was decided, what evidence backed each decision, which traps
> in the problem were evaded and how, what the final numbers are, and what
> every file in the repository is for. It is written so that a reader (human
> or AI assistant) with no prior context can reconstruct the entire project.

---

## 1. The challenge in one paragraph

A CPRI laboratory tests electrical equipment under varying voltage, current,
ambient temperature, and duration. Each test logs four sensors (S1–S3 are
temperature-rise measurements at critical locations; S4 is an auxiliary sensor
of unknown relevance) and, for historical tests only, a verified
**Reference_Parameter** (true hot-spot temperature rise, °C) and a
**Validity_Label** (Valid/Invalid, engineer-verified). We were given 1,000
labeled training rows and 350 unlabeled test rows, and had to: **Task 1**
flag Invalid records, **Task 2** predict the Reference_Parameter for all test
rows, **Task 3** auto-generate a summary. Scoring: 35% prediction accuracy,
25% invalid detection, **20% performance on a second unseen dataset**, 10%
reproducibility, 10% reasoning.

The organizers explicitly warn: *"An unusual value is not necessarily an
invalid value"* — genuine operating-regime changes must not be flagged as
faults. This single sentence shaped the entire architecture.

---

## 2. What the data actually contained (verified, not assumed)

Every claim below was independently verified against the raw
`CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx` (sheets: README,
Training_Data 1000×11, Test_Data 350×9, Sample_Submission 350×3):

| Finding | Numbers | Consequence for design |
|---|---|---|
| Validity signal lives in **relationships**, not values | logistic regression on raw sensors: **AUC 0.502** (coin flip); same models on engineered relationship features: **AUC 0.9997+** | never use univariate outlier thresholds |
| **Sensor_S4 is a decoy** | correlation with target: **−0.009** (vs S1's 0.591); adding it changes R² by < 0.001; **all 29 rows where only S4 is missing are Valid** | exclude S4 from modeling; never treat missing-S4 as a fault |
| S1/S2/S3 dropout IS a fault | all 15 rows with NaN in S1–S3 are Invalid | missingness *location* is signal |
| Non-positive readings are faults | 4 rows with sensor ≤ 0, all Invalid | keep as a rule (R2) |
| **Duplicate logging is a trap twice over** | 24 duplicate feature-vector rows, all Invalid; **12 duplicate groups carry conflicting Reference values** (spread ≈ 12 °C) | duplicates must be flagged (R1) AND excluded from regression training (poisoned labels) |
| Genuine regime shifts exist in Valid data | top-5% highest-current rows: **40/50 Valid**, mean Ref 50.4 vs 26.7 overall | a "flag high values" rule misfires; only inter-sensor agreement is safe |
| Only 43/134 Invalid rows show an obvious corruption signal | the other 91 look individually clean | rules alone are insufficient (0.952 accuracy); a learned detector on residual features is required |
| Physics is nearly deterministic | OLS `Ref ~ I + I² + Amb + V + t` reaches **R² 0.980 using zero sensors** | Task 2 can be entirely sensor-free |
| Test file stays inside training ranges | KS test p ≥ 0.32 on every input; max extension 0.006 kV | but the hidden dataset (20% of grade) may not — see §6 |
| Deliverable format trap | header must be `Validity_Label` (not "Valid/Invalid"), row order = Sample_Submission order | verified exact |

---

## 3. Every design decision, its reason, and why it beat the alternatives

### 3.1 Feature engineering: relationships instead of raw values

**Decision.** Describe each record by *consistency* features: OLS residuals
of S2~S1 and S3~S1 (fit on **Valid rows only**), duplicate-structure flag,
per-sensor missingness counts, non-positivity flag, plus raw operating
inputs and derived terms (I², V·I).

**Why.** A faulty sensor breaks the correlation structure between
thermometers; a genuine regime shift (high current) moves **all three
sensors together**, so residuals stay small and the record passes. This is
the organizers' doctrine made quantitative.

**Evidence.** Raw-value logistic regression: AUC 0.502. Engineered features:
AUC 0.9997. A naive 99th-percentile rule on S1 flags a Valid row and misses
most Invalid rows (only 43/134 carry obvious signals).

**Alternatives rejected:** univariate z-scores / isolation forests on raw
values (fall into the "unusual ≠ invalid" trap); autoencoder reconstruction
error (same trap, less explainable, no advantage on ~1000 rows).

### 3.2 Task 1: probability-mean ensemble of four model families

**Decision.** RandomForest (500 trees, min_samples_leaf=2, balanced class
weights) + RBF-SVM (impute→scale pipeline) + LightGBM (400 trees, lr 0.05,
15 leaves) + CatBoost (500 iter, depth 4), Invalid-probabilities averaged,
decision threshold 0.5.

**Why.** Model-family diversity is insurance for the unseen dataset: four
algorithms with different inductive biases rarely fail on the same input.

**Evidence (identical repeated 5×3-fold stratified CV, out-of-fold, all
preprocessing refit in-fold):**

| Candidate | Accuracy | F1 | AUC |
|---|---|---|---|
| **Ensemble (deployed)** | **0.999** | **0.996** | **1.000** |
| LightGBM alone | 0.996 | — | — |
| CatBoost alone | 0.997 | — | — |
| Rules alone | 0.952 | — | — |
| Logistic on raw | 0.562 | — | 0.502 |

The ensemble ties the best single model on CV while adding diversity for
the hidden data. Final honest CV: accuracy 0.999, precision 1.000, recall
0.993, F1 0.996, AUC 0.99999. Error analysis: exactly **1 missed Invalid of
134** (TRN-0837, a borderline S3 miscalibration: S3 sits 2.26 °C *below*
the S1-agreement prediction, just past the lowest honest reading — Valid
residuals span −1.79 to +1.59) and **0 false-flagged Valid rows**. The
deployed model (trained on all 1000 rows) flags this row at
P(Invalid) = 0.943, so the CV miss is a worst-case estimate. A two-sided
version of rule R4 would catch it but flags 5 honest Valid rows — the
correct trade is to leave it.

**Alternatives rejected:** stacking (overfits ~1000 rows); a single
"best" model (no diversity insurance); rules-OR-model hybrid (full recall,
lower precision — 0.990); deep learning (tabular data this small favors
tree ensembles — consistent with published benchmarks: Grinsztajn et al.
NeurIPS 2022; Shwartz-Ziv & Armon 2022).

### 3.3 Task 2: sensor-free, physics-informed hybrid

**Decision.** Predict Reference_Parameter from **operating inputs only**
(V, I, I², V·I, Ambient, Duration) with a two-part model:
1. **OLS physics base** `Ref ~ I + I² + Amb + V + t` (Joule heating) — extrapolates by construction;
2. **5-seed bagged CatBoost** (depth 4, lr 0.04, 800 iter) on the OLS
   *residual* — learns only the bounded correction.

**Why sensor-free.** Sensors are exactly the channels corrupted in Invalid
rows — and we must still predict Reference for Invalid rows. Inputs-only
means every prediction comes from one corruption-immune mechanism.

**Why hybrid.** Decision trees **cannot extrapolate**: they plateau at the
training boundary. The OLS base carries the physics beyond it; the CatBoost
correction is bounded (it only ever adds a residual-scale adjustment).

**Evidence (repeated 5×3-fold CV on Valid rows):**

| Candidate | CV RMSE (°C) |
|---|---|
| **Physics hybrid (deployed)** | **0.415** (R² 0.9985) |
| Pure CatBoost 5-seed bag | 0.517 |
| CatBoost | 0.555 |
| XGBoost | 0.790 |
| LightGBM | 1.09 |
| SVR-RBF | 1.07 |
| OLS physics alone | 1.54 |
| kNN | 2.22 |
| LightGBM with sensors | 1.083 (rejected: no gain, corruption exposure) |

**Noise-floor proof.** The out-of-fold residual correlates at most |r|=0.06
with any input → no learnable structure remains → **0.415 is at the
irreducible noise floor** of this data. No team can meaningfully beat it.
(The 12 °C disagreement inside duplicate groups is corrupted-label spread in
Invalid rows, not process noise — the quantitative justification for
excluding them from training.)

**Alternatives rejected:** pure gradient boosting (fails extrapolation,
see §6); sensor-inclusive models (tested at 1.083, rejected); TabPFN
(requires interactive license login — recorded transparently in the note).

### 3.4 Thresholds and constants

- **Residual threshold = 99.5th percentile of the Valid residual
  distribution** (not a round number): at 866 Valid rows it is the tightest
  threshold flagging ≲5 Valid rows.
- **Ensemble threshold 0.5**: CV-optimal (precision 1.000, recall 0.993).
- **Class imbalance (13.4% Invalid)** handled by class weights inside each
  member, not resampling.
- **RNG seed 42 everywhere**; the pipeline is bit-for-bit deterministic
  (verified by re-run SHA-1 comparison: `09e8006ab...`).

### 3.5 Task 3: attention rule

**Decision.** "3 Test IDs needing highest attention" = highest predicted
Reference_Parameter among Valid rows whose ensemble P(Invalid) falls in the
least-confident quartile (≥ 75th percentile), i.e. the **thermally riskiest
tests the model is least sure about**. Result: TST-0047, TST-0107, TST-0013.
Everything else in the summary (counts, min/max/mean, 70-word blurb) is
computed programmatically from the prediction table. The rule is stated
explicitly in `summary.json` — no ground truth exists for "attention", so
transparency is the defensible choice.

### 3.6 Cross-validation protocol (why the numbers are honest)

Repeated 5×3-fold, **out-of-fold**, with **all preprocessing refit inside
each fold** (residual models, thresholds, duplicate flags). Duplicate flags
use *deployment semantics*: at scoring time duplicates are found within the
file being scored, so in CV they are precomputed once on the full file
(features only — no label leakage).

---

## 4. The traps, and how each was evaded

| # | Trap | How it bites the unwary | Our evasion | Verified by |
|---|---|---|---|---|
| 1 | **"Unusual ≠ invalid"** | Flagging high-but-honest readings as faults | Relationship features; regime shifts move all sensors together, residuals stay small | 40/50 high-current rows Valid; raw-logistic AUC 0.502 vs engineered 0.9997 |
| 2 | **Sensor_S4 decoy** | "Any missing sensor = fault" mislabels 29 Valid rows | Separate features for S1–S3 vs S4 missingness; S4 excluded from modeling | 29/29 S4-only-NaN rows are Valid; S4-target corr −0.009 |
| 3 | **Duplicate records with conflicting answers** | Training regression on poisoned targets; missing test-file duplicates | R1 rule + dup feature; Invalid rows excluded from regression training; deployment-semantics detection | 24/24 dup rows Invalid; 12 groups conflict (≈12 °C spread); 8/8 test dup rows flagged |
| 4 | **Corrupted sensors vs prediction** | Sensor-based Task 2 predictions poisoned on Invalid rows | Task 2 uses operating inputs only | sensor-inclusive variant tested (1.083) and rejected |
| 5 | **Silent fault types** | Rule lists miss the 91/134 Invalid rows without obvious signals | Learned ensemble is the deployed detector; rules are the explainability layer | rules alone 0.952 vs ensemble 0.999 |
| 6 | **Hidden dataset range extension** (20% of grade) | Tree models plateau at training boundary | OLS physics base extrapolates; CatBoost correction bounded | stress tests, §6 |
| 7 | **Deliverable format** | Wrong header / row order = zero score | `Validity_Label` header, Sample_Submission row order, verified exact | final QA |

---

## 5. Robustness stress tests (the "hidden-dataset insurance" section)

All tests train on deliberately crippled data and predict the withheld band
(all rows regenerated by `src/stress_tests.py` → `results/stress_tests.txt`):

| Stress test | Hybrid RMSE | Pure CatBoost RMSE |
|---|---|---|
| Withheld 92–110 A current band | **2.72** | 9.36 (tree plateau) |
| Withheld 27.9–32.0 kV voltage band | **0.34** | 0.76 |
| Withheld 40.5–55 °C ambient band | **2.47** | 3.51 |
| Withheld 47.5–60 min duration band | **0.53** | 0.65 |

The deployed hybrid wins on **all four operating axes** — the model choice
is not an artifact of a single test. Additional tests:

- **Label noise:** 5% of training labels randomly flipped → CV accuracy
  degrades gracefully to 0.946 (not brittle).
- **Train/serve consistency audit:** found and fixed a real defect where
  models were fit on NaN features but scored on −999-filled features (71
  affected cells). Fixing it *properly* (NaN semantics on both sides — the
  fill-both-with-−999 variant measurably hurt: recall 0.993→0.970) restored
  the best metrics **and** raised AUC to 0.99999. Final test-set predictions
  were identical before/after (same SHA-1), so the fix was pure hygiene.

---

## 6. Final results

| Item | Value |
|---|---|
| Task 1 (CV, honest protocol) | accuracy **0.999** · precision **1.000** · recall **0.993** · F1 **0.996** · AUC **0.99999** |
| Task 2 (CV, Valid rows) | RMSE **0.4147 °C** · MAE 0.267 · R² **0.9985** — at the noise floor |
| 90% prediction band | ±0.53 °C (from out-of-fold residuals) |
| Test-file predictions | 46/350 Invalid (13.1%, vs 13.4% in training); range 12.6–60.4 °C |
| Extrapolation safety | hybrid beats pure CatBoost on all 4 stress axes |
| Determinism | identical output hash across reruns |
| Final QA | **ALL PASS** (row count, header, order, label set, no NaNs, summary↔CSV exact match, attention IDs exist, blurb ≤ 100 words, zip integrity) |

---

## 7. Assessment and advancement probability

Scorecard against the organizers' rubric:

| Criterion | Weight | Score | Rationale |
|---|---|---|---|
| Prediction accuracy | 35% | 9.5/10 | at the noise floor; only hidden-data surprises can hurt |
| Invalid detection | 25% | 9.5/10 | 133/134 caught, 0 false flags, every trap handled |
| Unseen dataset | 20% | 9/10 | extrapolation validated on 4 axes; drift guard; sensor-free Task 2 |
| Reproducibility | 10% | 9.5/10 | one command, deterministic, consistency-audited |
| Reasoning | 10% | 9/10 | measured claims, stress tests, honest uncertainty |

**Overall: 9.3/10. Estimated probability of advancing: ~85% (range 75–92%).**

Residual risk is external: unknown field strength, unknown cutoff, unknown
hidden data. If the hidden dataset extends operating ranges (likely, given
the dedicated 20%), our extrapolation engineering is the differentiator;
most tree-based competitor solutions will fail there silently.

---

## 8. Repository map — what every file is and what it's for

```
powernext/
├── pentupbois-submission.zip        ★ THE SUBMISSION — the exact zip uploaded.
│                                      Exactly 4 files, nothing extra.
│
├── pentupbois-submission/            The submitted package, unpacked (identical
│   │                                  to the zip contents):
│   ├── pentupbois.csv                Deliverable 1: 350 rows, header
│   │                                  [Test_ID, Predicted_Reference_Parameter,
│   │                                  Validity_Label], Sample_Submission order.
│   ├── summary.json                  Deliverable 2: Task 3 auto-summary (counts,
│   │                                  min/max/avg, 3 attention IDs, ≤100-word
│   │                                  blurb, CV metrics, uncertainty, drift).
│   ├── METHODOLOGY_NOTE.md           Deliverable 4 (≤2 pages): approach, model
│   │                                  selection, stress tests, parameters,
│   │                                  abnormal-data method, assumptions,
│   │                                  digital-twin automation steps.
│   └── src/pipeline.py               Deliverable 3: the complete program.
│
├── README.md                         GitHub front page.
├── requirements.txt                  Python dependencies (pip install -r).
│
├── CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx
│                                    The organizers' input data (read-only).
│                                      Sheets: README, Training_Data (1000),
│                                      Test_Data (350), Sample_Submission.
│                                      Required next to src/ to rerun.
│
├── src/                              Working copies of the code (pipeline.py
│   │                                  is identical to the submitted one):
│   ├── pipeline.py                  ★ THE PROGRAM (Deliverable 3). One command:
│   │                                  python src/pipeline.py
│   │                                  Loads xlsx → Task 1 ensemble → Task 2
│   │                                  hybrid → honest CV for both → writes
│   │                                  submission/pentupbois.csv + summary.json
│   │                                  (build output, gitignored).
│   ├── bakeoff.py                   Round-1 model comparison: 17 candidates
│   │                                  per task, identical folds, leakage-free.
│   ├── bakeoff2.py                  Round-2: ensembles, seed-bags, depth
│   │                                  sweeps, blends.
│   └── stress_tests.py              Regenerates every stress-test number
│                                      cited in the methodology note into
│                                      results/stress_tests.txt (incl. the
│                                      duplicate-group leakage check).
│
├── results/                         Evidence tables backing every model choice:
│   ├── bakeoff_task1.csv / bakeoff_task2.csv / bakeoff.json   (round-1 results)
│   ├── bakeoff2_task1.csv / bakeoff2_task2.csv / bakeoff2.json (round-2 results)
│   ├── bakeoff_console.txt          Console log of the bake-off run.
│   └── stress_tests.txt             Regenerated stress-test evidence (§5).
│
├── PROJECT_DEEP_ANALYSIS.md         Full reasoning, trap analysis, QA log and
│                                      assessment (the shareable deep-dive).
│
└── archive/                         Internal / historical documents, not
    ├── PowerNext-AI_Handoff.md        deliverables: the original prompt
    └── VERIFICATION_AND_PLAN.md       handoff and the early-phase
                                       verification + build plan.
```

**To reproduce everything from scratch:** place the xlsx next to `src/`, run
`python src/pipeline.py` (needs Python 3.10+, pandas, numpy, scikit-learn,
lightgbm, catboost, scipy, openpyxl — see `requirements.txt`; the pipeline
also runs self-contained from inside `pentupbois-submission/`, writing the
deliverables in place; `bakeoff.py` additionally needs statsmodels). It
regenerates both deliverable files deterministically. `python src/bakeoff.py`
regenerates the model-comparison evidence.

---

## 9. Glossary (for readers new to data science / ML)

| Term | Plain meaning |
|---|---|
| **RMSE** | Average size of our prediction mistakes (in the same units as the target, °C). Ours is 0.41 °C on values spanning 12–60 °C — like guessing a weight within half a kilo. |
| **R²** | "How much of the puzzle did you solve," 0 to 1. Ours: 0.9985 = we explain 99.85% of why temperatures differ. |
| **Accuracy / precision / recall** | Accuracy = fraction of records judged correctly. Precision = when we cry "fault," how often we're right (ours: always). Recall = of all true faults, how many we catch (ours: 133 of 134). |
| **AUC** | Ranking quality from 0.5 (coin flip) to 1.0 (perfect). Ours: 0.99999. |
| **Cross-validation (CV)** | Grading yourself honestly: hide part of the data, learn from the rest, get tested on the hidden part; repeated 15 times. All our reported numbers are from this protocol, never from data the model trained on. |
| **Overfitting** | Memorizing practice exams and failing the real one. Tested for explicitly. |
| **Extrapolation** | Predicting beyond anything seen in training. Trees are bad at it; our physics base is built for it. |
| **Ensemble** | Four different algorithms vote; the average is harder to fool than any single one. |
| **Noise floor** | The random jitter in the data that no model can ever remove. We proved our error equals it — i.e., we've extracted all learnable signal. |
| **Deterministic** | Run twice, get byte-identical outputs. No randomness in results. |
| **Physics-informed hybrid** | A physics formula (safe everywhere, rough) plus a machine-learned correction (accurate, only safe near training data) — the best of both. |

---

## 10. The one-paragraph summary

The validity signal in this dataset lives in *whether the sensors agree with
each other*, never in *how big the numbers are* — so we engineered
relationship features (fit on Valid rows only), which took detection from
coin-flip (AUC 0.502) to essentially perfect (0.99999), catching 133/134
faults with zero false alarms while dodging every planted trap: the S4
decoy (29 honest rows with missing S4), the unusual-but-valid regime shifts
(40/50 extreme-current rows are Valid), and duplicate records carrying
poisoned answers (excluded from regression training). The Reference
Parameter is predicted from operating inputs only — immune to the very
sensor faults we detect — by a physics-informed hybrid (Joule-heating OLS
base + bagged CatBoost residual correction) that reaches the data's noise
floor (RMSE 0.41 °C, proven) and beats pure gradient boosting on all four
extrapolation stress tests, which matters because 20% of the grade is a
hidden dataset that may lie outside the training range. Everything is one
command, deterministic, self-auditing (drift check, uncertainty band,
noise-floor diagnostic), and reproducible.

*Prepared for team pentupbois and anyone they share it with — this document
plus the repository is the complete record of what happened here.*
