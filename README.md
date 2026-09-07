# PowerNext-AI — Black-Box Test Bench Challenge

**Team pentupbois** · CPRI State-Level Hackathon, Screening Round

An end-to-end, fully deterministic ML pipeline that:

1. **Fault detection** — classifies each of 350 black-box electrical test-bench
   records as `Valid` or `Invalid`
2. **Reference_Parameter regression** — predicts the certified temperature
   rise (°C) for every record, valid or not
3. **Auto-summary** — emits a machine-readable `summary.json` with counts,
   extremes, attention Test IDs, uncertainty band and an input-drift check

...all from one command, seed 42, bit-for-bit reproducible.

## Results (honest out-of-fold cross-validation)

| Task | Metric | Score |
|---|---|---|
| Fault detection | Accuracy / F1 / AUC | **0.999 / 0.996 / 1.000** |
| Reference_Parameter | RMSE / MAE / R² | **0.41 °C / 0.27 / 0.9985** |

- Catches **133 of 134** known-invalid training records with **zero** false flags.
- ~3× more accurate than pure gradient boosting when predicting beyond the
  training current range (RMSE 2.7 vs 9.4 on a withheld 92–110 A band) — tree
  models plateau at the training boundary; our physics base extrapolates.

## The one idea everything hangs on

> **A record is invalid because its internal story contradicts itself — not
> because any single value looks unusual.**

A classifier on raw sensor values scores AUC 0.502 (a coin flip); on
*relationship* features (inter-sensor residuals, duplicate structure,
missingness pattern) it exceeds 0.999. This is the organizers' doctrine —
"an unusual value is not necessarily an invalid value" — made quantitative.

- **Task 1** — probability-mean ensemble of RandomForest + SVM + LightGBM +
  CatBoost on relationship features, with four physically-motivated rules
  (duplicate logging, impossible readings, thermometer dropout, inter-sensor
  disagreement) as an explainability layer.
- **Task 2** — physics-informed hybrid: `Ref ~ I + I² + Ambient + V + t`
  (Joule heating, OLS) carries the extrapolatable physics; a 5-seed bagged
  CatBoost learns only the bounded residual correction. Deliberately
  **sensor-free** — sensors are exactly what breaks in invalid records, so
  predictions are corruption-immune.
- **Task 3** — programmatic summary; the 3 attention IDs are the highest-risk
  valid tests the model is least confident about.

### Traps in the data, and how they were handled

| Trap | Handling |
|---|---|
| Sensor_S4 looks like a sensor but is pure noise | excluded from modeling (kept in raw data) |
| NaN in S4 alone (29/29 Valid) vs NaN in S1–S3 (15/15 Invalid) | missingness *location* is the signal, not missingness itself |
| Duplicate feature-vectors with conflicting certified answers | flagged as duplicate logging (Invalid) |
| Genuine high-current regime shifts look "abnormal" but are Valid | all sensors move together → residuals stay small → never flagged |
| Invalid rows carry poisoned Reference values (~12 °C spread in duplicate groups) | excluded from regression training |

## Repository layout

```
pentupbois-submission.zip     ★ the exact 4-file package uploaded to the judges
pentupbois-submission/          the same package, unpacked (canonical deliverables)
├── pentupbois.csv              deliverable 1: predictions in required format
├── summary.json                deliverable 2: Task 3 auto-summary
├── METHODOLOGY_NOTE.md         deliverable 4: ≤2-page methodology
└── src/pipeline.py             deliverable 3: the complete program
src/                            working code (pipeline.py + bake-off + stress tests)
results/                        evidence tables backing every model choice
CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx   organizers' input data
PROJECT_DEEP_ANALYSIS.md        deep-dive: every decision, trap, and QA check
archive/                        internal/historical documents
```

## Reproduce

With the xlsx next to `src/` (as in this repo):

```bash
python src/pipeline.py       # -> submission/pentupbois.csv + submission/summary.json
python src/stress_tests.py   # -> results/stress_tests.txt
python src/bakeoff.py        # -> model bake-off tables in results/
```

Requires Python 3.11+ with `pandas numpy statsmodels scikit-learn lightgbm
catboost scipy openpyxl`. No manual record edits; all thresholds derive from
the data at run time; outputs verified identical on re-run.

Note: `submission/` is a build output (gitignored) — the canonical submitted
deliverables live in [`pentupbois-submission/`](pentupbois-submission/).

## Documentation

- [Methodology note — the submitted 2-pager](pentupbois-submission/METHODOLOGY_NOTE.md)
- [Deep analysis — every decision, trap and QA check](PROJECT_DEEP_ANALYSIS.md)
