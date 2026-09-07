# PowerNext-AI submission package — team pentupbois

Contents:
- `submission/pentupbois.csv` — predictions in Sample_Submission row order
  (columns: Test_ID, Predicted_Reference_Parameter, Validity_Label)
- `submission/summary.json` — auto-generated Task 3 summary (counts,
  min/max/avg, attention IDs, methodology blurb, CV metrics, uncertainty
  band, input-drift check)
- `METHODOLOGY_NOTE.md` — 2-page methodology + robustness stress tests +
  digital-twin automation steps
- `VERIFICATION_AND_PLAN.md` — early-phase claim-by-claim data verification
  (status note inside)
- `src/` — full source: `pipeline.py` (one-command run), `bakeoff.py` /
  `bakeoff2.py` (model comparison), `stress_tests.py` (robustness evidence)
- `results/` — bake-off tables and regenerated stress-test evidence

Reproduce from the repository root (xlsx placed next to `src/`):

```
python src/pipeline.py      # regenerates submission/pentupbois.csv + summary.json
python src/bakeoff.py       # regenerates the model comparison tables
python src/stress_tests.py  # regenerates results/stress_tests.txt
```

Requires: python 3.11+, pandas, numpy, statsmodels, scikit-learn, lightgbm,
catboost, scipy, openpyxl. Deterministic (seed 42). No manual per-record edits.
