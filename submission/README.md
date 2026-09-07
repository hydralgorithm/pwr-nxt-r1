# PowerNext-AI submission package — team pentupbois

Contents:
- `pentupbois.csv` — predictions in Sample_Submission row order
  (columns: Test_ID, Predicted_Reference_Parameter, Validity_Label)
- `summary.json` — auto-generated Task 3 summary (counts, min/max/avg,
  attention IDs, methodology blurb, CV metrics)
- `METHODOLOGY_NOTE.md` — 2-page methodology + digital-twin automation
- `src/` — full source (`pipeline.py` one-command run; `bakeoff.py` model
  comparison; results in `results/`)

Reproduce from the repository root:

```
python src/pipeline.py     # regenerates submission/pentupbois.csv + summary.json
python src/bakeoff.py      # regenerates the model comparison tables
```

Requires: python 3.11+, pandas, numpy, statsmodels, scikit-learn, lightgbm,
catboost, openpyxl. Deterministic (seed 42). No manual per-record edits.
