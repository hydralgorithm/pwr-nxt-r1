# PowerNext-AI — Black-Box Test Bench Challenge
## Handoff Document & Execution Prompt for the Receiving AI Agent

**Status of this document:** This is a planning/handoff artifact, not a finished submission. Section 2 ("Prior Insights") reports findings from an *earlier session* against `training_data.csv` / `test_data.csv`. **Those files are not present in this session's upload folder as of writing this document.** Treat every number in Section 2 as an unverified claim until you re-derive it yourself from the actual files. This is not optional — it is the first instruction in Section 4.

---

## 1. Full Problem Statement (verbatim intent, lightly reorganized)

**Organizer:** CPRI, with institutional partner MIT Bengaluru
**Round:** Screening Round — "The Black-Box Test Bench Challenge"

### 1.1 Physical setup
A CPRI lab runs repeated electrical tests on a test specimen. For each test:
- A specified **Applied_Voltage** (kV) is applied.
- A specified **Load_Current** (A) is passed through the specimen.
- This is held for a **Test_Duration** (min).
- **Ambient_Temperature** (°C) is recorded.
- Three physical thermometers record temperature rise at different locations: **Sensor_S1, Sensor_S2, Sensor_S3** (°C above ambient).
- One auxiliary sensor of unknown relevance, **Sensor_S4** (arbitrary units), is also logged.
- A calibrated, independently-verified **Reference_Parameter** — the true hot-spot temperature rise — is available *only* for historical tests.
- Each historical test also carries a **Validity_Label** (Valid/Invalid) assigned by engineers.

### 1.2 The core difficulty (stated explicitly by the organizers)
> "An unusual value is not necessarily an invalid value... the equipment may genuinely enter a different operating regime. Such behaviour should not automatically be treated as a sensor fault or invalid test."

So there are two very different reasons a record can look "weird":
1. **A real fault** — sensor malfunction, missing/corrupted data, physically inconsistent readings, logging error.
2. **A real physical regime shift** — under extreme voltage/current/duration combinations, the specimen genuinely heats differently. This is scientifically the *most* interesting data, and must not be discarded.

The whole challenge is graded on how well a solution tells these two apart — not how it handles obviously clean data.

### 1.3 Data provided
- `training_data.csv` — ~800–1,000 records with all inputs, all sensors, **Reference_Parameter**, and **Validity_Label**.
- `test_data.csv` — ~200–300 records with the same inputs/sensors but **no** Reference_Parameter or Validity_Label.
- Known contamination: missing values, duplicate records, measurement noise, sensor spikes, incorrect values, possibly irrelevant parameter(s), genuine regime shifts.

### 1.4 Required tasks
| Task | What's required |
|---|---|
| **Task 1** | Identify records that are likely sensor error / corrupted / invalid — while distinguishing this from genuine regime change. |
| **Task 2** | Learn the (undisclosed) relationship between inputs/sensors and Reference_Parameter from verified historical data; predict Reference_Parameter for every row of `test_data.csv`. |
| **Task 3** | A program-generated summary: record count, invalid count, min/max/avg predicted Reference_Parameter, 3 Test IDs needing highest attention, and a ≤100-word methodology blurb. |

### 1.5 Deliverables
1. `<TeamName>.csv` with **exactly** columns: `Test_ID, Predicted_Reference_Parameter, Valid/Invalid`
2. `summary.json` or `summary.csv` (auto-generated, not hand-typed)
3. Full source code / notebook — must run start-to-finish with **zero manual per-record edits**
4. A ≤2-page methodology note covering: approach, important parameters, abnormal-data detection method, assumptions, and **digital-twin automation steps** (how this becomes a live monitoring pipeline, not a one-off CSV run)
5. Everything zipped as `your-teamname-submission.zip`

### 1.6 Evaluation weights
| Criterion | Weight |
|---|---|
| Reference_Parameter prediction accuracy | 35% |
| Invalid-record detection accuracy | 25% |
| **Performance on a second, unseen dataset** | 20% |
| Reproducibility / code quality | 10% |
| Engineering reasoning / explanation | 10% |

**The single most consequential line in the brief:** 20% of the grade comes from a dataset you will *never see*. This rules out anything hand-tuned to the quirks of the specific `test_data.csv` you were given (no per-Test_ID hacks, no eyeballed thresholds picked to make this file's histogram look right). The pipeline must be a genuinely general function learned from `training_data.csv`.

---

## 2. Prior Insights (UNVERIFIED — re-derive before trusting)

These were reported from an earlier analysis pass. They are directionally plausible and worth using as *hypotheses to test*, not facts to build on blindly.

1. **Duplicate-condition conflicts (claimed):** ~24 training rows share identical operating conditions (voltage, current, ambient, duration) with another row but disagree on sensor readings / Reference_Parameter — and every one of those 24 was labeled Invalid. If true, this is a clean, physically-justified rule (a deterministic system can't give two answers to the same inputs), not a hand-tuned threshold.
2. **Sensor mutual agreement as the invalid-detector (claimed):** Among Valid rows, S1/S2/S3 pairwise correlations were ~0.89–0.99; among Invalid rows they collapsed to ~0.36–0.42. A model predicting S3 from S1 (fit on Valid data) had residuals ~0.55 for Valid rows vs ~5.24 for Invalid rows.
3. **Feature relevance ranking (claimed):** Load_Current (r≈0.88) > Sensor_S2 (0.80) > Sensor_S1 (0.59) > Sensor_S3 (0.50) > Applied_Voltage (0.27) > Ambient_Temp (0.24) > Test_Duration (≈0) > Sensor_S4 (≈0) against Reference_Parameter.
4. **Regime shift (claimed):** A plain linear model (Voltage, Current, Ambient, S1, S2, S3 → Reference_Parameter) on Valid data gave R²≈0.87, with a U-shaped residual pattern against Load_Current — under-predicting at both low and high current, over-predicting mid-range. Interpreted as a genuine non-linearity, not noise.
5. **Housekeeping (claimed):** Missing values concentrated in sensors (S4 worst: 29 train / 11 test). Label balance ~866 Valid / 134 Invalid (~13.4% invalid) — moderately imbalanced.

---

## 3. Prompt for the Receiving Agent

Paste this as your instruction to the agent that will actually build the pipeline:

> You are picking up the PowerNext-AI Black-Box Test Bench Challenge. Attached/linked is a handoff document (`PowerNext-AI_Handoff.md`) containing the full problem statement (Section 1) and a set of *unverified* prior findings (Section 2).
>
> Your first job is **not** to build the final pipeline. Your first job is to **independently verify or refute every claim in Section 2** using the actual `training_data.csv` and `test_data.csv` — reading them with the appropriate data-analysis skill/tooling available to you (e.g. pandas-based inspection; use any spreadsheet/data-analysis skill your environment provides rather than assuming file structure). Report back:
> - Which claims held up (with your own numbers), which didn't, and anything new you found that Section 2 missed.
> - Only after verification, propose the Task 1 / Task 2 / Task 3 implementation plan, following the structure in Section 4 below, in both a technical and a plain-language form.
> - Where there is a genuine, defensible fork between two reasonable approaches (see Section 5), stop and lay out both sides — do not silently pick one — and ask the user which they prefer, unless the choice is a clear correctness issue (e.g. don't train Task 2 on Invalid rows) rather than a style preference.

---

## 4. Proposed Plan — Technical and Beginner Explanations Side by Side

### Step 0 — Verify before trusting
- **Technical:** Load both CSVs, check row/column counts against the ~800–1,000 / ~200–300 expectation, dtype-check every column, and re-run the duplicate-condition, sensor-correlation, and feature-importance checks from Section 2 from scratch.
- **Beginner:** Before building anything, double-check that what a previous pass claimed about the data is actually true — treat it like fact-checking a rumor before acting on it.

### Step 1 — Clean the obvious junk
- **Technical:** Drop exact duplicate rows; separately flag (don't necessarily drop) rows with identical input conditions but divergent outputs — these are candidate structural Invalids per Insight 1, not row-deletion candidates yet. Quantify missingness per column; decide imputation vs. row-exclusion per column based on how central that column is (e.g., missing Load_Current is more serious than missing S4).
- **Beginner:** Get rid of exact copies, and separately note down (but don't throw away yet) any tests where the exact same setup produced two different results — that's a red flag, not proof.

### Step 2 — Build the Task 1 classifier (Valid/Invalid)
- **Technical:** Engineer *relationship* features rather than raw values — S1–S2, S1–S3, S2–S3 residuals/correlation-consistency, residual of each sensor against a model fit only on Valid rows, and the duplicate-condition-conflict flag. Train a supervised classifier (logistic regression as a baseline, gradient-boosted trees or random forest as the main model) on `Validity_Label`, with stratified train/validation split given the ~13% imbalance. Evaluate with precision/recall/F1 on the Invalid class specifically (not just accuracy, which is misleading under imbalance).
- **Beginner:** Rather than eyeballing which numbers "look weird," teach a model what past engineers actually called Invalid, using clues like "do the three thermometers agree with each other" rather than "is any single thermometer reading unusually high" — because a genuinely hot test will have all three sensors agreeing.

### Step 3 — Build the Task 2 regression (Reference_Parameter)
- **Technical:** Train only on rows the Task 1 model calls (or the ground truth calls, for training) Valid — feeding Invalid rows into this regression teaches it sensor noise as if it were physics. Test a non-linear model (gradient boosting, or a linear model with a Load_Current² / spline term) given the claimed U-shaped residual pattern. Run an ablation with/without Sensor_S4 and Test_Duration and report the delta in validation error, rather than dropping them on the strength of a single correlation number.
- **Beginner:** Only teach the "predict the true temperature" model using tests we trust are clean — otherwise it learns from broken thermometers as if they were telling the truth. Then check: does the possibly-useless sensor (S4) or the test duration actually improve predictions, or can we drop them?

### Step 4 — Apply both models to `test_data.csv`
- **Technical:** Run the Task 1 classifier to label every test row; run the Task 2 regressor on every row regardless of predicted validity (the deliverable format requires a prediction for every row either way). Store both outputs.
- **Beginner:** Use both trained models on the new, unlabeled tests — every test gets a predicted temperature and a valid/invalid flag.

### Step 5 — Task 3 auto-summary
- **Technical:** Compute counts and min/max/avg programmatically from the output CSV — never hand-type these. For "3 Test IDs requiring highest attention," pick a stated, defensible rule (e.g., highest predicted Reference_Parameter among rows flagged Valid, or lowest classifier-confidence rows) and state the rule in the ≤100-word blurb.
- **Beginner:** Have the program itself write its own report card at the end — how many tests it looked at, how many it flagged as suspicious, and which 3 tests most deserve a human's attention.

### Step 6 — Package deliverables
- **Technical:** Assemble `<TeamName>.csv`, `summary.json`, notebook/scripts, and the 2-page methodology note (including the digital-twin automation section) into one folder, zip as `your-teamname-submission.zip`.
- **Beginner:** Put everything the organizers asked for into one folder and zip it up with the exact name they specified.

---

## 5. Open Forks — Where to Stop and Ask the User

These are genuine judgment calls, not correctness issues, so the agent should lay out both sides and ask rather than silently choosing:

1. **Threshold/rule-based vs. fully learned Task 1 classifier.** A rule-based approach (duplicate-conflict flag + sensor-disagreement threshold) is more explainable and easier to justify in the methodology note, but may generalize worse to the hidden second dataset than a trained classifier. *Trade-off:* explainability vs. the 20%-weighted generalization criterion.
2. **How aggressively to use Sensor_S4.** Drop entirely (simpler, matches the claimed near-zero correlation) vs. keep with low weight in a tree-based model (which can exploit weak non-linear signal a correlation coefficient would miss). *Trade-off:* simplicity/defensibility vs. squeezing out marginal accuracy.
3. **Model family for Task 2.** A linear model with an explicit engineered non-linear term (e.g., Load_Current²) is more interpretable and easier to describe physically in the note; a gradient-boosted tree model likely fits the U-shape better with less manual feature engineering but is harder to explain and slightly more overfitting-prone on ~800 rows. *Trade-off:* interpretability vs. raw fit quality.
4. **How to define "highest attention" Test IDs for Task 3.** Highest predicted temperature (physically riskiest) vs. lowest model confidence (most uncertain) vs. records that were borderline Valid/Invalid. These serve different purposes and the organizers didn't specify which.

---

## 6. Deliverables Checklist (for final QA before zipping)

- [ ] `<TeamName>.csv` — columns exactly `Test_ID, Predicted_Reference_Parameter, Valid/Invalid`
- [ ] `summary.json` or `summary.csv` — generated by running the code, not hand-edited
- [ ] Full pipeline runs end-to-end with zero manual per-record fixes
- [ ] ≤2-page methodology note: approach, important parameters, abnormal-detection method, assumptions, digital-twin automation steps
- [ ] All findings in this document's Section 2 re-verified (or corrected) against the actual data
- [ ] Every threshold/parameter choice has a stated justification, not just "it looked right on this file"
- [ ] Zipped as `your-teamname-submission.zip`
