# Model Card: GlucoTwin Multi-Horizon Forecaster

## Model Details
- **Architecture**: Multi-Horizon Deep Sequence and Gradient Boosted Ensemble (Persistence, Linear Trend, LightGBM, XGBoost, PyTorch Multimodal GRU, Temporal Fusion Transformer)
- **Framework**: PyTorch 2.6.0+cu124, LightGBM 4.6.0, XGBoost 3.1.1, Scikit-Learn 1.6.1
- **Lookback Window**: 6 hours (72 steps at 5-minute sampling interval)
- **Forecast Horizons**: +30 min (6 steps), +60 min (12 steps), +120 min (24 steps)
- **Target Variables**: Continuous future glucose (mg/dL) with multi-quantiles (q10, q50 median, q90) and adverse glycemic events (Hyperglycemia >= 180 mg/dL, Severe Hyperglycemia >= 250 mg/dL, Hypoglycemia <= 70 mg/dL, Postprandial Spike excursion)
- **Uncertainty Method**: Split Conformal Prediction with finite-sample coverage correction and monotonic quantile sorting
- **Personalization**: LoRA residual adapters (rank=4, alpha=8.0) fine-tuned on individual patient initial 3-day calibration windows

## Intended Use
- **Primary Purpose**: Clinical decision support system for endocrinologists and physicians managing Indian adults with Type 2 Diabetes and prediabetes.
- **Input Modalities**:
  1. Dynamic CGM sequence (glucose, velocity, acceleration, rolling volatility, time in range)
  2. Dynamic wearable activity (heart rate, step counts, active minutes)
  3. Known future context (circadian time-of-day sine/cosine encoding, scheduled meal carbs, planned activity)
  4. Static patient clinical context (age, sex, BMI, HbA1c, fasting glucose, diabetes status)
- **Out of Scope**: Autonomous closed-loop insulin delivery without medical oversight, acute critical care resuscitation, pediatric populations.

## 5-Fold Leave-Patients-Out Cross-Validation Results

The evaluation protocol enforces zero patient overlap across 5 stratified folds covering all 45 CGMacros participants (14 T2D, 16 prediabetes, 15 healthy; 134,292 feature windows). All scalers and feature transformers are fitted strictly on training patients per fold.

### 1. Continuous Trajectory Prediction Performance (All 6 Models)

| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | MARD (%) | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **+30 min** | **LightGBM** | **17.62 +/- 0.89** | **11.52 +/- 0.54** | **8.40 +/- 0.53** | **91.10 +/- 0.68** | 99.39 +/- 0.32 |
| | **XGBoost** | 18.17 +/- 0.96 | 11.80 +/- 0.57 | 8.52 +/- 0.47 | 90.52 +/- 0.31 | 99.11 +/- 0.78 |
| | **Multimodal GRU** | 18.86 +/- 1.01 | 12.69 +/- 0.52 | 9.21 +/- 0.38 | 89.63 +/- 0.87 | 99.34 +/- 0.25 |
| | **Persistence** | 20.06 +/- 0.89 | 12.90 +/- 0.44 | 9.13 +/- 0.38 | 88.00 +/- 1.26 | **99.58 +/- 0.12** |
| | **TFT** | 22.52 +/- 1.87 | 14.56 +/- 0.73 | 10.24 +/- 0.43 | 86.72 +/- 0.75 | 99.07 +/- 0.42 |
| | **Linear Trend** | 27.93 +/- 0.70 | 18.42 +/- 0.27 | 13.35 +/- 0.63 | 78.58 +/- 1.25 | 99.05 +/- 0.13 |
| **+60 min** | **Multimodal GRU** | **26.43 +/- 1.94** | 18.19 +/- 1.07 | 13.16 +/- 0.53 | 78.78 +/- 1.13 | **98.62 +/- 0.48** |
| | **LightGBM** | 26.86 +/- 2.77 | **17.61 +/- 1.60** | **12.45 +/- 0.86** | **80.81 +/- 1.46** | 98.07 +/- 1.02 |
| | **XGBoost** | 27.62 +/- 2.78 | 18.05 +/- 1.58 | 12.64 +/- 0.71 | 80.31 +/- 1.26 | 97.92 +/- 0.94 |
| | **TFT** | 28.32 +/- 2.17 | 18.82 +/- 0.98 | 13.15 +/- 0.36 | 78.87 +/- 1.16 | 98.46 +/- 0.54 |
| | **Persistence** | 29.64 +/- 1.86 | 19.15 +/- 1.06 | 13.39 +/- 0.70 | 77.51 +/- 1.64 | 98.61 +/- 0.38 |
| | **Linear Trend** | 50.54 +/- 1.34 | 34.11 +/- 0.72 | 24.53 +/- 0.87 | 57.09 +/- 0.92 | 94.80 +/- 0.41 |
| **+120 min** | **Multimodal GRU** | **32.86 +/- 2.63** | **23.33 +/- 1.58** | 17.05 +/- 0.68 | 68.68 +/- 2.15 | **97.96 +/- 0.60** |
| | **TFT** | 34.64 +/- 3.07 | 23.44 +/- 1.62 | **16.30 +/- 0.67** | **70.08 +/- 1.48** | 97.29 +/- 0.78 |
| | **LightGBM** | 35.20 +/- 5.30 | 23.89 +/- 3.08 | 16.82 +/- 1.57 | 68.87 +/- 3.34 | 96.35 +/- 2.30 |
| | **XGBoost** | 36.20 +/- 5.16 | 24.40 +/- 2.97 | 17.02 +/- 1.36 | 68.44 +/- 3.05 | 96.15 +/- 2.17 |
| | **Persistence** | 40.14 +/- 3.15 | 26.56 +/- 1.91 | 18.82 +/- 1.46 | 66.51 +/- 2.54 | 96.76 +/- 0.81 |
| | **Linear Trend** | 86.20 +/- 2.76 | 59.73 +/- 1.94 | 43.91 +/- 1.82 | 36.51 +/- 0.78 | 85.74 +/- 1.17 |

### 2. Conformal Uncertainty Calibration (Target 80% Coverage)

| Model | Horizon | Raw Coverage (%) | Calibrated Coverage (%) | Raw Width (mg/dL) | Calibrated Width (mg/dL) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multimodal GRU** | +30 min | 78.98 +/- 1.70 | **80.18 +/- 2.39** | 37.53 +/- 1.21 | 38.74 +/- 2.85 |
| | +60 min | 72.28 +/- 1.96 | **79.49 +/- 2.76** | 45.77 +/- 1.60 | 53.98 +/- 4.23 |
| | +120 min | 68.02 +/- 2.65 | **79.44 +/- 2.23** | 54.09 +/- 3.11 | 69.11 +/- 4.55 |
| **LightGBM** | +30 min | 77.74 +/- 1.46 | **80.42 +/- 1.36** | 36.23 +/- 4.58 | 38.27 +/- 4.08 |
| | +60 min | 76.31 +/- 1.75 | **80.27 +/- 2.36** | 54.36 +/- 6.12 | 58.80 +/- 5.18 |
| | +120 min | 73.61 +/- 3.45 | **79.32 +/- 3.22** | 64.93 +/- 7.81 | 72.79 +/- 6.36 |
| **XGBoost** | +30 min | 78.30 +/- 1.50 | **80.90 +/- 1.48** | 37.12 +/- 4.29 | 39.17 +/- 3.79 |
| | +60 min | 77.00 +/- 2.54 | **80.73 +/- 2.68** | 55.12 +/- 6.20 | 59.43 +/- 4.99 |
| | +120 min | 74.10 +/- 4.00 | **79.23 +/- 3.29** | 64.79 +/- 6.68 | 72.26 +/- 5.41 |

### 3. Adverse Event Early Warning Performance (With Real Lead Times)

Alerts defined in params.yaml (sustained_threshold rule, operating threshold 0.50, preceding window 120 min, early cutoff 30 min):

| Event Type | Horizon | Model | AUROC | AUPRC | Lead Time Median (min) | Lead Time IQR (min) | Share >=30m Early (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hyperglycemia** (>180 mg/dL) | +30 min | LightGBM | **0.96 +/- 0.01** | **0.85 +/- 0.07** | 5.0 | 70.0 | 33.87% |
| | +60 min | LightGBM | **0.90 +/- 0.02** | **0.66 +/- 0.10** | 13.5 | 63.0 | 32.44% |
| | +60 min | XGBoost | 0.90 +/- 0.03 | 0.66 +/- 0.13 | 23.0 | 96.0 | **44.97%** |
| | +120 min | Multimodal GRU | **0.83 +/- 0.04** | **0.47 +/- 0.16** | 0.0 | 43.75 | 25.34% |
| **Severe Hyperglycemia** (>250 mg/dL) | +30 min | LightGBM | **0.99 +/- 0.01** | **0.78 +/- 0.06** | 0.0 | 20.0 | 17.09% |
| | +60 min | Multimodal GRU | **0.95 +/- 0.02** | **0.42 +/- 0.18** | 0.0 | 0.0 | 7.74% |
| | +120 min | Multimodal GRU | **0.89 +/- 0.03** | **0.24 +/- 0.13** | 0.0 | 0.0 | 3.58% |
| **Hypoglycemia** (<=70 mg/dL) | +30 min | XGBoost | **0.98 +/- 0.01** | **0.53 +/- 0.25** | 0.0 | 8.0 | 15.25% |
| | +30 min | LightGBM | 0.91 +/- 0.09 | 0.48 +/- 0.27 | 11.5 | 44.5 | 26.89% |
| | +60 min | XGBoost | **0.95 +/- 0.03** | **0.34 +/- 0.27** | 0.0 | 0.0 | 5.14% |
| | +60 min | LightGBM | 0.84 +/- 0.12 | 0.35 +/- 0.28 | 8.0 | 24.0 | 18.25% |
| | +120 min | TFT | **0.89 +/- 0.09** | **0.24 +/- 0.23** | 0.0 | 0.0 | 7.99% |
| **Postprandial Spike** | +60 min | Multimodal GRU | **0.84 +/- 0.01** | **0.40 +/- 0.02** | 7.0 | 67.0 | **36.00%** |
| | +60 min | TFT | 0.77 +/- 0.02 | 0.31 +/- 0.02 | 4.0 | 56.0 | 34.61% |
| | +60 min | XGBoost | 0.76 +/- 0.02 | 0.30 +/- 0.02 | 0.0 | 51.5 | 31.87% |


### 3.1 Validation-Tuned Operating Points (Target Sensitivity >= 0.80)

Thresholds tuned on validation folds to achieve 80% sensitivity, evaluated on held out test folds. Alert episodes start when probability crosses threshold after being below for >= 30 min. Lead time is measured from alert episode start within 120 min before glucose crossing to the actual crossing.

| Event Type | Horizon | Alert Threshold | Target Sensitivity | Test Recall | Test Precision | False Alerts / Pt-Day | Lead Time Median (min) | Lead Time IQR (min) | Share >=30m Early (%) | Events per Fold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hyperglycemia (>180 mg/dL)** | +30 min | 0.33 | 0.80 | 0.65 +/- 0.10 | 0.62 +/- 0.04 | 0.78 +/- 0.21 | 18.0 | 30.0 | 36.20 +/- 3.28% | [143, 126, 126, 126, 148] |
| **Hyperglycemia (>180 mg/dL)** | +60 min | 0.52 | 0.80 | 0.18 +/- 0.11 | 0.15 +/- 0.04 | 2.39 +/- 0.80 | 68.0 | 46.2 | 83.84 +/- 13.65% | [143, 126, 126, 126, 148] |
| **Hyperglycemia (>180 mg/dL)** | +120 min | 0.38 | 0.80 | 0.19 +/- 0.04 | 0.29 +/- 0.10 | 1.11 +/- 0.51 | 75.5 | 63.8 | 77.07 +/- 8.15% | [143, 125, 126, 126, 148] |
| **Severe Hyperglycemia (>250 mg/dL)** | +30 min | 0.06 | 0.80 | 0.67 +/- 0.12 | 0.36 +/- 0.14 | 0.58 +/- 0.36 | 37.5 | 34.0 | 64.75 +/- 14.42% | [26, 57, 15, 39, 49] |
| **Severe Hyperglycemia (>250 mg/dL)** | +60 min | 0.08 | 0.80 | 0.46 +/- 0.10 | 0.37 +/- 0.14 | 0.49 +/- 0.37 | 51.0 | 36.2 | 78.65 +/- 11.28% | [26, 58, 16, 39, 49] |
| **Severe Hyperglycemia (>250 mg/dL)** | +120 min | 0.16 | 0.80 | 0.14 +/- 0.04 | 0.36 +/- 0.22 | 0.40 +/- 0.39 | 56.0 | 19.0 | 80.48 +/- 19.54% | [26, 58, 16, 39, 48] |
| **Hypoglycemia (<=70 mg/dL)** | +30 min | 0.52 | 0.80 | 0.21 +/- 0.15 | 0.02 +/- 0.01 | 3.40 +/- 0.99 | 62.0 | 21.5 | 66.00 +/- 37.74% | [5, 13, 28, 25, 15] |
| **Hypoglycemia (<=70 mg/dL)** | +60 min | 0.03 | 0.80 | 0.41 +/- 0.11 | 0.12 +/- 0.05 | 0.81 +/- 0.24 | 45.5 | 46.0 | 54.59 +/- 4.75% | [5, 13, 28, 25, 15] |
| **Hypoglycemia (<=70 mg/dL)** | +120 min | 0.57 | 0.80 | 0.25 +/- 0.10 | 0.03 +/- 0.02 | 2.41 +/- 0.66 | 68.0 | 27.0 | 82.67 +/- 18.31% | [4, 14, 28, 25, 15] |
| **Postprandial Spike** | +30 min | 0.42 | 0.80 | 0.59 +/- 0.03 | 0.50 +/- 0.06 | 1.99 +/- 0.51 | 38.0 | 34.0 | 72.65 +/- 7.15% | [239, 240, 248, 257, 262] |
| **Postprandial Spike** | +60 min | 0.50 | 0.80 | 0.61 +/- 0.05 | 0.53 +/- 0.05 | 1.71 +/- 0.25 | 61.0 | 28.0 | 90.07 +/- 2.05% | [239, 240, 248, 257, 262] |
| **Postprandial Spike** | +120 min | 0.66 | 0.80 | 0.34 +/- 0.03 | 0.41 +/- 0.05 | 1.70 +/- 0.45 | 101.0 | 27.0 | 95.84 +/- 2.55% | [239, 240, 248, 257, 262] |

### 4. Personalization: LoRA vs Simple Bias Correction (All 45 Patients)

Forecast horizon: +60 minutes. Evaluated across all 45 CGMacros participants (days 1 to 3 calibration vs days 4+ evaluation):

- **Cohort Size**: 45 patients
- **Simple Bias Correction Baseline**:
  - Improved Patients: 35 / 45 (77.8%)
  - Mean RMSE: 24.14 mg/dL -> 23.23 mg/dL (Gain: +0.91 mg/dL)
  - Mean MARD: 12.82% -> 12.46% (Gain: +0.36%)
- **LoRA Adapter Personalization**:
  - Improved Patients: 39 / 45 (**86.7%**)
  - Mean RMSE: 24.14 mg/dL -> 23.20 mg/dL (**Gain: +0.94 mg/dL**)
  - Mean MARD: 12.82% -> 11.90% (**Gain: +0.91%**)

### 5. Performance Stratified by Clinical Phenotype (+60 min)

| Cohort Phenotype | Patients | Windows | RMSE (mg/dL) | MAE (mg/dL) | MARD (%) | Clarke Zone A (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Healthy** | 15 | 45,718 | 19.58 +/- 7.13 | 14.02 +/- 4.82 | 12.03 +/- 3.02 | 82.34 +/- 8.19 |
| **Prediabetes** | 16 | 48,124 | 24.07 +/- 7.24 | 17.49 +/- 4.74 | 12.93 +/- 1.85 | 79.14 +/- 4.55 |
| **Type 2 Diabetes** | 14 | 40,450 | 32.76 +/- 8.81 | 23.95 +/- 6.74 | 14.44 +/- 2.89 | 73.77 +/- 8.46 |

### 6. What-If Counterfactual Engine Validation and Physiological Boundaries

The What-If engine (`src/glucotwin/twin/simulator.py`) performs counterfactual re-inference through the patient's served fold model (enforcing zero-leakage cross-validation).

#### 6.1 Capabilities (What the Engine Can Do)
- **Model-Based Meal Response**: Evaluates carbohydrate shocks and meal timing offsets (+0m to +120m) by re-inferring future glucose through the patient's held-out test fold model.
- **Uncertainty Quantification**: Returns baseline and scenario quantiles (q10, q50, q90) at +30, +60, and +120 minutes, along with peak glucose shifts and horizon deltas.
- **Assumption-Based and Unvalidated Activity Clearance**: Applies a bounded post-hoc physiological deduction for physical activity (METs > 1.0, 0.38 mg/dL per active minute per net MET, ceiling of 50 mg/dL deduction). This is clearly labeled as assumption-based and unvalidated on trial data in every API response. Origin of coefficient: The clearance coefficient (0.38 mg/dL per active minute per net MET) is an illustrative assumption chosen by the team, with no source and no validation. The effect is capped at 50 mg/dL.
- **Physiology Monotonicity Constraint**: For added carbohydrates, projects the scenario trajectory so that glycemic change from baseline is non-negative and non-decreasing across carbohydrate scales (isotonic projection). When triggered, responses flag `constraint_applied = True` and report `constraint_correction_mgdl`.
- **Cross-Fold Linear Calibration (Optional Flag)**: Linear mapping fit on training-fold meals can be optionally applied via `calibrate_meal_response=true`. By default, the simulator outputs raw model predictions.
- **Provenance Transparency**: Every returned value carries explicit provenance flags: `model_based` for meal impacts, `assumption_based_unvalidated` for exercise deductions, and `food_table` for Indian nutritional lookups from `data/external/indian_meal_templates.csv` (Indian meal library with approximate nutrient values). Note on food reference: IFCT 2017 contains 528 key foods in raw form (plus eggs and poultry), so cooked dishes such as dosa, idli, biryani and dal are not in it; the values here are illustrative estimates; IFCT 2017 (Longvah, Ananthan, Bhaskarachary and Venkaiah, NIN, 2017) is the recommended reference for verifying raw ingredient values; no IFCT data is copied into this repository.

#### 6.2 Limitations (What the Engine Cannot Do)
- **Meal Rise Underestimation**: The raw simulator tends to underestimate meal rises (predicted rise about half of observed postprandial excursions, e.g. mean predicted rise of 32.8 mg/dL vs 66.8 mg/dL observed in Type 2 Diabetes).
- **Not a Closed-Loop Pump**: Does not calculate or prescribe autonomous insulin dosages.
- **Lookback-Constrained**: Does not perform multi-day forward projections; predictions are bounded to the clinically validated 2-hour (+120 min) horizon.
- **No Unmodeled Drug Interactions**: Does not simulate acute drug-drug pharmacodynamic interactions beyond the patient's baseline static EHR features.

#### 6.3 Empirical Validation on Real CGMacros Data
Evaluated on held-out test data in `reports/whatif_validation.json` against explicit pre-specified targets in `params.yaml`:
- **Monotonicity Test (200 real test windows, carbs scaled 0x to 2x)**:
  - Raw Model Monotonicity: 145 / 200 windows (**72.50%**) exhibit monotonically non-decreasing predicted peak glucose (Pre-specified target: 90.0% -> **FAILED**). The raw unconstrained neural network exhibits saturation and small downward dips on 27.5% of windows.
  - Constrained Physiology Projection: 200 / 200 windows (**100.00%**) exhibit monotonically non-decreasing peaks (Pre-specified target: 99.0% -> **PASSED**).
  - Constraint Impact: Applied on 55 / 200 windows (**27.50%**). Mean correction across all windows: **0.3943 mg/dL** (mean among constrained windows: **1.4338 mg/dL**).
- **Data Agreement Test (Held-out CGMacros test meals, N=497 meals)**:
  - Baseline Construction: Candidate meal is strictly removed from baseline inputs (carbs = 0.0, carbs_last_2h = 0.0, time_since_meal_mins advancing). Scenario counterfactual injects candidate meal carbs at meal timestamp.
  - Comparable Definitions: Predicted peak glucose evaluated over +30m, +60m, and +120m horizons compared against observed peak glucose within the same 120-minute postprandial window.
  - Observed 2-Hour Rise Definition: Peak postprandial glucose within 120 minutes of meal onset minus glucose at meal onset: delta_g_obs = max(t in [t_meal, t_meal+120m]) g(t) - g(t_meal).
  - Cross-Fold Mean: Pearson r = **0.3994 +/- 0.1118**, Spearman rho = **0.4042 +/- 0.1207**, Positive effect share = **99.20% +/- 1.60%** (Target: r >= 0.30 -> **PASSED**).
  - Pooled Partial Correlation Controlling for Carbohydrate Intake: **r = 0.3298** (confirms model captures patient-specific glycemic dynamics beyond meal size alone).
  - Stratified Breakdown by Diabetes Status:
    - **Type 2 Diabetes (14 patients, 121 meals)**: Pearson r = **0.3931**, Spearman rho = **0.4074**, Partial r (controlling carbs) = **0.3248**, Positive effect share = **98.35%**, Mean predicted rise = 32.83 mg/dL, Mean observed rise = 66.82 mg/dL (predicted rise roughly half of observed).
    - **Prediabetes (16 patients, 167 meals)**: Pearson r = **0.3031**, Spearman rho = **0.2825**, Partial r (controlling carbs) = **0.2149**, Positive effect share = **100.00%**, Mean predicted rise = 28.44 mg/dL, Mean observed rise = 53.28 mg/dL.
    - **Healthy (15 patients, 209 meals)**: Pearson r = **0.3465**, Spearman rho = **0.3919**, Partial r (controlling carbs) = **0.2789**, Positive effect share = **99.04%**, Mean predicted rise = 25.51 mg/dL, Mean observed rise = 42.44 mg/dL.
- **Cross-Fold Calibration Evaluation (5 held-out folds, N=497 meals)**:
  - Linear mapping from predicted to observed rise fit on training folds and evaluated on held-out test folds:
  - Raw Model: MAE = **31.76 mg/dL**, Bias = **-23.87 mg/dL**.
  - Calibrated Model: MAE = **29.40 mg/dL**, Bias = **-0.29 mg/dL**.
  - Held-out MAE Improvement by Fold: Fold 0 (+5.95 mg/dL), Fold 1 (-1.97 mg/dL), Fold 2 (-3.38 mg/dL), Fold 3 (+6.44 mg/dL), Fold 4 (+4.77 mg/dL).
  - Calibration Rule Outcome: Intercept meal calibration improved held-out MAE in only 3 of 5 folds, failing the pre-set decision rule of 4 of 5 folds.
  - Decision: The simulator defaults to the raw model output (`calibrate_meal_response=false`). Linear calibration is preserved as an optional flag (`calibrate_meal_response=true`) for callers who prefer scaled estimates.
- **Zero-Meal Sanity Test**:
  - A 0 g carbohydrate meal produces exactly **0.000000 mg/dL** change across all horizons (Pre-specified target: <= 0.0001 mg/dL -> **PASSED**).

#### 6.4 Physiological Boundaries
- All continuous glucose predictions and trajectory intervals are strictly clamped to the clinically safe range of **[40.0, 400.0] mg/dL**.
- Exercise clearance deduction is strictly capped at **50.0 mg/dL** to prevent physiological underflow.

## Generated Artifacts and Figures
- Cross-Validation Summary: [metrics_cv.json](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/metrics_cv.json)
- What-If Validation Report: [whatif_validation.json](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/whatif_validation.json)
- Phenotype Stratified Table: [metrics_by_status.csv](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/metrics_by_status.csv)
- Per-Patient Granular Results: [per_patient_results.parquet](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/per_patient_results.parquet)
- Diagnostic Figures:
  - [Clarke Error Grid Analysis](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/figures/clarke_error_grid.png)
  - [Forecast Traces with Conformal Intervals](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/figures/forecast_traces.png)
  - [Conformal Calibration Curves](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/figures/calibration_curves.png)
  - [SHAP Feature Importance Summary](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/figures/shap_summary.png)
  - [TFT Multi-Head Temporal Attention Map](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/figures/tft_attention.png)
  - [Personalization Error Trend Trajectory](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/reports/figures/personalization_trend.png)

## Training Data License and Attribution
- **Training Dataset**: The models, scalers, calibrators, operating thresholds, and manifests in this repository were trained and evaluated on the CGMacros dataset.
- **Data License**: CGMacros is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0), checked on the PhysioNet page.
- **Model Checkpoints License**: In accordance with the ShareAlike provision of CC BY-NC-SA 4.0, all model weights and checkpoints in `models/checkpoints/` and derived evaluation reports in `reports/` are distributed under CC BY-NC-SA 4.0 for non-commercial research and demonstration use with attribution. See [models/checkpoints/NOTICE.md](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/models/checkpoints/NOTICE.md).
- **Data Citation**:
  - Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). CGMacros: a scientific dataset for personalized nutrition and diet monitoring (version 1.0.0). PhysioNet. https://doi.org/10.13026/3z8q-x658
  - Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215-e220.
- **Redistribution Policy**: No raw clinical data is redistributed in this repository. For comprehensive dataset licensing information, see [DATA_LICENSES.md](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/DATA_LICENSES.md).
