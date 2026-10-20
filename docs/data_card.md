# Data Card: GlucoTwin Multimodal Data Layer

## 1. Overview
GlucoTwin harmonizes multimodal data across static Electronic Health Records (EHR), continuous glucose monitoring (CGM) time series, wearable physiological telemetry (heart rate, steps, activity), and meal intake logs. The data architecture supports predictive modeling for Type 2 Diabetes (T2D) management and risk stratification.

All datasets conform to strict Pandera validation schemas and are partitioned at the patient level to prevent data leakage.

## 2. Source Datasets

### 2.1 CGMacros
- Description: Open continuous glucose monitoring cohort with paired physical activity, heart rate, meal logs, and laboratory bio markers.
- Patients: 45 participants with valid time series and baseline lab biomarkers.
- Devices: Dexcom G6 and Abbott FreeStyle Libre Pro CGM, Fitbit wearable activity trackers, Contour Next fingerstick glucometers.
- Modalities:
  - Static: Age, biological sex, BMI, weight (lbs to kg), height (inches to cm), HbA1c (PDL Lab), fasting glucose, lipid panel (Cholesterol, HDL, LDL, Triglycerides), fasting insulin.
  - Dynamic: 5-minute CGM glucose, heart rate (bpm), physical activity level (METs / 10), and detailed macronutrient meal logs (carbs, protein, fat, fiber, calories).
  - Validation: 3 paired fingerstick blood glucose calibration measurements per subject.
- Feature Windows: Produces 134,292 feature windows across 45 participants (min: 1,988, median: 2,867, max: 3,654 windows per patient).

### 2.2 AI-READI Synthetic (Static Patient Cohort)
- Description: Synthetic clinical and wearable summary dataset modeled after the AI-READI multimodal diabetes study, structured using OMOP Common Data Model tables.
- Role in GlucoTwin: Contains cross-sectional glucose and wearable activity summary statistics. It is maintained strictly as a static EHR patient table for baseline clinical features, risk stratification, and persona generation. It does not produce continuous dynamic time-series windows and is excluded from sliding sequence split manifests.
- Patients: 10,518 synthetic individuals labeled with `is_synthetic = true`.
- OMOP Concept Mapping:
  - 4245997: BMI (kg/m2)
  - 3004249: Systolic Blood Pressure (mmHg)
  - 3012888: Diastolic Blood Pressure (mmHg)
  - 3036277: Height (cm)
  - 3025315: Weight (kg)
  - 3004410: Hemoglobin A1c (%)
  - 3004501: Fasting Glucose (mg/dL)
  - 3007070: HDL Cholesterol (mg/dL)
  - 3028288: LDL Cholesterol (mg/dL)
  - 3027114: Total Cholesterol (mg/dL)
  - 3022192: Serum Triglycerides (mg/dL)
  - 3016723: Serum Creatinine (mg/dL)
  - 3016244: Fasting Insulin (uIU/mL)
- Clinical Conditions: OMOP concept 201826 (Type 2 Diabetes Mellitus) and 37018196 (Pre-diabetes).

### 2.3 HUPA-UCM (Type 1 Pretraining Cohort)
- Description: Hospital Universitario Principe de Asturias Type 1 Diabetes cohort.
- Patients: 25 patients with paired FreeStyle Libre CGM and Fitbit telemetry.
- Handling & Processing:
  - Tagged as `diabetes_type = "T1D"`. Excluded by default from T2D fine-tuning and evaluation via configuration (`include_hupa_t1d: false`).
  - Automatic delimiter detection handles both semicolon and tab-delimited raw CSV formats across patients.
  - Carb units: Spanish exchange units (raciones) are converted to grams (1 ración = 10 grams carbohydrate). Direct gram entries (gramos) are preserved.
  - Robust file ingestion filters out 71 empty placeholder daily files (<25 bytes).

## 3. Unified Schemas

### 3.1 Patients Schema (`PatientsSchema`)
| Column | Type | Nullable | Range / Allowed Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | String | No | Unique ID | Primary patient identifier |
| `source_dataset` | String | No | cgmacros, ai_readi_synthetic, hupa_ucm | Originating dataset |
| `is_synthetic` | Boolean | No | True / False | Flag for synthetic vs real data |
| `diabetes_type` | String | No | T2D, T1D, none, prediabetes, unknown | Clinical diagnosis category |
| `diabetes_status` | String | No | T2D, prediabetes, healthy, T1D, unknown | Stratification status |
| `age` | Float | Yes | 0.0 to 120.0 | Age in years |
| `sex` | String | Yes | M, F, other, Unknown | Biological sex |
| `bmi` | Float | Yes | 10.0 to 70.0 | Body Mass Index (kg/m2) |
| `hba1c` | Float | Yes | 3.0 to 20.0 | Glycated hemoglobin (%) |
| `fasting_glucose` | Float | Yes | 30.0 to 500.0 | Fasting blood glucose (mg/dL) |
| `systolic_bp` | Float | Yes | 50.0 to 300.0 | Systolic blood pressure (mmHg) |
| `diastolic_bp` | Float | Yes | 30.0 to 200.0 | Diastolic blood pressure (mmHg) |
| `cholesterol` | Float | Yes | 20.0 to 1000.0 | Total cholesterol (mg/dL) |
| `triglycerides` | Float | Yes | 10.0 to 2500.0 | Triglycerides (mg/dL) |
| `hdl` | Float | Yes | 5.0 to 300.0 | HDL cholesterol (mg/dL) |
| `ldl` | Float | Yes | 5.0 to 500.0 | LDL cholesterol (mg/dL) |
| `creatinine` | Float | Yes | 0.1 to 25.0 | Serum creatinine (mg/dL) |
| `insulin` | Float | Yes | 0.1 to 500.0 | Fasting insulin (uIU/mL) |
| `height_cm` | Float | Yes | 50.0 to 250.0 | Height in cm |
| `weight_kg` | Float | Yes | 20.0 to 300.0 | Weight in kg |

### 3.2 Timeseries Schema (`TimeseriesSchema`)
| Column | Type | Nullable | Bounds | Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | String | No | Patient ID | Identifier linking to patient table |
| `timestamp` | Datetime (UTC) | No | Continuous 5-min grid | Standardized UTC timestamp |
| `glucose_mgdl` | Float | Yes | 40.0 to 500.0 mg/dL | Sensor or fingerstick blood glucose |
| `heart_rate` | Float | Yes | 30.0 to 220.0 bpm | Telemetric heart rate |
| `steps` | Float | Yes | >= 0.0 | Cumulative step count in 5-min window |
| `activity_level` | Float | Yes | >= 0.0 | Metabolic equivalent of task (METs) |
| `sleep_stage` | String | Yes | Categorical | Sleep state (light, deep, rem, awake) |
| `hrv` | Float | Yes | >= 0.0 ms | Heart rate variability |
| `missing_flag` | Boolean | No | True / False | Set to True when glucose is missing |

### 3.3 Meals Schema (`MealsSchema`)
| Column | Type | Nullable | Bounds | Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | String | No | Patient ID | Identifier linking to patient table |
| `timestamp` | Datetime (UTC) | No | UTC timestamp | Meal ingestion event timestamp |
| `carbs_g` | Float | No | 0.0 to 400.0 g | Total carbohydrate content in grams |
| `protein_g` | Float | Yes | 0.0 to 300.0 g | Total protein content in grams |
| `fat_g` | Float | Yes | 0.0 to 300.0 g | Total dietary fat in grams |
| `fiber_g` | Float | Yes | 0.0 to 100.0 g | Dietary fiber in grams |
| `calories` | Float | Yes | 0.0 to 3000.0 kcal | Total energy content in kilocalories |
| `meal_label` | String | Yes | Text description | Meal category or food description |

## 4. Harmonization Rules
1. **Grid Regularization**: All time series are mapped to a continuous 5-minute regular grid aligned to UTC.
2. **Outlier Filtering**: Out-of-bounds readings (glucose < 40 or > 500 mg/dL, heart rate < 30 or > 220 bpm) are dropped and tracked.
3. **Physiological Interpolation**: Linear interpolation is strictly bounded to missing intervals of 30 minutes or less (up to 6 consecutive 5-minute steps). Larger gaps remain NaN with `missing_flag = true`.
4. **Quality Reporting**: Detailed missingness percentages, gap counts, and record transformations are exported to `data/processed/data_quality_report.json`.

## 5. Feature Windows and Targets
- **Lookback Window**: 6 hours (72 time steps at 5-minute intervals).
- **Forecast Horizons**: 30 minutes (6 steps), 60 minutes (12 steps), and 120 minutes (24 steps).
- **Modality Partitioning**:
  - Observed Sequence: Historical glucose, rate of change (5m, 15m, 30m, 60m), rolling mean and standard deviation, heart rate, steps, activity level, and meal carbs.
  - Known Future Inputs: Sinusoidal hour of day and day of week, minutes elapsed since last meal, and cumulative carbs in previous 2 hours.
  - Static EHR Inputs: Demographics, baseline HbA1c, fasting glucose, blood pressure, and lipid profiles.
- **Classification Event Labels**:
  - Hyperglycemia: Target glucose >= 180 mg/dL.
  - Severe Hyperglycemia: Target glucose >= 250 mg/dL.
  - Hypoglycemia: Target glucose <= 70 mg/dL.
  - Glucose Spike: Rise of >= 40 mg/dL within 60 minutes from current baseline.

## 6. Partitioning and Leakage Safety
- **Window-Bounded Splitting**: Splits are built strictly over patients who have at least one feature window. Patients without time series (such as static EHR records) remain excluded from time series sequence split manifests.
- **Joint Source and Status Stratification**: Splits are stratified jointly by `source_dataset` and `diabetes_status`. For CGMacros, T2D, prediabetes, and healthy cohorts are each balanced across train (70%), validation (15%), and test (15%) partitions.
- **Strict Disjoint Partitions**: Assertions verify that train, validation, and test patient sets are mutually exclusive with zero patient overlap.

## 7. Food Database (Indian meal library with approximate nutrient values)
- **Path**: `data/external/indian_meal_templates.csv`
- **Fields**: `food`, `serving_g`, `carbs_g`, `protein_g`, `fat_g`, `fiber_g`, `kcal`, `gi_value`, `gi_source`, `composition_source`, `is_approximate`, `notes`.
- **Nutritional Context and Reference Standard**:
  IFCT 2017 contains 528 key foods in raw form (plus eggs and poultry), so cooked dishes such as dosa, idli, biryani and dal are not in it. Dish values in `data/external/indian_meal_templates.csv` are derived from IFCT 2017 raw ingredient values (extracted locally from the reference publication, Longvah et al., NIN, 2017) multiplied by typical home recipe ingredient quantities documented in `data/external/meal_recipes.csv`.
- **Verification and Provenance Policy**:
  - `composition_source` is set to `"derived from IFCT 2017 raw ingredient values (local extraction, not redistributed) and team recipe assumptions"`.
  - `gi_source` is set to `"approximate, not verified"`.
  - `is_approximate` is set to `true` everywhere, because recipe portions, moisture retention/losses, and cooking changes are team assumptions.
  - `notes` is set to `"unverified, from model recall"`.
  No raw tables, book text, or intermediate extracts are redistributed in this repository.

## 8. Data Licenses and Attributions
- **CGMacros Training Cohort**: Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0), checked on the PhysioNet page.
  - Primary Citation: Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). CGMacros: a scientific dataset for personalized nutrition and diet monitoring (version 1.0.0). PhysioNet. https://doi.org/10.13026/3z8q-x658
  - Resource Citation: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215-e220.
- **HUPA-UCM**: License CC BY 4.0, seen on the institutional repository record, confirmed on the Mendeley Data page. Citation: Hidalgo JI, Alvarado J, Botella M, Aramendi A, Velasco JM, Garnica O. HUPA-UCM diabetes dataset. Data in Brief 55 (2024) 110559, doi 10.1016/j.dib.2024.110559; dataset doi 10.17632/3hbcscwz44.1.
- **Synthetic AI-READI (ADVANCE Center, Vanderbilt University Medical Center and Washington University)**: Terms not published on the pages reviewed, raw files not redistributed, source https://hiplab.vumc.org/synthetix, derived from the NIH Bridge2AI AI-READI cohort, cite 'A synthetic multi-modal dataset for type 2 diabetes', Scientific Reports.
- **Synthea**: Not used yet.
- **IFCT 2017**: Raw ingredient nutritional values used with acknowledgment; no tables redistributed; recipes and GI are assumptions.
- **Redistribution Status**: Raw data redistributed: No for every row. See [DATA_LICENSES.md](file:///C:/Users/pande/.gemini/antigravity/scratch/glucotwin/DATA_LICENSES.md) for complete details.

