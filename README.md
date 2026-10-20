# GlucoTwin

An India-aware digital twin for proactive type 2 diabetes management.
Built for the Happiest Health Digital Twin Challenge 2026.

## Team

### Team name
ZIZ

### College
Chandigarh University

### Team leader
[LIKHITH SAI SEEMAKURTHI, seemakurthilikhithsai0@gmail.com]

### Members
[PRITESH KUMAR PANDEY, pandeypritesh958@gmail.com]

## Links
- Project walkthrough and live demo: [Watch on YouTube](https://youtu.be/DC1_NouZk5E)
- Presentation: [docs/presentation/ZIZ_GlucoTwin_Presentation.pdf](docs/presentation/ZIZ_GlucoTwin_Presentation.pdf)
- Architecture diagram: [docs/architecture/GlucoTwin_Architecture.pdf](docs/architecture/GlucoTwin_Architecture.pdf)
- Screenshots: [docs/screenshots](docs/screenshots)

## Problem and use case

Diabetes care is mostly reactive. A patient is seen every few months, and a glucose spike is noticed after it has happened. Between visits the doctor has no continuous view of the patient.

GlucoTwin builds a virtual copy of one patient and keeps it up to date. It joins two data streams:
- Static data: age, sex, BMI, HbA1c, fasting glucose and similar fields from the health record.
- Dynamic data: a continuous glucose monitor (CGM) stream, with heart rate, steps and meals where available.

From these it predicts glucose 30, 60 and 120 minutes ahead, warns about adverse events, explains each prediction, adapts to the patient, and lets a doctor test "what if" changes such as a different Indian meal or a walk.

The first target condition is type 2 diabetes. All demo data in this repository is synthetic or de-identified research data.

## What it does

- Forecasts glucose at +30, +60 and +120 minutes, with calibrated uncertainty bands (q10 to q90, conformal calibration).
- Alerts only for events that passed a stated reliability gate. Everything else is shown as a risk estimate, never as an alert.
- Safety floor rules on the observed reading (low, urgent low, rapid fall). These are rule-based, not predicted.
- Explains each forecast (permutation importance for the GRU, SHAP for LightGBM).
- Personalizes with small per-patient LoRA-style adapters that are only promoted if they improve held-out error.
- What-if simulator with an Indian meal library and an activity option, with the provenance of every input shown.
- Doctor summary with time in range, GMI, CV and an AGP view when enough data exists. It is template based and uses no language model.

## Results

Evaluation uses 5-fold cross-validation where every test patient is unseen (leave-patients-out) on the 45 CGMacros participants (14 type 2 diabetes, 16 prediabetes, 15 healthy; 134,292 windows). Values are mean ± standard deviation across folds.

<!-- ## 3.1 5-Fold Leave-Patients-Out Cross-Validation Results -->
| Horizon | Persistence RMSE | LightGBM RMSE | GRU RMSE (served) |
| :--- | :--- | :--- | :--- |
| +30 min | 20.06 ± 0.89 | 17.62 ± 0.89 | 18.86 ± 1.01 |
| +60 min | 29.64 ± 1.86 | 26.86 ± 2.77 | 26.43 ± 1.94 |
| +120 min | 40.14 ± 3.15 | 35.20 ± 5.30 | 32.86 ± 2.63 |
<!-- ## 4. Repository Structure -->

RMSE is in mg/dL. The full table (MAE, MARD, Clarke error grid, TFT, XGBoost) is on the Evidence page of the dashboard and in reports/.

Gain over persistence is real but modest. At +30 min LightGBM lowers the per-patient RMSE by 2.39 mg/dL (95% bootstrap CI 1.92 to 2.83) and improves 43 of 45 patients. The GRU, LightGBM and XGBoost are within one standard deviation of each other at +30 and +60 min.

Uncertainty: the conformal 80% interval covered 80.2% of held-out points at +30 min (GRU).

Usable alerts: hyperglycemia +30 min (recall 65 ± 10%, precision 61 ± 4%, 0.8 ± 0.2 false alerts per patient-day), severe hyperglycemia +30 and +60 min, and spike +30 and +60 min. Hypoglycemia prediction did not meet the gate, so low glucose is covered by observed-value rules.

Personalization: adapters improved 39 of 45 patients (MARD 12.82% to 11.90%). A simple per-patient bias correction gave a similar RMSE gain, so treat this as a small effect.

What-if: the meal effect is directionally right (cross-fold Pearson r = 0.40 ± 0.11) but underestimated, about half of the observed rise in type 2 diabetes meals (32.7 vs 66.8 mg/dL).

## Architecture

See the diagram in docs/architecture/. In short:

```
EHR features (static) ----\
                           +--> Data layer (validation, windows, patient-level splits)
CGM / wearable stream ----/           | (Kafka replay)
                                      v
                             Twin core: GRU fusion model + LightGBM fallback
                             conformal calibration, LoRA adapters, explanations
                                      |
         +----------------------------+---------------------------+
         v                            v                           v
Forecast + alerts             What-if simulator            Doctor summary
         \                            |                           /
          +--------- FastAPI (REST + WebSocket) -----------------+
                                      |
                            Next.js doctor dashboard
```

State is stored in PostgreSQL with TimescaleDB. Prometheus and Grafana monitor latency and alerts.

## Tech stack

| Area | Tools |
| :--- | :--- |
| Language | Python 3.11, TypeScript |
| Data | Polars, pandas, pandera |
| Models | PyTorch (GRU, TFT experiment), LightGBM, XGBoost, SHAP |
| Backend | FastAPI, SQLAlchemy, Alembic, aiokafka |
| Storage and streaming | PostgreSQL with TimescaleDB, Kafka (KRaft) |
| Frontend | Next.js 14, Tailwind CSS, Recharts, Zustand, Framer Motion, lucide-react |
| Ops | Docker Compose, GitHub Actions, MLflow, Prometheus, Grafana, ruff, mypy, pytest, Vitest |

## Data

No raw data is included in this repository.

| Source | Use | License |
| :--- | :--- | :--- |
| CGMacros (PhysioNet) | All model training and evaluation | CC BY-NC-SA 4.0 |
| HUPA-UCM (Mendeley Data) | Optional type 1 pretraining, not used for evaluation | CC BY 4.0 |
| Synthetic AI-READI (ADVANCE Center) | Static patient table only | Terms not published |
| Indian meal library | Team estimates, derived from IFCT 2017 raw ingredient values and recipe assumptions | See DATA_LICENSES.md |

Citations and details: [DATA_LICENSES.md](DATA_LICENSES.md) and [docs/data_card.md](docs/data_card.md).

## Run it

You need Docker Desktop. The trained model files are in models/checkpoints, so no data download is needed.

```bash
cp .env.example .env # make sure DEMO_MODE=true
# on Windows Command Prompt use 'copy .env.example .env'
docker compose up --build
```

| Service | Address |
| :--- | :--- |
| Dashboard | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Grafana | http://localhost:3001 |
| Prometheus | http://localhost:9090 |
| MLflow | http://localhost:5000 |

In demo mode the stack generates three synthetic demo patients at startup and also lists four synthetic personas. Predictions come from the shipped checkpoints.

Tests:

```bash
python -m venv .venv && source .venv/bin/activate # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
cd frontend && npm ci && npm test
```

Reproducing the training needs the three datasets above, placed under data/raw/:

```bash
python -m src.glucotwin.cli data run-all
python -m src.glucotwin.cli models cv
```

## Repository layout

```
src/glucotwin/    data layer, models, twin core, serving, streaming, summary
frontend/         Next.js dashboard
models/           trained checkpoints (CC BY-NC-SA 4.0)
reports/          cross-validated metrics and figures
deploy/           Dockerfiles, Kubernetes manifests, Prometheus, Grafana
docs/             model card, data card, limitations, screenshots, architecture, presentation
tests/            unit and integration tests
scripts/          demo data, smoke test, clean-clone CI checks
```

## Limitations

- Only 14 type 2 diabetes participants were available for evaluation.
- The link between the health record and the wearable stream is synthetic.
- Predicted hypoglycemia alerts are not reliable. Low glucose is covered by observed-value rules.
- Meal rises are underestimated by about half in validation.
- Meal nutrient values are estimates, and glycemic index values are unverified.
- The activity effect in the what-if simulator is an assumption with no validation.
- There has been no prospective or clinical study.

More detail is in [docs/model_card.md](docs/model_card.md) and [docs/limitations.md](docs/limitations.md).

## License

Source code: MIT (see [LICENSE](LICENSE)).

Trained model files and evaluation reports: CC BY-NC-SA 4.0, because they were trained on CGMacros. Non-commercial use with attribution and share-alike. See models/checkpoints/NOTICE.md and reports/NOTICE.md.

This repository is for non-commercial research and demonstration.

## Credits

- CGMacros: Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., and Das, A. (2025). CGMacros: a scientific dataset for personalized nutrition and diet monitoring (version 1.0.0). PhysioNet. https://doi.org/10.13026/3z8q-x658
- HUPA-UCM: Hidalgo JI, Alvarado J, Botella M, Aramendi A, Velasco JM, Garnica O. HUPA-UCM diabetes dataset. Data in Brief 55 (2024) 110559. https://doi.org/10.1016/j.dib.2024.110559
- Food composition reference: Longvah T, Ananthan R, Bhaskarachary K, Venkaiah K. Indian Food Composition Tables 2017. National Institute of Nutrition, ICMR (used as a reference only, no tables are copied).
- Main libraries: PyTorch, LightGBM, XGBoost, SHAP, FastAPI, Next.js, Recharts, Tailwind CSS, shadcn/ui, Framer Motion, and lucide-react (ISC license) for icons.

## How it was built

The project was built with AI-assisted coding (Google Antigravity). The team specified the design, reviewed the results and tests, and is responsible for the code and the claims in this repository.

## Disclaimer

GlucoTwin is decision support for research and demonstration. It is not a medical device and must not be used to diagnose, treat or dose insulin.
