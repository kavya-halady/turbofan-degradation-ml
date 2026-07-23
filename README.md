# ✈️ turbofan-degradation-ml— Predictive Maintenance MLOps Pipeline

**Predict Remaining Useful Life (RUL) of industrial turbofan engines from sensor telemetry, served via a production-grade ML pipeline with monitoring and automated retraining.**

---

## 📌 Problem Statement

Unplanned equipment failure costs manufacturers millions in downtime. Instead of fixing machines on a fixed schedule (reactive/preventive maintenance), this project builds a **predictive maintenance system** that:

1. Predicts **Remaining Useful Life (RUL)** — how many operating cycles are left before an engine is likely to fail (regression).
2. Flags **imminent failure risk** — will this engine fail within the next N cycles? (binary classification).
3. Serves both models through a REST API that a maintenance dashboard could call in real time.
4. Monitors incoming sensor data for **drift** and triggers retraining when the model's assumptions no longer hold.

This mirrors real systems used in aviation, manufacturing, and energy (GE Predix, Siemens MindSphere, Rolls-Royce engine health monitoring).

---

## 📊 Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**
- Source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (mirrored on Kaggle as "NASA Turbofan Jet Engine Data Set")
- 4 sub-datasets (FD001–FD004) simulating engines under different operating conditions and fault modes
- Each engine has multi-cycle sensor readings (21 sensors + 3 operational settings) until failure
- Perfect for time-series feature engineering and RUL regression — this is *the* canonical academic/industry benchmark for predictive maintenance

Start with **FD001** (single operating condition, single fault mode) to keep the first pass simple, then extend to FD002–FD004 once your pipeline works end to end.

---

## 🧱 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data & Features | pandas, NumPy, scikit-learn |
| Modeling | scikit-learn (baseline), XGBoost / LightGBM (primary), optional LSTM (PyTorch) for a stretch goal |
| Experiment Tracking | MLflow |
| Hyperparameter Tuning | Optuna (or GridSearchCV/RandomizedSearchCV) |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Monitoring & Drift | Evidently AI |
| Orchestration (stretch) | Prefect or Kubeflow Pipelines |
| Testing | pytest |

---

## 🗂️ Project Structure

```
turbineiq/
├── data/
│   ├── raw/                     # original C-MAPSS txt files (gitignored)
│   ├── interim/                 # cleaned, not yet featurized
│   └── processed/                # train/val/test splits, ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data/
│   │   ├── ingest.py             # Step 1: load & clean raw C-MAPSS files
│   │   └── preprocess.py         # missing values, normalization, RUL labeling
│   ├── features/
│   │   └── build_features.py     # Step 2: rolling stats, degradation trends, PCA
│   ├── data_split.py             # Step 3: engine-level train/val/test split
│   ├── models/
│   │   ├── train.py              # Step 4: model training (regression + classification)
│   │   ├── evaluate.py           # Step 5: metrics, cross-validation
│   │   └── tune.py               # Optuna hyperparameter search
│   ├── api/
│   │   ├── main.py               # Step 6: FastAPI app
│   │   └── schemas.py            # Pydantic request/response models
│   └── monitoring/
│       └── drift_check.py        # Step 7: Evidently drift reports
├── models/                       # serialized models (.pkl / .joblib), gitignored
├── mlruns/                       # MLflow tracking store, gitignored
├── tests/
│   ├── test_features.py
│   ├── test_api.py
│   └── test_model.py
├── .github/workflows/
│   └── ci.yml                    # lint, test, retrain-on-schedule
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config.yaml                   # paths, hyperparameters, thresholds
└── README.md
```

---

## 🔁 Pipeline Steps (mapped to what you asked to learn)

### Step 1 — Data Collection & Preprocessing (`src/data/`)
- Download C-MAPSS `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- Parse fixed-width sensor logs into a tidy DataFrame (engine_id, cycle, 3 settings, 21 sensors)
- Drop sensors with zero variance (several C-MAPSS sensors are constant — a great real "handling messy data" lesson)
- Compute the **RUL label** for training data: `RUL = max_cycle_for_engine − current_cycle`
- Normalize sensor readings with `MinMaxScaler` or `StandardScaler` (fit on train only, save the scaler)

### Step 2 — Feature Engineering (`src/features/`)
- Rolling mean/std over a window of cycles per engine (captures degradation trend, not just a snapshot)
- Cumulative sum / slope of key sensors (rate of degradation)
- Piecewise RUL clipping (cap RUL at ~125 cycles — standard C-MAPSS trick since engines behave normally early on and only degrade near end-of-life)
- Optional: PCA to reduce correlated sensor dimensions

### Step 3 — Data Splitting (`src/data_split.py`)
- Split by **engine ID**, not by row — never leak a single engine's cycles across train/val/test
- Use the provided `test_FD001.txt` + `RUL_FD001.txt` as your true holdout test set
- For the classification task ("fail within N cycles"), check class balance and use stratified splitting or `class_weight='balanced'` if imbalanced

### Step 4 — Model Selection & Training (`src/models/train.py`)
- Baseline: Linear Regression / Logistic Regression (sanity check)
- Primary: **XGBoost / LightGBM** regressor for RUL, classifier for failure-within-N-cycles
- Stretch goal: LSTM sequence model on raw cycle windows (great for learning deep learning on time series later)
- Log every run (params, metrics, model artifact) to **MLflow**

### Step 5 — Model Evaluation & Optimization (`src/models/evaluate.py`, `tune.py`)
- Regression metrics: RMSE, MAE, and the official **NASA scoring function** (asymmetric — penalizes late predictions more than early ones, which is realistic: predicting failure too late is worse than a false alarm)
- Classification metrics: precision, recall, F1, ROC-AUC (recall matters more here — missing a real failure is costly)
- Hyperparameter tuning via **Optuna**
- 5-fold cross-validation grouped by engine ID to avoid leakage

### Step 6 — Model Deployment (`src/api/main.py`)
- **FastAPI** service exposing:
  - `POST /predict/rul` → returns predicted remaining cycles
  - `POST /predict/failure-risk` → returns failure probability within N cycles
  - `GET /health` → liveness check
  - `GET /model-info` → current model version, training date, metrics
- Containerize with **Docker**; `docker-compose.yml` spins up API + MLflow UI together
- Load the serialized model + scaler at startup, not per-request

### Step 7 — Continuous Learning & Monitoring (`src/monitoring/`)
- **Evidently AI** report comparing incoming production data distribution vs. training distribution (data drift)
- Track live prediction error once ground truth (actual failures) becomes available
- **GitHub Actions** scheduled job: retrain weekly / when drift score exceeds threshold, log new run to MLflow, promote to production only if it beats the current model on holdout metrics
- Simple model registry pattern: `models/production/` vs `models/candidate/`

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/turbineiq.git
cd turbineiq
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Get the data (place raw C-MAPSS files in data/raw/)
python src/data/ingest.py

# 2. Build features
python src/features/build_features.py

# 3. Train & evaluate
python src/models/train.py --config config.yaml
python src/models/evaluate.py

# 4. Serve the model
uvicorn src.api.main:app --reload

# 5. Or run everything containerized
docker-compose up --build
```

Example API call once running:
```bash
curl -X POST http://localhost:8000/predict/rul \
  -H "Content-Type: application/json" \
  -d '{"engine_id": 12, "cycle_window": [...sensor readings...]}'
```

---

## 📈 Suggested Milestones (learn-as-you-build checklist)

- [ ] Load and explore FD001 in a notebook, plot sensor degradation curves per engine
- [ ] Build the cleaning + RUL-labeling script, write a unit test for it
- [ ] Engineer rolling-window features, compare model performance with vs. without them
- [ ] Train baseline vs. XGBoost, log both to MLflow, compare in the MLflow UI
- [ ] Tune hyperparameters with Optuna, beat your baseline on the NASA scoring function
- [ ] Wrap the best model in FastAPI, test locally with `curl` / Postman
- [ ] Dockerize the API, run it in a container
- [ ] Add an Evidently drift report comparing FD001 test data to FD002 (different operating condition) to *see* drift in action
- [ ] Set up a GitHub Actions workflow that runs tests on every push
- [ ] (Stretch) Add a scheduled retraining workflow
- [ ] (Stretch) Extend to FD002–FD004 for multi-condition robustness
- [ ] (Stretch) Swap XGBoost for an LSTM and compare

---

## 📄 License

MIT — free to use for learning and portfolio purposes.

## 🙏 Acknowledgments

- A. Saxena and K. Goebel (2008). *"Turbofan Engine Degradation Simulation Data Set"*, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
