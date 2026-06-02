# 🏥 HAVM — Healthcare Assumption Validation & Monitoring Framework

A production-ready framework for **monitoring deployed healthcare AI models** and **detecting assumption violations** in real time. HAVM continuously validates data integrity, model confidence, and demographic fairness to ensure safe and reliable AI-assisted clinical decision-making.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Governance Dashboard                    │
│               (Streamlit — dashboard.py)                │
├──────────┬──────────┬──────────────┬────────────────────┤
│  Data    │   OOD    │ Uncertainty  │    Fairness        │
│  Drift   │Detection │  Monitoring  │   Monitoring       │
│  (KS)    │ (IsoFor) │  (RF proba)  │  (Group metrics)   │
├──────────┴──────────┴──────────────┴────────────────────┤
│              Random Forest Classifier                    │
│                  (train_model.py)                        │
├─────────────────────────────────────────────────────────┤
│        Synthetic Data Engine (data_loader.py)            │
│         UCI Heart Disease – inspired features            │
└─────────────────────────────────────────────────────────┘
```

| Module | File | Purpose |
|--------|------|---------|
| **Configuration** | `config.py` | Centralised constants, paths, and thresholds |
| **Data Loader** | `data_loader.py` | Synthetic data generation (training + drifted incoming) |
| **Model Training** | `train_model.py` | Random Forest training, evaluation, and persistence |
| **Drift Detection** | `drift_detection.py` | Kolmogorov-Smirnov test per feature |
| **OOD Detection** | `ood_detection.py` | Isolation Forest anomaly scoring |
| **Uncertainty** | `uncertainty_monitor.py` | Prediction confidence via `predict_proba` |
| **Fairness** | `fairness_monitor.py` | Per-group accuracy / precision / recall / F1 |
| **Dashboard** | `dashboard.py` | Interactive Streamlit governance UI |

---

## ✅ Prerequisites

- **Python 3.8+** (tested with 3.10 / 3.11 / 3.12)
- pip (or any compatible package manager)

---

## 🚀 Installation

```bash
# 1. Navigate to the project directory
cd havm

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Step 1 — Train the Model

```bash
python train_model.py
```

This will:
1. Generate synthetic training data (700 samples) and incoming data (200 samples with drift)
2. Train a Random Forest classifier (100 trees, seed 42)
3. Print accuracy & classification report
4. Save the model to `models/rf_model.joblib`

### Step 2 — Run Individual Monitors (Optional)

Each module can be executed standalone to produce a formatted CLI report:

```bash
python drift_detection.py
python ood_detection.py
python uncertainty_monitor.py
python fairness_monitor.py
```

### Step 3 — Launch the Governance Dashboard

```bash
streamlit run dashboard.py
```
*Note: If the `streamlit` command is not recognized, run:*
```bash
python -m streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501` and provides:
- **Real-time status cards** for all four monitors
- **Interactive charts** (Plotly) for drift, anomaly scores, confidence distributions, and fairness comparisons
- **Adjustable thresholds** via sidebar sliders
- **Data regeneration** and **model retraining** buttons

---

## 📁 Project Structure

```
havm/
├── config.py                 # Paths, feature list, thresholds
├── data_loader.py            # Synthetic data generation & loading
├── train_model.py            # Model training pipeline
├── drift_detection.py        # KS-test drift detection
├── ood_detection.py          # Isolation Forest OOD detection
├── uncertainty_monitor.py    # Prediction uncertainty analysis
├── fairness_monitor.py       # Demographic fairness evaluation
├── dashboard.py              # Streamlit governance dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Auto-generated CSV datasets
│   ├── train_data.csv
│   └── incoming_data.csv
└── models/                   # Saved model artifacts
    └── rf_model.joblib
```

> The `data/` and `models/` directories are created automatically on first run.

---

## 🔬 Module Details

### 1. Data Drift Detection (`drift_detection.py`)
- **Method**: Two-sample Kolmogorov-Smirnov test
- **Scope**: All 13 features compared between training and incoming distributions
- **Alert**: Overall drift flagged when > 30% of features show significant shift (p < 0.05)

### 2. Out-of-Distribution Detection (`ood_detection.py`)
- **Method**: Isolation Forest (100 trees, 10% contamination)
- **Scope**: Trained on training features; scores every incoming sample
- **Alert**: OOD alert when > 5% of incoming samples are flagged as outliers

### 3. Uncertainty Monitoring (`uncertainty_monitor.py`)
- **Method**: Maximum predicted class probability from `RandomForestClassifier.predict_proba`
- **Threshold**: Predictions with confidence < 60% are flagged as uncertain
- **Alert**: High uncertainty when > 15% of predictions are uncertain

### 4. Fairness Monitoring (`fairness_monitor.py`)
- **Method**: Per-group accuracy, precision, recall, F1 across `sex` (Male / Female)
- **Metric**: Absolute disparity between groups
- **Alert**: Fairness violation when any metric disparity exceeds 10%

---

## 🛠️ Technology Stack

| Technology | Role |
|-----------|------|
| **Python** | Core language |
| **Streamlit** | Interactive dashboard |
| **Scikit-Learn** | Random Forest, Isolation Forest, metrics |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computation |
| **SciPy** | Kolmogorov-Smirnov statistical test |
| **Plotly** | Interactive charts |
| **Joblib** | Model serialisation |

---

## 📝 Notes

- **Synthetic data**: The framework generates realistic synthetic data inspired by the UCI Heart Disease Dataset. No external dataset download is required — everything runs offline out of the box.
- **Drift injection**: The incoming dataset has subtle distribution shifts (age +5, cholesterol +30, blood pressure +10) and 10 clearly anomalous records to demonstrate monitoring capabilities.
- **Reproducibility**: All random operations use `seed=42` for deterministic results.
- **Extensibility**: Each monitor module exposes a clean API (`get_*_status()`) that the dashboard consumes. New monitors can be plugged in by following the same pattern.

---

## 📄 License

This project is developed as a capstone project for educational purposes.
