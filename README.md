# MetroPT-3 Air Compressor Fault Prediction

A machine learning project that predicts faults in a metro train's air compressor (APU) from sensor readings, with a Streamlit web app for interactive predictions.

> **Status: experimental.** The model detects fault patterns it has seen in training, but it does **not** reliably detect new, unseen fault types (see [Results & Limitations](#results--limitations)).

## Dataset

**MetroPT-3** (air compressor of a metro train, Porto, Feb-Aug 2020), about 1.5M rows sampled every 10 seconds. Available from the UCI Machine Learning Repository.

The dataset is **not included** in this repo (file is too large for GitHub). Download it and place it in the project root as `MetroPT3(AirCompressor).csv`.

| Type | Features |
|---|---|
| Analog sensors | `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current` |
| Digital signals | `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses` |

**Labels:** the raw data has no fault column. `Fault_status = 1` is assigned for four known failure windows (18 Apr, 29-30 May, 5-7 Jun, 15 Jul 2020); everything else is `0`. Faults are about 2% of rows.

## Project structure

```
Metro-Air-Compressor-Fault-Prediction/
├── app.py                 # Streamlit web app
├── MetroPT3.ipynb         # EDA, labelling, training, evaluation
├── rf_model.pkl           # Trained Random Forest (created by the notebook)
├── columns.pkl            # Feature column order (created by the notebook)
├── requirements.txt
├── docs/
│   └── SYSTEM_DESIGN.md   # Architecture and data flow
├── .gitignore
└── README.md
```

## How it works

1. **Notebook** loads the CSV, creates `Fault_status`, splits data **by time** (train before 1 Jul 2020, test from 1 Jul 2020), trains a Random Forest with `class_weight="balanced"`, evaluates it, and saves `rf_model.pkl` and `columns.pkl`.
2. **Streamlit app** loads those files, collects the 15 sensor values from the user, and returns a fault probability. A fault is reported when the probability is at or above the threshold (`0.2` by default).

See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for details.

## Setup & run

```bash
git clone https://github.com/<your-username>/Metro-Air-Compressor-Fault-prediction.git
cd Metro-Air-Compressor-Fault-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Put `MetroPT3(AirCompressor).csv` in the project root.
   dataset link : https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip
3. Open and run all cells of `MetroPT3.ipynb` (this creates `rf_model.pkl` and `columns.pkl`).
4. Start the app:

```bash
streamlit run app.py
```

## Results & Limitations

Time-based evaluation (train on earlier faults, test on a later one):

| Held-out fault | Fault recall |
|---|---|
| 18 Apr | ~2% |
| 30 May | ~99% |
| 5-7 Jun | ~48% |
| 15 Jul | 0% |

- **Accuracy is misleading** here: about 98-99% of rows are normal, so a model that always predicts "no fault" scores about 99%. Judge the model by fault-class recall and precision.
- An earlier version used a random train/test split, which leaks near-identical neighbouring rows into the test set and gave inflated (~99.9%) scores.
- Each failure looks different in the sensors, so the model generalises poorly to new failure types.
- Predictions are made from a single reading, with no temporal context.

## Possible improvements

- Rolling-window / trend features over time
- Anomaly detection trained on normal behaviour only
- More fault episodes, or leave-one-episode-out cross-validation
- Threshold tuning for the precision/recall trade-off

## Tech stack

Python, pandas, scikit-learn, joblib, Streamlit

## License

Add a license of your choice (e.g. MIT). The MetroPT-3 dataset has its own license; check the source page.
