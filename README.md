# Multivariate Time Series Prediction Using Deep Learning

Predicting household appliance energy consumption every 10 minutes from indoor sensors, outdoor weather and recent energy use. The project compares two classic baselines (Linear Regression, Random Forest) with three deep learning sequence models (LSTM, GRU, CNN-LSTM) built in PyTorch, then tunes the deep learning models and evaluates everything on a held-out test month.

**Final model: LSTM** (1 LSTM layer with 64 units → dropout 0.2 → dense 32 → output, 28,481 parameters), saved as `models/lstm_model.pth`.

- Best on every test metric, with 12% lower MAE than Linear Regression.
- Hyperparameter tuning changed test results by less than 3%, so the simpler untuned LSTM was kept.
- Random Forest fails on the test month because May is warmer (up to 26.1 °C outside) than anything in the training data (max 19.7 °C), and tree models cannot extrapolate.
- Most of the remaining error comes from sudden consumption spikes (for example an oven or kettle switching on) that the sensors cannot anticipate.

## Dataset

[Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) stored as `data/raw/energy_data_set.csv`.

- 19,735 readings at 10 minute intervals, 11 January – 27 May 2016, from one house in Belgium
- No missing values, duplicates or time gaps
- **Target:** `Appliances`, the appliance energy use in Wh per 10 minutes (median 60 Wh, max 1,080 Wh, strongly right-skewed)
- **Inputs:** lighting energy, temperature and humidity in 9 locations (`T1`–`T9`, `RH_1`–`RH_9`), outdoor weather (`T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`) and two random noise columns (`rv1`, `rv2`) that are removed

## Project structure

```
├── data/
│   ├── raw/energy_data_set.csv          # original dataset
│   └── processed/                       # created by the notebooks
│       ├── energy_data_eda.csv
│       └── energy_data_features.csv
├── notebooks/
│   ├── EDA_preprocessing.ipynb          # 1. exploration and data checks
│   ├── feature_engineering.ipynb        # 2. lag, rolling and time features
│   ├── model_development_and_training.ipynb  # 3. baselines + LSTM, GRU, CNN-LSTM
│   └── model_optimization.ipynb         # 4. tuning and final test evaluation
├── models/                              # trained models, scalers, feature list
├── reports/                             # tuning results and final test results (CSV)
├── requirements.txt
└── README.md
```

## Pipeline

1. **EDA and preprocessing** (`EDA_preprocessing.ipynb`): sorts the data by time and checks missing values, duplicates and time gaps. It studies the target distribution, daily and weekly patterns, correlations and outliers. Outliers flagged by the IQR rule (10.8%) are real high-usage events, so they are kept.
2. **Feature engineering** (`feature_engineering.ipynb`): creates lag features (10, 30 and 60 minutes ago), rolling mean and standard deviation over the previous 1 and 3 hours, time features and a temperature × humidity interaction. All energy-based features use only past values to avoid leakage. Feature relevance is checked with correlation, Random Forest importance and RFE. Output: `energy_data_features.csv`.
3. **Model development** (`model_development_and_training.ipynb`):
   - **Features:** encodes hour and day of week as sine/cosine and drops `rv1`, `month` and `day_of_month`.
   - **Split and scaling:** splits the data chronologically into 70% train, 10% validation and 20% test, and scales it with `StandardScaler` fitted on the training data only.
   - **Inputs:** baselines use one row each; the deep learning models use sliding windows of 6 steps (1 hour) with 37 features.
   - **Models:** trains Linear Regression, Random Forest, LSTM, GRU and CNN-LSTM.
   - **Training setup:** Huber loss, Adam, weight decay, dropout, gradient clipping, a learning-rate scheduler and early stopping.
4. **Optimization** (`model_optimization.ipynb`): random search over learning rate, batch size, epochs, layers and units (20 trials per model), followed by dropout tuning and early stopping. It ends with a test-set comparison against the baselines, MAPE and residual analysis.

## Setup

Requirements: Python 3.13 (other recent Python 3 versions may work but are not tested).

```bash
git clone https://github.com/iamafrar/Multivariate-Time-Series-Prediction-Using-Deep-Learning.git
cd Multivariate-Time-Series-Prediction-Using-Deep-Learning

python3 -m venv .venv
source .venv/bin/activate     
pip install -r requirements.txt
```

Training runs on an Apple Silicon GPU (MPS) when available and falls back to the CPU otherwise.

## How to run

Run the notebooks **in order**. Each one saves the files the next one needs.

**Jupyter**

```bash
jupyter notebook notebooks/
```

Open each notebook in turn and choose **Run All**.
| Notebook | Approximate runtime | Main outputs |
|---|---|---|
| `EDA_preprocessing.ipynb` | < 1 min | `data/processed/energy_data_eda.csv` |
| `feature_engineering.ipynb` | 1–2 min | `data/processed/energy_data_features.csv` |
| `model_development_and_training.ipynb` | a few minutes | `models/*_model.pth`, `X_scaler.pkl`, `y_scaler.pkl`, `feature_columns.pkl`, `linear_regression.pkl`, `random_forest.pkl` |
| `model_optimization.ipynb` | 30–40 min (78 training runs) | `models/*_optimized_model.pth`, `reports/*.csv` |

Runtimes were measured on an Apple Silicon Mac and will be longer on CPU only. A fixed random seed (42) is used, but results can vary slightly between hardware.

## Saved outputs

| File | Description |
|---|---|
| `models/lstm_model.pth` | Final LSTM model (state dict plus input size, hidden size, dropout, sequence length) |
| `models/gru_model.pth`, `models/cnn_lstm_model.pth` | GRU and CNN-LSTM models |
| `models/*_optimized_model.pth` | Tuned models with their hyperparameters |
| `models/X_scaler.pkl`, `models/y_scaler.pkl` | Scalers fitted on the training data; required to use the models on new data |
| `models/feature_columns.pkl` | The 37 input columns, in order |
| `models/linear_regression.pkl`, `models/random_forest.pkl` | Baseline models |
| `reports/final_test_results.csv` | Test metrics for all models |
| `reports/model_optimization_comparison.csv` | Test metrics before and after tuning |
| `reports/*_tuning_results.csv`, `reports/*_dropout_results.csv` | Random search and dropout experiments |

## Limitations and future work

- One house and 4.5 months of data (winter to spring), so the model may not generalise to other homes or seasons.
- The model predicts one step ahead (the current 10-minute interval). Multi-step forecasting (for example 1 hour ahead) is a natural extension.
- The differences between LSTM, GRU and CNN-LSTM are small. Several random seeds and walk-forward validation would make the ranking more reliable.
- Possible next experiments: gradient boosting (XGBoost, LightGBM), Transformer-based models, prediction intervals and SHAP explanations.
