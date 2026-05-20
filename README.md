# Hypercast

End-to-end MLOps platform for short-term energy price forecasting across ERCOT electricity markets. Combines an Airflow ETL pipeline, LSTM baseline, and Transformer-based model with MLflow experiment tracking.

## Overview

Hypercast ingests, validates, and processes ISO-style energy market data, then trains and evaluates time-series forecasting models to predict real-time electricity prices.

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | Apache Airflow, Python |
| Modeling | PyTorch (LSTM, Transformer), NumPy |
| Experiment tracking | MLflow |
| Environment | Python 3.10+, venv |

## Project Structure

```
hypercast/
├── airflow/
│   ├── dags/            # Airflow DAGs (ercot_ingestion.py)
│   └── plugins/
├── data/                # Raw and processed market data
├── models/
│   ├── lstm_baseline.py          # LSTM baseline forecaster
│   ├── transformer_forecaster.py # Transformer with positional encoding
│   └── train_baseline.py        # Training entrypoint
├── notebooks/           # Exploratory analysis
└── scripts/             # Utility and data generation scripts
```

## Models

- **LSTM Baseline** — Recurrent sequence model for short-horizon price prediction.
- **TimeSeriesTransformer** — Multi-head self-attention encoder with sinusoidal positional encoding, configurable depth (`d_model`, `nhead`, `num_encoder_layers`).

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Airflow (from airflow/ directory)
airflow db init
airflow scheduler &
airflow webserver

# 4. Train a model
python models/train_baseline.py
```

## Features

- Airflow DAG for automated ERCOT data ingestion and validation
- Pluggable model architecture (swap LSTM ↔ Transformer via config)
- MLflow integration for tracking runs, metrics, and model artifacts
- Modular design — data, models, and training scripts are fully decoupled
