# scripts/tune_hyperparameters.py
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from scripts.train_transformer import train_transformer

def objective(trial):
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'nhead': trial.suggest_categorical('nhead', [4, 8]),
        'num_layers': trial.suggest_int('num_layers', 2, 6),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [256, 512, 1024]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-4),
        'epochs': 50  # Reduced for tuning
    }
    
    mape = train_transformer(config, trial=trial)
    return mape

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hypercast-tuning")
    
    mlflc = MLflowCallback(
        tracking_uri="file:./mlruns",
        metric_name="test_mape"
    )
    
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=20, callbacks=[mlflc])
    
    print("Best trial:")
    print(f"  MAPE: {study.best_trial.value:.2f}%")
    print(f"  Params: {study.best_trial.params}")