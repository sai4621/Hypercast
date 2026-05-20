# scripts/train_transformer.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from models.transformer_forecaster import TimeSeriesTransformer
import optuna

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_metrics(y_true, y_pred):
    """Calculate multiple metrics"""
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {'mape': mape, 'rmse': rmse, 'mae': mae}

def train_transformer(config, trial=None):
    # Load data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # Train/val/test split
    train_size = int(0.7 * len(X_train))
    val_size = int(0.15 * len(X_train))
    
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    
    X_val = X_train[train_size:train_size+val_size]
    y_val = y_train[train_size:train_size+val_size]
    
    X_test = X_train[train_size+val_size:]
    y_test = y_train[train_size+val_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split), 
        torch.FloatTensor(y_train_split)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesTransformer(
        input_size=X_train.shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        horizon=y_train.shape[1]
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    early_stopping = EarlyStopping(patience=15)
    
    # Training loop
    best_val_mape = float('inf')
    
    with mlflow.start_run():
        mlflow.log_params(config)
        
        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    predictions = model(X_batch)
                    
                    val_predictions.append(predictions.cpu().numpy())
                    val_targets.append(y_batch.numpy())
            
            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)
            
            val_metrics = calculate_metrics(val_targets, val_predictions)
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_mape': val_metrics['mape'],
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae']
            }, step=epoch)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val MAPE={val_metrics['mape']:.2f}%, "
                  f"Val RMSE={val_metrics['rmse']:.2f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['mape'])
            
            # Save best model
            if val_metrics['mape'] < best_val_mape:
                best_val_mape = val_metrics['mape']
                torch.save(model.state_dict(), 'models/best_transformer.pt')
                mlflow.pytorch.log_model(model, "model")
            
            # Early stopping
            early_stopping(val_metrics['mape'])
            if early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # For Optuna trials
            if trial is not None:
                trial.report(val_metrics['mape'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Test set evaluation
        model.load_state_dict(torch.load('models/best_transformer.pt'))
        model.eval()
        
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                predictions = model(X_batch)
                test_predictions.append(predictions.cpu().numpy())
                test_targets.append(y_batch.numpy())
        
        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)
        
        test_metrics = calculate_metrics(test_targets, test_predictions)
        
        mlflow.log_metrics({
            'test_mape': test_metrics['mape'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae']
        })
        
        print(f"\nFinal Test MAPE: {test_metrics['mape']:.2f}%")
        print(f"Final Test RMSE: {test_metrics['rmse']:.2f}")
        
        return test_metrics['mape']

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hypercast-transformer")
    
    config = {
        'batch_size': 64,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'lr': 0.0001,
        'weight_decay': 1e-5,
        'epochs': 100
    }
    
    final_mape = train_transformer(config)