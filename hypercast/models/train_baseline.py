# scripts/train_baseline.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from models.lstm_baseline import LSTMForecaster

def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_model(config):
    # Load data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # Train/val split
    split_idx = int(0.8 * len(X_train))
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = LSTMForecaster(
        input_size=X_train.shape[2],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        horizon=y_train.shape[1]
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Start MLFlow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(config)
        
        # Training loop
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_mape = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    predictions = model(X_batch)
                    val_loss += criterion(predictions, y_batch).item()
                    
                    # Calculate MAPE
                    mape = calculate_mape(
                        y_batch.numpy(), 
                        predictions.numpy()
                    )
                    val_mape += mape
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_mape /= len(val_loader)
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mape': val_mape
            }, step=epoch)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val MAPE={val_mape:.2f}%")
        
        # Save model
        mlflow.pytorch.log_model(model, "model")
        
        return model, val_mape

if __name__ == "__main__":
    # Set MLFlow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hypercast-baseline")
    
    config = {
        'batch_size': 32,
        'hidden_size': 128,
        'num_layers': 2,
        'lr': 0.001,
        'epochs': 50
    }
    
    model, final_mape = train_model(config)
    print(f"Final validation MAPE: {final_mape:.2f}%")