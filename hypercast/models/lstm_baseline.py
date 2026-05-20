# models/lstm_baseline.py
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, horizon=24):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, horizon)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predict next 24 hours
        predictions = self.fc(last_hidden)
        
        return predictions

# Test the model
if __name__ == "__main__":
    model = LSTMForecaster(input_size=10, hidden_size=128, horizon=24)
    
    # Dummy input: batch_size=32, sequence=168, features=10
    x = torch.randn(32, 168, 10)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 24)