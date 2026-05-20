# models/transformer_forecaster.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 input_size,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=3,
                 dim_feedforward=512,
                 dropout=0.1,
                 horizon=24):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, horizon)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Use last time step for prediction
        last_hidden = encoded[:, -1, :]  # (batch, d_model)
        
        # Project to horizon
        output = self.output_projection(last_hidden)  # (batch, horizon)
        
        return output

# Test the model
if __name__ == "__main__":
    model = TimeSeriesTransformer(
        input_size=10,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        horizon=24
    )
    
    x = torch.randn(32, 168, 10)  # batch=32, seq=168, features=10
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 24)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")