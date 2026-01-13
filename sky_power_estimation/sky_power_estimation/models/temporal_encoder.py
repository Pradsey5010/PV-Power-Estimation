"""
Temporal Encoder Module - LSTM and Transformer encoders.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        output_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.output_proj(lstm_out)
        
        if return_sequence:
            return lstm_out
        
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        output = torch.sum(lstm_out * attn_weights, dim=1)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3,
                 num_heads: int = 8, dropout: float = 0.1, max_seq_len: int = 100):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_sequence: bool = False):
        batch_size = x.size(0)
        
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        
        output = self.transformer(x, src_key_padding_mask=mask)
        
        if return_sequence:
            return self.output_norm(output[:, 1:])
        return self.output_norm(output[:, 0])


class TemporalEncoder(nn.Module):
    def __init__(self, encoder_type: str = "transformer", input_dim: int = 512,
                 hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 8,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        
        if encoder_type == "lstm":
            self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout, bidirectional)
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_sequence: bool = False):
        if isinstance(self.encoder, TransformerEncoder):
            return self.encoder(x, mask=mask, return_sequence=return_sequence)
        return self.encoder(x, return_sequence=return_sequence)
