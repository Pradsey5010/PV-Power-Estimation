"""
Temporal Encoder Module

LSTM and Transformer-based encoders for temporal sequence modeling
of weather data and image features over time.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """
    Learnable time embedding based on hour of day and day of year.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.hour_embed = nn.Embedding(24, embed_dim // 2)
        self.month_embed = nn.Embedding(12, embed_dim // 2)
        
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, hour: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hour: Hour of day [batch_size, seq_len]
            month: Month of year [batch_size, seq_len]
            
        Returns:
            Time embeddings [batch_size, seq_len, embed_dim]
        """
        hour_emb = self.hour_embed(hour)
        month_emb = self.month_embed(month)
        
        time_emb = torch.cat([hour_emb, month_emb], dim=-1)
        return self.projection(time_emb)


class LSTMEncoder(nn.Module):
    """
    LSTM-based temporal encoder for sequence modeling.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        output_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Attention for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            hidden: Optional initial hidden state
            return_sequence: Whether to return full sequence or aggregated
            
        Returns:
            output: Encoded features [batch_size, hidden_dim] or 
                   [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Project output
        lstm_out = self.output_proj(lstm_out)
        
        if return_sequence:
            return lstm_out
        
        # Attention-weighted aggregation
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        output = torch.sum(lstm_out * attn_weights, dim=1)
        
        return output


class GRUEncoder(nn.Module):
    """
    GRU-based temporal encoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        output, h_n = self.gru(x)
        output = self.output_proj(output[:, -1, :])
        return output


class TransformerEncoder(nn.Module):
    """
    Transformer-based temporal encoder for sequence modeling.
    
    Uses self-attention to capture long-range dependencies in
    weather and image feature sequences.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Model dimension (d_model)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Learnable CLS token for sequence aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
            return_sequence: Whether to return full sequence
            
        Returns:
            output: Encoded features [batch_size, hidden_dim] or
                   [batch_size, seq_len, hidden_dim]
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Update mask for CLS token
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Transformer forward
        output = self.transformer(x, src_key_padding_mask=mask)
        
        if return_sequence:
            return self.output_norm(output[:, 1:])  # Exclude CLS token
        
        # Return CLS token representation
        return self.output_norm(output[:, 0])


class TemporalEncoder(nn.Module):
    """
    Factory class for temporal encoders.
    
    Supports LSTM, GRU, and Transformer encoders.
    
    Args:
        encoder_type: Type of encoder ("lstm", "gru", "transformer")
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads (transformer only)
        dropout: Dropout rate
        bidirectional: Use bidirectional (LSTM/GRU only)
    """
    
    def __init__(
        self,
        encoder_type: str = "transformer",
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        
        encoder_type = encoder_type.lower()
        
        if encoder_type == "lstm":
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif encoder_type == "gru":
            self.encoder = GRUEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Choose from: lstm, gru, transformer"
            )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            return_sequence: Whether to return full sequence
            
        Returns:
            Encoded temporal features
        """
        if isinstance(self.encoder, TransformerEncoder):
            return self.encoder(x, mask=mask, return_sequence=return_sequence)
        else:
            return self.encoder(x, return_sequence=return_sequence)


if __name__ == "__main__":
    # Test temporal encoders
    batch_size = 4
    seq_len = 12
    input_dim = 128
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing Temporal Encoders...")
    
    for encoder_type in ["lstm", "gru", "transformer"]:
        print(f"\n{encoder_type.upper()} Encoder:")
        encoder = TemporalEncoder(
            encoder_type=encoder_type,
            input_dim=input_dim,
            hidden_dim=256
        )
        
        output = encoder(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        output_seq = encoder(x, return_sequence=True)
        print(f"  Sequence output shape: {output_seq.shape}")
