"""
Feature Fusion Module - Methods for combining multi-modal features.
"""

import torch
import torch.nn as nn
from typing import List


class ConcatFusion(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int = 512,
                 hidden_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        
        total_dim = sum(input_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(features, dim=-1)
        return self.fusion(concat)


class AttentionFusion(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        
        self.num_modalities = len(input_dims)
        self.output_dim = output_dim
        
        self.modality_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim))
            for dim in input_dims
        ])
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.query_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.modality_embeddings = nn.Embedding(len(input_dims), output_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        batch_size = features[0].size(0)
        
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.modality_projections)):
            proj_feat = proj(feat).unsqueeze(1)
            mod_emb = self.modality_embeddings(torch.tensor([i], device=feat.device)).unsqueeze(0)
            proj_feat = proj_feat + mod_emb
            projected.append(proj_feat)
        
        kv = torch.cat(projected, dim=1)
        query = self.query_token.expand(batch_size, -1, -1)
        
        attn_out, _ = self.cross_attention(query, kv, kv)
        attn_out = self.norm1(query + attn_out)
        
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)
        
        return out.squeeze(1)


class GatedFusion(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())
            for dim in input_dims
        ])
        
        total_dim = sum(input_dims)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, output_dim),
                nn.Sigmoid()
            )
            for _ in input_dims
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(features, dim=-1)
        
        gated_features = []
        for feat, proj, gate in zip(features, self.projections, self.gates):
            proj_feat = proj(feat)
            gate_values = gate(concat)
            gated_feat = proj_feat * gate_values
            gated_features.append(gated_feat)
        
        fused = torch.stack(gated_features, dim=0).sum(dim=0)
        return self.output_proj(fused)


class FusionLayer(nn.Module):
    def __init__(self, method: str = "attention", input_dims: List[int] = [512, 256, 64],
                 output_dim: int = 512, hidden_dim: int = 1024, dropout: float = 0.2,
                 num_heads: int = 8):
        super().__init__()
        
        if method == "concat":
            self.fusion = ConcatFusion(input_dims, output_dim, hidden_dim, dropout)
        elif method == "attention":
            self.fusion = AttentionFusion(input_dims, output_dim, num_heads, dropout)
        elif method == "gated":
            self.fusion = GatedFusion(input_dims, output_dim, dropout)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        self.output_dim = output_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return self.fusion(features)
