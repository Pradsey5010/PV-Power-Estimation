"""
Feature Fusion Module

Methods for combining image features, temporal features, and weather data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion.
    
    Concatenates all input features and projects to output dimension.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        output_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.2
    ):
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
        """
        Args:
            features: List of feature tensors to fuse
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        concat = torch.cat(features, dim=-1)
        return self.fusion(concat)


class AttentionFusion(nn.Module):
    """
    Cross-attention based fusion for combining multiple modalities.
    
    Uses attention mechanism to dynamically weight the importance
    of each modality based on the context.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_modalities = len(input_dims)
        self.output_dim = output_dim
        
        # Project each modality to same dimension
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            for dim in input_dims
        ])
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Query token (learnable)
        self.query_token = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(len(input_dims), output_dim)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors [batch_size, dim_i]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        batch_size = features[0].size(0)
        
        # Project each modality
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.modality_projections)):
            proj_feat = proj(feat).unsqueeze(1)  # [batch, 1, output_dim]
            # Add modality embedding
            mod_emb = self.modality_embeddings(
                torch.tensor([i], device=feat.device)
            ).unsqueeze(0)
            proj_feat = proj_feat + mod_emb
            projected.append(proj_feat)
        
        # Stack modalities as key/value
        kv = torch.cat(projected, dim=1)  # [batch, num_modalities, output_dim]
        
        # Query
        query = self.query_token.expand(batch_size, -1, -1)
        
        # Cross attention
        attn_out, _ = self.cross_attention(query, kv, kv)
        attn_out = self.norm1(query + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)
        
        return out.squeeze(1)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for modality combination.
    
    Uses learned gates to control information flow from each modality.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        output_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_modalities = len(input_dims)
        self.output_dim = output_dim
        
        # Project each modality
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            for dim in input_dims
        ])
        
        # Gate networks for each modality
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
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Concatenate for gate computation
        concat = torch.cat(features, dim=-1)
        
        # Project and gate each modality
        gated_features = []
        for i, (feat, proj, gate) in enumerate(
            zip(features, self.projections, self.gates)
        ):
            proj_feat = proj(feat)
            gate_values = gate(concat)
            gated_feat = proj_feat * gate_values
            gated_features.append(gated_feat)
        
        # Sum gated features
        fused = torch.stack(gated_features, dim=0).sum(dim=0)
        
        return self.output_proj(fused)


class BilinearFusion(nn.Module):
    """
    Bilinear fusion for two modalities.
    
    Captures multiplicative interactions between features.
    """
    
    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int = 512,
        rank: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Low-rank bilinear pooling
        self.proj1 = nn.Linear(dim1, rank)
        self.proj2 = nn.Linear(dim2, rank)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(rank, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat1: First modality features [batch_size, dim1]
            feat2: Second modality features [batch_size, dim2]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        proj1 = self.proj1(feat1)
        proj2 = self.proj2(feat2)
        
        # Element-wise product
        bilinear = proj1 * proj2
        
        return self.output_proj(bilinear)


class FusionLayer(nn.Module):
    """
    Factory class for fusion methods.
    
    Supports concatenation, attention, gated, and bilinear fusion.
    
    Args:
        method: Fusion method ("concat", "attention", "gated", "bilinear")
        input_dims: List of input feature dimensions
        output_dim: Output feature dimension
        hidden_dim: Hidden dimension for fusion network
        dropout: Dropout rate
        num_heads: Number of attention heads (attention only)
    """
    
    def __init__(
        self,
        method: str = "attention",
        input_dims: List[int] = [512, 256, 64],
        output_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        num_heads: int = 8
    ):
        super().__init__()
        
        method = method.lower()
        
        if method == "concat":
            self.fusion = ConcatFusion(
                input_dims=input_dims,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif method == "attention":
            self.fusion = AttentionFusion(
                input_dims=input_dims,
                output_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif method == "gated":
            self.fusion = GatedFusion(
                input_dims=input_dims,
                output_dim=output_dim,
                dropout=dropout
            )
        elif method == "bilinear":
            if len(input_dims) != 2:
                raise ValueError("Bilinear fusion requires exactly 2 input dimensions")
            self.fusion = BilinearFusion(
                dim1=input_dims[0],
                dim2=input_dims[1],
                output_dim=output_dim,
                dropout=dropout
            )
        else:
            raise ValueError(
                f"Unknown fusion method: {method}. "
                f"Choose from: concat, attention, gated, bilinear"
            )
        
        self.output_dim = output_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors to fuse
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        if isinstance(self.fusion, BilinearFusion):
            return self.fusion(features[0], features[1])
        return self.fusion(features)


if __name__ == "__main__":
    # Test fusion modules
    batch_size = 4
    dims = [512, 256, 64]
    
    features = [torch.randn(batch_size, dim) for dim in dims]
    
    print("Testing Fusion Methods...")
    
    for method in ["concat", "attention", "gated"]:
        print(f"\n{method.upper()} Fusion:")
        fusion = FusionLayer(method=method, input_dims=dims)
        output = fusion(features)
        print(f"  Input dims: {dims}")
        print(f"  Output shape: {output.shape}")
    
    # Test bilinear
    print("\nBILINEAR Fusion:")
    bilinear = FusionLayer(method="bilinear", input_dims=[512, 256])
    output = bilinear([features[0], features[1]])
    print(f"  Input dims: [512, 256]")
    print(f"  Output shape: {output.shape}")
