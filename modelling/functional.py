import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

from modelling import PointWiseFeedForward, MultiHeadAttention, LayerNorm

        
class BaseTransformerLayer(nn.Module):

    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(input_dim, num_heads, dropout)
        self.feature_transformation = PointWiseFeedForward(input_dim, feature_dim, dropout)
        self.layer_norm_1 = LayerNorm(input_dim, bias=True)
        self.layer_norm_2 = LayerNorm(input_dim, bias=True)

    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        attention_output = self.self_attention(input, input, input, attention_mask)
        attention_output = self.layer_norm_1(attention_output + input)
        feature_output = self.feature_transformation(attention_output)
        return self.layer_norm_2(feature_output + attention_output)
    
    
class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True, dropout=dropout)
        self.layer_norm_1 = LayerNorm(input_dim, bias=True)

        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, dropout)
        self.layer_norm_2 = LayerNorm(input_dim, bias=True)

        self.feature_transformation = PointWiseFeedForward(input_dim, feature_dim, dropout)
        self.layer_norm_3 = LayerNorm(input_dim, bias=True)

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
                
        self_attention_output = self.self_attention(input, input, input, encoder_attention_mask)
        self_attention_output = self.layer_norm_1(self_attention_output + input)

        encoder_attention_output = self.encoder_attention(self_attention_output, encoder_output, encoder_output, attention_mask) # Instead of applying the mask here
        encoder_attention_output = self.layer_norm_2(encoder_attention_output + self_attention_output)

        feature_output = self.feature_transformation(encoder_attention_output)
        out = self.layer_norm_3(feature_output + encoder_attention_output)

        if encoder_attention_mask is not None:
            out = out.masked_fill(encoder_attention_mask.unsqueeze(-1) == 0, 0)

        return out
