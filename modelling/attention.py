import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import einops

class Attention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_future=False, dropout=0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mask_future = mask_future

        ''' 
        For multi-head attention, we need to project the input queries, keys, and values into multiple heads.
        For computational efficiency, we can do this with a single linear projection for each of the inputs.
        So before linear transformation, heads will be concatenated and after linear transformation, heads will
        be split into multiple heads.
        '''
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        q = einops.rearrange(self.query_transform(q), 'b n (h d) -> b h n d', h=self.n_heads)
        k = einops.rearrange(self.key_transform(k), 'b n (h d) -> b h n d', h=self.n_heads)
        v = einops.rearrange(self.value_transform(v), 'b n (h d) -> b h n d', h=self.n_heads)

        out = self.multihead_attention(q, k, v, attn_mask)

        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        return self.resid_drop(self.output_transform(out))


    def multihead_attention(self, q, k, v, mask):
        """Handles multi-head scaled dot-product attention."""
        d_k = k.size(-1)
        scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=k.device))

        # Compute scaled dot-product scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / scale

        # Apply future mask if enabled
        if self.mask_future:
            scores = self.apply_future_mask(scores, multihead=True)

        # Apply custom attention mask
        if mask is not None:
            scores = self.apply_custom_mask(scores, mask, multihead=True)

        # Compute attention weights and output
        attn = F.softmax(scores, dim=-1)
        output = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        return output


    def singlehead_attention(self, q, k, v, mask):
        """Handles single-head scaled dot-product attention."""
        d_k = k.size(-1)
        scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=k.device))

        # Compute scaled dot-product scores
        scores = torch.einsum("bqd,bkd->bqk", q, k) / scale

        # Apply future mask if enabled
        if self.mask_future:
            scores = self.apply_future_mask(scores, multihead=False)

        # Apply custom attention mask
        if mask is not None:
            scores = self.apply_custom_mask(scores, mask, multihead=False)

        # Compute attention weights and output
        attn = F.softmax(scores, dim=-1)
        output = torch.einsum("bqk,bkd->bqd", attn, v)
        return output


    def apply_future_mask(self, scores, multihead):
        """Applies future masking to the scores."""
        future_mask = torch.triu(
            torch.ones(scores.size(-2), scores.size(-1), device=scores.device),
            diagonal=1,
        ).bool()
        if multihead:
            future_mask = future_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        else:
            future_mask = future_mask.unsqueeze(0)  # (1, L, L)
        return scores.masked_fill(future_mask, float("-inf"))


    def apply_custom_mask(self, scores, mask, multihead):
        """Applies a custom mask to the scores."""

        # Adjust mask dimensions for multihead
        if mask.dim() == 2:  # (B, L_k)
            mask = mask.unsqueeze(1)  # (B, 1, L_k)
            if multihead:
                mask = mask.unsqueeze(1)  # (B, 1, 1, L_k)
        elif mask.dim() == 3:  # (B, L_q, L_k)
            if multihead:
                mask = mask.unsqueeze(1)  # (B, 1, L_q, L_k)

        return scores.masked_fill(mask == 0, float("-inf"))
