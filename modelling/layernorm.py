import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ 
    LayerNorm with an optional bias. PyTorch nn.LayerNorm doesn't support bias=False until recently: 
    AttributeError: 'NoneType' object has no attribute 'zero_' -> https://github.com/pytorch/pytorch/issues/108048
    For benefits of bias=False, see https://arxiv.org/abs/1911.07013
    """

    def __init__(self, d_model, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        nominator = input - input.mean(dim=-1, keepdim=True)
        denominator = torch.sqrt(input.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
        norm = self.weight * nominator / denominator + self.bias
        return norm