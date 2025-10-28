import torch
import torch.nn as nn
from gwo_benchmark.base import GWOModule

class SelfAttentionGWO(GWOModule):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    @property
    def C_D(self) -> int:
        # GLOBAL_INDEXED (P) + FULL_ROW (S) + DYNAMIC_ATTENTION (W)
        return 1 + 1 + 2

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        return [self.mha]

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return attn_output