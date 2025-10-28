import torch
import torch.nn as nn
from gwo_benchmark.base import GWOModule

class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MlpMixerBlockGWO(GWOModule):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.token_mix = MlpBlock(num_patches, num_patches * 2)

    @property
    def C_D(self) -> int:
        # GLOBAL_INDEXED (P) + FULL_ROW (S) + SHARED_KERNEL (W)
        return 1 + 1 + 1

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        return []

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)
        return x + y