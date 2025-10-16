
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm, DropPath
from .basic_blocks import ASPP2d


class ASPPQuadraticAdapter(nn.Module):
    """
    Quadratic adapter with ASPP for multi-scale processing of the correlation matrix.

    Forward:
        h = RMSNorm(x)
        base = Linear(h)
        C' = ASPP(C)    # [B,1,N,N] -> [B,1,N,N]
        s = C' @ h      # [B,N,N] @ [B,N,d] -> [B,N,d]
        quad = Linear(h * s)    # or Ux/Vc/Wo if rank>0 (kept simple: rank=0 default)
        gate = sigmoid(Linear(h) / temp)
        y = gate * (base + alpha * quad)
        out = x + DropPath( Dropout( act(y) ) )
    """
    def __init__(self, d, activation='gelu', dropout=0., droppath=0., rank=0, aspp_rates=(1, 2, 4)):
        super().__init__()
        self.norm = RMSNorm(d)
        self.base = nn.Linear(d, d, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(dropout)
        self.dp = DropPath(droppath)

        self.gate = nn.Linear(d, d)
        nn.init.constant_(self.gate.bias, 0.0)
        self.gate_temp = nn.Parameter(torch.tensor(1.0))

        act_map = {'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}
        self.act = act_map.get(str(activation).lower(), nn.GELU)()

        # ASPP for multi-scale filtering of corr
        self.aspp = ASPP2d(in_ch=1, out_ch=1, rates=aspp_rates, norm="gn", act=nn.GELU)

        # Quadratic term (simple full projection by default)
        self.rank = int(rank) if rank is not None else 0
        if self.rank > 0:
            # Keep an optional low-rank head (no masks/atrous)
            self.Ux = nn.Linear(d, self.rank, bias=False)
            self.Vc = nn.Linear(d, self.rank, bias=False)
            self.Wo = nn.Linear(self.rank, d, bias=False)
        else:
            self.quad = nn.Linear(d, d, bias=True)
            nn.init.normal_(self.quad.weight, 0, 1.0 / math.sqrt(d))
            nn.init.zeros_(self.quad.bias)

    def forward(self, x, corr):
        """
        x:    [B, N, d]
        corr: [B, N, N]  (square)
        """
        h = self.norm(x)
        base = self.base(h)

        # ASPP over corr
        corr_aspp = self.aspp(corr.unsqueeze(1)).squeeze(1)  # [B, N, N]

        # Spatial mixing then quadratic
        s = torch.einsum('bni,bid->bnd', corr_aspp, h)       # [B, N, d]

        if self.rank > 0:
            ux = self.Ux(h)                                   # [B, N, r]
            vs = self.Vc(s)                                   # [B, N, r]
            quad = self.Wo(ux * vs)                           # [B, N, d]
        else:
            quad = self.quad(h * s)                           # [B, N, d]

        gate = torch.sigmoid(self.gate(h) / self.gate_temp.clamp_min(0.1))
        y = gate * (base + self.alpha * quad)

        y = self.drop(self.act(y))
        return x + self.dp(y)