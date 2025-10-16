"""
Main QuETT (Quadratic Enhanced Topology Transformer) model implementation (ASPP-only).
"""
import torch
import torch.nn as nn
from .quadratic_components import ASPPQuadraticAdapter
from .cnn_components import SimpleCNNCorrExtractor
from .cluster_pooling import DEC


class QuETT(nn.Module):
    """QuETT (ASPP-only): CNNCorr (optional) + stack of ASPPQuadraticAdapter + pooling/readout."""
    def __init__(
        self,
        args,
        node_sz,
        corr_pearson_sz,
        layers,
        dropout=0.,
        cluster_num=4,
        pooling=True,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True,
        droppath=0.1,
        rank=8,
        num_classes=2,  # Add num_classes parameter
        # CNNCorr / ASPP preprocessor
        use_cnncorr=True, cnncorr_base_ch=16, cnncorr_diag_mode="zero",
        cnncorr_symmetrize=True, cnncorr_norm="gn", cnncorr_dropout=0.0,
        cnncorr_aspp_rates=(1, 2, 4), use_enhanced_aspp=False,
        # Quadratic (ASPP) adapter
        quadratic_aspp_rates=(1, 2, 4),
        # Norms
        use_layer_norm=True, ln_eps=1e-6,
    ):
        super().__init__()
        d = int(corr_pearson_sz)
        self.pooling = bool(pooling)
        self.use_cnncorr = bool(use_cnncorr)
        self.use_layer_norm = bool(use_layer_norm)
        self.use_enhanced_aspp = bool(use_enhanced_aspp)

        # === Adapter stack: ASPP-only ===
        self.adapters = nn.ModuleList([
            ASPPQuadraticAdapter(
                node_sz,  # Use node_sz for correlation matrix dimension
                activation=args.activation,
                dropout=dropout,
                droppath=droppath,
                rank=rank,
                aspp_rates=quadratic_aspp_rates,
            )
            for _ in range(int(layers))
        ])

        # LayerNorms
        if self.use_layer_norm:
            self.ln_pre = nn.LayerNorm(node_sz, eps=ln_eps)
            self.ln_post = nn.LayerNorm(node_sz, eps=ln_eps)

        # === CNNCorr frontend (optional) ===
        if self.use_cnncorr:
            self.cnncorr = SimpleCNNCorrExtractor(
                in_ch=1,
                base_ch=cnncorr_base_ch,
                out_dim=d,
                diag_mode=cnncorr_diag_mode,
                symmetrize=cnncorr_symmetrize,
                norm=cnncorr_norm,
                dropout=cnncorr_dropout,
                aspp_rates=cnncorr_aspp_rates,
            )

            # Enhanced ASPP with extra 1x1s (optional)
            self.aspp_processor = self._create_enhanced_aspp_processor(d, cnncorr_aspp_rates)

            # Gate for mixing CNNCorr+ASPP vs 1x1 path
            self.beta_cnn = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

            if self.use_enhanced_aspp:
                self.aspp_cnn_gate = nn.Parameter(torch.tensor(0.5))
                self.aspp_final_proj = nn.Conv2d(self.aspp_processor['aspp_1x1_2'].out_channels, 1, 1, bias=True)
                with torch.no_grad():
                    self.aspp_final_proj.weight.fill_(1.0 / self.aspp_processor['aspp_1x1_2'].out_channels)
                    self.aspp_final_proj.bias.zero_()

            # Simple 1x1 path + affine scale/shift
            self.opt_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            nn.init.constant_(self.opt_conv.weight, 1.0)
            nn.init.constant_(self.opt_conv.bias, 0.0)

            self.scale_shift = nn.Linear(1, 2, bias=False)
            with torch.no_grad():
                self.scale_shift.weight.copy_(torch.tensor([[1.0], [0.0]]))  # scale=1, shift=0

        # === Readout / Pooling ===
        if self.pooling:
            enc_h = 32
            encoder_input_size = d * d
            self.encoder = nn.Sequential(
                nn.Linear(encoder_input_size, enc_h), nn.LeakyReLU(),
                nn.Linear(enc_h, enc_h), nn.LeakyReLU(),
                nn.Linear(enc_h, encoder_input_size),
            )
            self.dec = DEC(
                cluster_number=cluster_num,
                hidden_dimension=d,
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment,
            )
            self.dim_reduction = nn.Sequential(nn.Linear(d, 8), nn.LeakyReLU())
            self.fc = nn.Sequential(
                nn.Linear(8 * cluster_num, 256), nn.LeakyReLU(),
                nn.Linear(256, 32), nn.LeakyReLU(),
                nn.Linear(32, num_classes),
            )
        else:
            self.readout = nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(), nn.Linear(64, num_classes))


    @torch.no_grad()
    def load_base_from(self, base_state_dict, strict=False, prefix_map=None):
        """Copy base Linear (W_C, b) into each ASPPQuadraticAdapter."""
        missing = []
        for i, adapter in enumerate(self.adapters):
            w_name = f"adapters.{i}.base.weight"
            b_name = f"adapters.{i}.base.bias"
            src_w_key = w_name if prefix_map is None else prefix_map.get(w_name, w_name)
            src_b_key = b_name if prefix_map is None else prefix_map.get(b_name, b_name)
            src_w = base_state_dict.get(src_w_key, None)
            src_b = base_state_dict.get(src_b_key, None)
            if src_w is not None and src_b is not None:
                adapter.base.weight.copy_(src_w)
                adapter.base.bias.copy_(src_b)
            elif strict:
                if src_w is None: missing.append(src_w_key)
                if src_b is None: missing.append(src_b_key)
        if strict and missing:
            raise RuntimeError(f"Missing base keys: {missing}")

    # ===== CNNCorr helpers =====
    def _cnncorr_nodes(self, corr_mat):
        corr_4d = corr_mat.unsqueeze(1)
        return self.cnncorr(corr_4d)

    def _create_enhanced_aspp_processor(self, d, aspp_rates):
        from .basic_blocks import ASPP2d
        from .normalization import _make_norm
        return nn.ModuleDict({
            'aspp': ASPP2d(1, d // 4, rates=aspp_rates, norm='gn', act=nn.GELU),
            'aspp_1x1_1': nn.Conv2d(d // 4, d // 2, 1, bias=False),
            'aspp_1x1_2': nn.Conv2d(d // 2, d, 1, bias=False),
            'aspp_1x1_3': nn.Conv2d(d, d, 1, bias=False),
            'bn1': _make_norm('gn', d // 2),
            'bn2': _make_norm('gn', d),
            'bn3': _make_norm('gn', d),
            'act': nn.GELU(),
            'dropout': nn.Dropout2d(0.1),
        })

    def _optimized_1x1_cnn(self, corr_mat):
        bz, n, _ = corr_mat.shape
        corr_4d = corr_mat.unsqueeze(1)
        x_cnn = self.opt_conv(corr_4d).squeeze(1)  # [B, N, N]
        ones = torch.ones(bz, n, n, 1, device=corr_mat.device, dtype=corr_mat.dtype)
        scale_shift = self.scale_shift(ones)  # [B, N, N, 2]
        scale = scale_shift[..., 0].squeeze(-1)
        shift = scale_shift[..., 1].squeeze(-1)
        x_cnn = scale * x_cnn + shift
        x_cnn = x_cnn + corr_mat
        x_cnn = 0.5 * (x_cnn + x_cnn.transpose(-1, -2))
        x_cnn = x_cnn - torch.diag_embed(torch.diagonal(x_cnn, dim1=-2, dim2=-1))
        return x_cnn

    def _enhanced_aspp_processing(self, corr_mat):
        corr_4d = corr_mat.unsqueeze(1)  # [B, 1, N, N]
        aspp_out = self.aspp_processor['aspp'](corr_4d)        # [B, d//4, N, N]
        x = self.aspp_processor['aspp_1x1_1'](aspp_out)
        x = self.aspp_processor['bn1'](x)
        x = self.aspp_processor['act'](x)
        x = self.aspp_processor['aspp_1x1_2'](x)               # [B, d, N, N]
        x = self.aspp_processor['bn2'](x)
        x = self.aspp_processor['act'](x)
        x = self.aspp_processor['aspp_1x1_3'](x)
        x = self.aspp_processor['bn3'](x)
        x = self.aspp_processor['dropout'](x)
        x_corr = self.aspp_final_proj(x).squeeze(1)            # [B, N, N]
        x_corr = x_corr + corr_mat
        x_corr = 0.5 * (x_corr + x_corr.transpose(-1, -2))
        x_corr = x_corr - torch.diag_embed(torch.diagonal(x_corr, dim1=-2, dim2=-1))
        return x_corr

    def reset_learnable_gates(self):
        """Reset gates for reproducibility."""
        if hasattr(self, 'beta_cnn'):
            self.beta_cnn.data.fill_(-2.0)
        if hasattr(self, 'aspp_cnn_gate'):
            self.aspp_cnn_gate.data.fill_(0.5)
        for a in self.adapters:
            if hasattr(a, 'gate'):
                nn.init.constant_(a.gate.bias, 0.0)
            if hasattr(a, 'gate_temp'):
                a.gate_temp.data.fill_(1.0)
            if hasattr(a, 'alpha'):
                a.alpha.data.zero_()

    def forward(self, corr: torch.Tensor, corr_mat: torch.Tensor = None):
        """CNNCorr (optional) → ASPPQuadraticAdapter stack → pooling/readout."""
        B, N, M = corr.shape
        assert N == M, "corr must be [B, N, N]"
        if corr_mat is not None:
            b2, n2, m2 = corr_mat.shape
            assert (b2, n2, m2) == (B, N, M), "corr_mat must match corr shape"

        x = corr
        if self.use_layer_norm:
            x = self.ln_pre(x)
        corr_in = self.ln_pre(corr) if self.use_layer_norm else corr

        # CNNCorr front-end
        if self.use_cnncorr and (corr_mat is not None):
            if self.use_enhanced_aspp:
                x_aspp = self._enhanced_aspp_processing(corr_mat)
                x_cnn = self._optimized_1x1_cnn(corr_mat)
                gate = torch.sigmoid(self.aspp_cnn_gate)
                x = gate * x_aspp + (1 - gate) * x_cnn
            else:
                x = self._optimized_1x1_cnn(corr_mat)

        # ASPP adapters
        for adapter in self.adapters:
            x = adapter(x, corr_in)
            x = 0.5 * (x + x.transpose(-1, -2))
            x = x - torch.diag_embed(torch.diagonal(x, dim1=-2, dim2=-1))


        # Fuse with original corr (optional, gated)
        if self.use_cnncorr and (corr_mat is not None):
            x_std = torch.std(x, dim=(-2, -1), keepdim=True) + 1e-8
            corr_std = torch.std(corr_in, dim=(-2, -1), keepdim=True) + 1e-8
            x_scaled = x / x_std
            corr_scaled = corr_in / corr_std
            gate = torch.sigmoid(self.beta_cnn)
            x = gate * x_scaled + (1 - gate) * corr_scaled
            x = x * x_std

        # Readout
        if self.pooling:
            if self.use_layer_norm:
                x = self.ln_post(x)
            x = x.contiguous()
            graph_level_topo, _ = self.dec(x)
            graph_level_topo = self.dim_reduction(graph_level_topo)
            graph_level_topo = graph_level_topo.reshape(B, -1)
            return self.fc(graph_level_topo)
        else:
            x_mean = x.mean(dim=1)
            return self.readout(x_mean)