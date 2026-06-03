"""Project-local FORGE template: context_fusion_gru."""

import torch
import torch.nn as nn


class ForgeModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(getattr(configs, "enc_in", 5))
        self.feature_dim = int(getattr(configs, "feature_dim", self.enc_in))
        self.hidden_dim = int(getattr(configs, "hidden_dim", 256))
        self.layers = int(getattr(configs, "layer", 2))
        self.dropout_p = float(getattr(configs, "dropout", 0.1))
        factor_dim = max(self.feature_dim - self.enc_in, 1)

        # FORGE_COMPONENT: input_embedding
        self.voltage_embedding = nn.Linear(1, self.hidden_dim)
        self.factor_encoder = nn.Sequential(
            nn.Linear(factor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # FORGE_COMPONENT: factor_fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid(),
        )

        # FORGE_COMPONENT: temporal_memory
        recurrent_dropout = self.dropout_p if self.layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )

        # FORGE_COMPONENT: regularization
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        # FORGE_COMPONENT: prediction_head
        self.fc_out = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x):
        B, L, M = x.shape

        # FORGE_COMPONENT: normalization
        voltage = x[:, :, : self.enc_in]
        means = voltage.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(voltage, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        voltage_norm = (voltage - means) / stdev

        if M > self.enc_in:
            factors = x[:, :, self.enc_in :]
            factor_context = self.factor_encoder(factors)
        else:
            factor_context = x.new_zeros(B, L, self.hidden_dim)

        voltage_tokens = voltage_norm.reshape(B * self.enc_in, L, 1)
        voltage_tokens = self.voltage_embedding(voltage_tokens)
        context_tokens = (
            factor_context.unsqueeze(1)
            .expand(B, self.enc_in, L, self.hidden_dim)
            .reshape(B * self.enc_in, L, self.hidden_dim)
        )
        gate = self.fusion_gate(torch.cat([voltage_tokens, context_tokens], dim=-1))
        fused = voltage_tokens + gate * context_tokens

        output, _ = self.gru(fused)
        last = self.norm(output[:, -1, :])
        pred = self.fc_out(self.dropout(last))
        pred = pred.reshape(B, self.enc_in, self.pred_len).permute(0, 2, 1)

        return pred * stdev + means

