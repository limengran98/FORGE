"""Project-local FORGE template: multiscale_context_gru."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(getattr(configs, "enc_in", 5))
        self.feature_dim = int(getattr(configs, "feature_dim", self.enc_in))
        self.hidden_dim = int(getattr(configs, "hidden_dim", 256))
        self.layers = int(getattr(configs, "layer", 2))
        self.dropout_p = float(getattr(configs, "dropout", 0.1))
        self.factor_dim = max(self.feature_dim - self.enc_in, 0)

        # FORGE_COMPONENT: input_embedding
        self.voltage_embedding = nn.Linear(1, self.hidden_dim)
        if self.factor_dim > 0:
            self.factor_encoder = nn.Sequential(
                nn.Linear(self.factor_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.factor_encoder = None

        # FORGE_COMPONENT: factor_fusion
        self.context_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.context_mix = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # FORGE_COMPONENT: temporal_memory
        recurrent_dropout = self.dropout_p if self.layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.short_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.dilated_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.temporal_mix = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.temporal_norm = nn.LayerNorm(self.hidden_dim)

        # FORGE_COMPONENT: regularization
        self.dropout = nn.Dropout(self.dropout_p)

        # FORGE_COMPONENT: prediction_head
        self.residual_head = nn.Linear(self.hidden_dim, self.pred_len)
        self.trend_head = nn.Linear(self.hidden_dim, self.pred_len)
        self.level_gate = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        # FORGE_COMPONENT: normalization
        voltage = x[:, :, : self.enc_in]
        means = voltage.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(voltage, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        voltage_norm = (voltage - means) / stdev

        if self.factor_encoder is not None and feature_dim > self.enc_in:
            factors = x[:, :, self.enc_in : self.enc_in + self.factor_dim]
            factor_mean = factors.mean(dim=1, keepdim=True).detach()
            factor_std = torch.sqrt(torch.var(factors, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            factors = (factors - factor_mean) / factor_std
            factor_context = self.factor_encoder(factors)
        else:
            factor_context = x.new_zeros(batch_size, seq_len, self.hidden_dim)

        voltage_tokens = voltage_norm.reshape(batch_size * self.enc_in, seq_len, 1)
        voltage_tokens = self.voltage_embedding(voltage_tokens)
        context_tokens = (
            factor_context.unsqueeze(1)
            .expand(batch_size, self.enc_in, seq_len, self.hidden_dim)
            .reshape(batch_size * self.enc_in, seq_len, self.hidden_dim)
        )

        # FORGE_COMPONENT: factor_fusion
        gate = self.context_gate(torch.cat([voltage_tokens, context_tokens], dim=-1))
        fused = self.context_mix(torch.cat([voltage_tokens, gate * context_tokens], dim=-1))

        # FORGE_COMPONENT: temporal_memory
        output, _ = self.gru(fused)
        conv_input = output.transpose(1, 2)
        short = torch.tanh(self.short_conv(conv_input)).transpose(1, 2)
        dilated = torch.tanh(self.dilated_conv(conv_input)[:, :, : output.size(1)]).transpose(1, 2)
        last = torch.cat([output[:, -1, :], short[:, -1, :], dilated[:, -1, :]], dim=-1)
        state = self.temporal_norm(self.temporal_mix(last))
        state = self.dropout(F.gelu(state))

        # FORGE_COMPONENT: prediction_head
        residual = self.residual_head(state)
        trend_weight = torch.tanh(self.trend_head(state)) * 0.10
        residual = residual.reshape(batch_size, self.enc_in, self.pred_len).permute(0, 2, 1)
        trend_weight = trend_weight.reshape(batch_size, self.enc_in, self.pred_len).permute(0, 2, 1)

        trend = voltage_norm[:, -1:, :] - voltage_norm[:, :1, :]
        horizon = torch.linspace(
            1.0 / float(self.pred_len),
            1.0,
            self.pred_len,
            device=x.device,
            dtype=x.dtype,
        ).view(1, self.pred_len, 1)
        pred_norm = residual + trend_weight * horizon * trend
        pred_norm = pred_norm + torch.tanh(self.level_gate) * voltage_norm[:, -1:, :]

        return pred_norm * stdev + means
