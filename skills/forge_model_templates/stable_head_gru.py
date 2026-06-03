"""Project-local FORGE template: stable_head_gru."""

import torch
import torch.nn as nn


class ForgeModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(getattr(configs, "enc_in", 5))
        self.hidden_dim = int(getattr(configs, "hidden_dim", 256))
        self.layers = int(getattr(configs, "layer", 2))
        self.dropout_p = float(getattr(configs, "dropout", 0.1))

        # FORGE_COMPONENT: input_embedding
        self.embedding = nn.Linear(1, self.hidden_dim)

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
        self.level_head = nn.Linear(self.hidden_dim, self.pred_len)
        self.delta_head = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x):
        B, L, _ = x.shape

        # FORGE_COMPONENT: normalization
        voltage = x[:, :, : self.enc_in]
        means = voltage.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(voltage, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        voltage_norm = (voltage - means) / stdev

        tokens = voltage_norm.reshape(B * self.enc_in, L, 1)
        tokens = self.embedding(tokens)
        output, _ = self.gru(tokens)
        last_state = self.dropout(self.norm(output[:, -1, :]))

        level = self.level_head(last_state)
        delta = 0.1 * torch.tanh(self.delta_head(last_state))
        pred = level + delta
        pred = pred.reshape(B, self.enc_in, self.pred_len).permute(0, 2, 1)

        last_observed = voltage_norm[:, -1:, :]
        pred = 0.85 * pred + 0.15 * last_observed
        return pred * stdev + means

