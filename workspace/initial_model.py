import torch
import torch.nn as nn


class ForgeModel(nn.Module):
    """Initial GRU baseline following the Ms-AeDNet GRU interface."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(getattr(configs, "enc_in", 5))
        self.hidden_dim = int(getattr(configs, "hidden_dim", 256))
        self.layers = int(getattr(configs, "layer", 2))

        # FORGE_COMPONENT: input_embedding
        self.embedding = nn.Linear(1, self.hidden_dim)

        # FORGE_COMPONENT: temporal_memory
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            batch_first=True,
        )

        # FORGE_COMPONENT: prediction_head
        self.fc_out = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x):
        B, L, _ = x.shape

        # FORGE_COMPONENT: normalization
        voltage = x[:, :, : self.enc_in]
        means = voltage.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(voltage, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        voltage = (voltage - means) / stdev

        # FORGE_COMPONENT: backbone
        voltage = voltage.reshape(B * self.enc_in, L, 1)
        voltage = self.embedding(voltage)
        output, _ = self.gru(voltage)

        # FORGE_COMPONENT: prediction_head
        output = self.fc_out(output[:, -1, :])
        output = output.reshape(B, self.enc_in, self.pred_len).permute(0, 2, 1)

        return output * stdev + means

