from types import SimpleNamespace

import pytest

from forge.harness_spec import get_enc_in, get_feature_dim
from forge.model_io import ModelInterfaceError, read_model_source, validate_model_source
from forge.paths import INITIAL_MODEL_PATH


def test_initial_model_matches_harness_interface():
    cfg = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=get_enc_in(),
        hidden_dim=16,
        layer=1,
        dropout=0.1,
        feature_dim=get_feature_dim(),
    )
    validate_model_source(read_model_source(INITIAL_MODEL_PATH), cfg, feature_dim=get_feature_dim())


def test_model_validation_error_includes_shape_contract():
    cfg = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=get_enc_in(),
        hidden_dim=16,
        layer=1,
        dropout=0.1,
        feature_dim=get_feature_dim(),
    )
    bad_source = """
import torch
import torch.nn as nn

class ForgeModel(nn.Module):
    def __init__(self, configs):
        super().__init__()

    def forward(self, x):
        return x.view(10, 5, 12)
"""
    with pytest.raises(ModelInterfaceError) as exc_info:
        validate_model_source(bad_source, cfg, feature_dim=get_feature_dim())
    msg = str(exc_info.value)
    assert "input_shape" in msg
    assert "expected_output_shape" in msg
