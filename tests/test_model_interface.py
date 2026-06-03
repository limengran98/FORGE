from types import SimpleNamespace

from forge.model_io import read_model_source, validate_model_source
from forge.paths import INITIAL_MODEL_PATH


def test_initial_model_matches_harness_interface():
    cfg = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=5,
        hidden_dim=16,
        layer=1,
        dropout=0.1,
        feature_dim=23,
    )
    validate_model_source(read_model_source(INITIAL_MODEL_PATH), cfg, feature_dim=23)

