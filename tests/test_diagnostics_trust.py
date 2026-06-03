import numpy as np

from forge.diagnostics import diagnose_result
from forge.graph import initial_task_graph
from forge.routing import route_feedback
from forge.trust import relation_id, update_relations_from_outcome


def test_diagnostics_detect_long_horizon_and_residual_autocorrelation(tmp_path):
    pred_path = tmp_path / "predictions.npz"
    y_true = np.zeros((4, 6, 2), dtype=np.float32)
    y_pred = np.arange(y_true.size, dtype=np.float32).reshape(y_true.shape) * 0.01
    np.savez_compressed(pred_path, y_pred_inverse=y_pred, y_true_inverse=y_true)
    result = {
        "success": True,
        "metrics": {
            "train": {"final_train_loss": 0.1, "final_val_loss": 0.2},
            "target": {"mae_inverse": 0.1},
        },
        "paths": {"predictions": str(pred_path)},
    }

    diagnostics = {item["name"]: item for item in diagnose_result(result)}
    assert diagnostics["long_horizon_error"]["severity"] > 0
    assert diagnostics["residual_autocorrelation"]["severity"] > 0


def test_trust_routing_records_feedback_component_propagation():
    state = initial_task_graph()
    feedback = {
        "features": {"run_success": 1.0},
        "diagnostics": [
            {
                "name": "long_horizon_error",
                "severity": 0.8,
                "confidence": 0.9,
                "evidence": {"long_horizon_mae": 2.0},
            }
        ],
    }
    route = route_feedback(feedback, state)
    relation_ids = {item["relation_id"] for item in route["propagations"]}
    assert relation_id("long_horizon_error", "temporal_memory") in relation_ids
    assert "temporal_memory" in route["active_components"]


def test_rule_routing_mode_disables_trust_propagation():
    feedback = {
        "features": {"run_success": 1.0},
        "diagnostics": [
            {
                "name": "long_horizon_error",
                "severity": 1.0,
                "confidence": 1.0,
                "evidence": {},
            }
        ],
    }
    route = route_feedback(feedback, mode="rule")
    assert route["propagations"] == []
    assert route["trust_policy"] == "rule_only"


def test_negative_executable_outcome_decreases_relation_trust():
    state = initial_task_graph()
    rid = relation_id("long_horizon_error", "temporal_memory")
    before = state["relations"][rid]["trust"]
    patch_record = {
        "component": "temporal_memory",
        "route_propagations": [
            {"diagnostic": "long_horizon_error", "component": "temporal_memory"},
        ],
    }
    previous_result = {
        "success": True,
        "metrics": {"target": {"mae_inverse": 1.0}},
        "paths": {"result": "prev.json"},
    }
    next_result = {
        "success": True,
        "metrics": {"target": {"mae_inverse": 1.2}},
        "paths": {"result": "next.json"},
    }
    previous_feedback = {
        "features": {"overfit_score": 0.1},
        "diagnostics": [{"name": "long_horizon_error", "severity": 0.7, "confidence": 1.0}],
    }
    next_feedback = {
        "features": {"overfit_score": 0.4},
        "diagnostics": [{"name": "long_horizon_error", "severity": 0.8, "confidence": 1.0}],
    }

    updates = update_relations_from_outcome(
        state,
        patch_record,
        previous_result,
        next_result,
        previous_feedback,
        next_feedback,
        "mae_inverse",
    )
    assert updates[0]["direction"] == "decrease"
    assert state["relations"][rid]["trust"] < before
