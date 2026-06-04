import numpy as np

from forge.diagnostics import diagnose_result
from forge.graph import initial_task_graph
from forge.memory import action_relation_id, update_action_memory_from_outcome
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
    assert route["selected_edit"]["component"] == "temporal_memory"
    assert route["selected_edit"]["edit_operator"] == "add_temporal_smoothing"
    assert route["edit_candidates"]


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


def test_validation_fallback_does_not_reward_noop_metrics():
    state = initial_task_graph()
    rid = relation_id("long_horizon_error", "temporal_memory")
    before = state["relations"][rid]["trust"]
    patch_record = {
        "component": "temporal_memory",
        "validation_fallback": True,
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
        "metrics": {"target": {"mae_inverse": 0.5}},
        "paths": {"result": "next.json"},
    }
    previous_feedback = {
        "features": {"overfit_score": 0.1},
        "diagnostics": [{"name": "long_horizon_error", "severity": 0.7, "confidence": 1.0}],
    }
    next_feedback = {
        "features": {"overfit_score": 0.1},
        "diagnostics": [{"name": "long_horizon_error", "severity": 0.2, "confidence": 1.0}],
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
    assert updates[0]["reward"]["reason"] == "patch_validation_fallback"
    assert state["relations"][rid]["trust"] < before


def test_action_memory_updates_feedback_component_edit_relation():
    state = initial_task_graph()
    rid = action_relation_id("long_horizon_error", "temporal_memory", "add_temporal_smoothing")
    before = state["action_memory"]["relations"][rid]["trust"]
    patch_record = {
        "component": "temporal_memory",
        "selected_edit": {
            "diagnostic": "long_horizon_error",
            "component": "temporal_memory",
            "edit_operator": "add_temporal_smoothing",
        },
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
        "diagnostics": [{"name": "long_horizon_error", "severity": 0.8, "confidence": 1.0}],
    }
    next_feedback = {
        "features": {"overfit_score": 0.3},
        "diagnostics": [{"name": "long_horizon_error", "severity": 0.9, "confidence": 1.0}],
    }

    updates = update_action_memory_from_outcome(
        state,
        patch_record,
        previous_result,
        next_result,
        previous_feedback,
        next_feedback,
        "mae_inverse",
    )
    rel = state["action_memory"]["relations"][rid]
    assert updates[0]["direction"] == "decrease"
    assert rel["trust"] < before
    assert state["action_memory"]["negative_experiences"]


def test_off_policy_mismatch_does_not_reward_selected_relation():
    state = initial_task_graph()
    rid = action_relation_id("target_degradation", "regularization", "increase_regularization")
    before = state["action_memory"]["relations"][rid]["trust"]
    patch_record = {
        "component": "temporal_memory",
        "edit_action": "add_temporal_smoothing",
        "component_mismatch": True,
        "edit_operator_mismatch": True,
        "selected_edit": {
            "diagnostic": "target_degradation",
            "component": "regularization",
            "edit_operator": "increase_regularization",
        },
    }
    previous_result = {
        "success": True,
        "metrics": {"target": {"mae_inverse": 1.0}},
        "paths": {"result": "prev.json"},
    }
    next_result = {
        "success": True,
        "metrics": {"target": {"mae_inverse": 0.5}},
        "paths": {"result": "next.json"},
    }
    previous_feedback = {
        "pemfc_context": {"dataset": "FC1"},
        "features": {"overfit_score": 0.1},
        "diagnostics": [{"name": "target_degradation", "severity": 0.9, "confidence": 1.0}],
    }
    next_feedback = {
        "pemfc_context": {"dataset": "FC1"},
        "features": {"overfit_score": 0.1},
        "diagnostics": [{"name": "target_degradation", "severity": 0.1, "confidence": 1.0}],
    }

    updates = update_action_memory_from_outcome(
        state,
        patch_record,
        previous_result,
        next_result,
        previous_feedback,
        next_feedback,
        "mae_inverse",
    )
    rel = state["action_memory"]["relations"][rid]
    assert updates[0]["evidence_policy"] == "off_policy"
    assert updates[0]["direction"] == "off_policy"
    assert updates[0]["reward"]["raw_reward"] > 0
    assert rel["trust"] == before
    assert rel.get("positive_count", 0) == 0


def test_negative_memory_blocks_repeated_dataset_failure():
    state = initial_task_graph()
    rid = action_relation_id("train_val_gap", "regularization", "increase_regularization")
    rel = state["action_memory"]["relations"][rid]
    rel["negative_count"] = 4
    rel["last_negative_update"] = 3
    rel["dataset_stats"] = {"FC2": {"negative_count": 3, "last_negative_update": 3}}
    state["action_memory"]["update_count"] = 4
    feedback = {
        "pemfc_context": {"dataset": "FC2"},
        "features": {"run_success": 1.0},
        "diagnostics": [{"name": "train_val_gap", "severity": 1.0, "confidence": 1.0, "evidence": {}}],
    }
    route = route_feedback(feedback, state)
    blocked = [row for row in route["edit_candidates"] if row["relation_id"] == rid]
    assert blocked and blocked[0]["blocked"] is True
    assert blocked[0]["suppression"]["cooldown_remaining"] > 0
    assert route["negative_reuse_suppression"]
    assert route["selected_edit"]["relation_id"] != rid


def test_relation_attention_metadata_and_temperature_are_recorded():
    state = initial_task_graph()
    feedback = {
        "current_target": 1.1,
        "best_target": 1.0,
        "pemfc_context": {"dataset": "FC1"},
        "features": {"run_success": 1.0, "improved_vs_best": 0.0},
        "diagnostics": [
            {"name": "long_horizon_error", "severity": 0.8, "confidence": 0.9, "evidence": {}},
            {"name": "residual_autocorrelation", "severity": 0.7, "confidence": 0.9, "evidence": {}},
        ],
    }
    route = route_feedback(feedback, state)
    attention = route["relation_attention"]
    assert attention["enabled"] is True
    assert attention["temperature"] >= attention["base_temperature"]
    weights = [row["attention_weight"] for row in route["edit_candidates"] if not row.get("blocked")]
    assert weights
    assert 0.99 <= sum(weights) <= 1.01
    assert route["selected_edit"]["attention_weight"] > 0
