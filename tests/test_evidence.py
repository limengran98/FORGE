from forge.evidence import build_run_evidence_audit, method_framework


def test_method_framework_has_four_modules():
    framework = method_framework(candidate_tournament_k=3)

    assert framework["schema"] == "forge.method_framework.v1"
    assert [module["id"] for module in framework["modules"]] == [
        "pemfc_native_diagnostic_harness",
        "evidence_reconstruction_graph",
        "test_time_adaptive_strategy_memory",
        "k_candidate_evidence_tournament",
    ]
    assert framework["modules"][-1]["configured_k"] == 3


def test_evidence_audit_computes_routing_metrics():
    graph_state = {
        "iterations": {
            "iter_000": {
                "patch": {
                    "component": "temporal_memory",
                    "edit_action": "add_gate",
                    "routed_component": "temporal_memory",
                    "route_propagations": [
                        {"diagnostic": "long_horizon_error", "component": "temporal_memory"}
                    ],
                    "repair_attempts": [],
                }
            },
            "iter_001": {
                "patch": {
                    "component": "temporal_memory",
                    "edit_action": "add_gate",
                    "routed_component": "temporal_memory",
                    "route_propagations": [
                        {"diagnostic": "long_horizon_error", "component": "temporal_memory"}
                    ],
                    "repair_attempts": [{"validation_error": "shape"}],
                    "validation_fallback": True,
                }
            },
        }
    }
    history = [
        {
            "iteration": 0,
            "success": True,
            "target": {"mae_inverse": 0.10},
            "primary_component": "temporal_memory",
            "active_components": ["temporal_memory"],
            "result": {"success": True, "metrics": {"target": {"mae_inverse": 0.10}}},
        },
        {
            "iteration": 1,
            "success": True,
            "target": {"mae_inverse": 0.08},
            "primary_component": "temporal_memory",
            "active_components": ["temporal_memory"],
            "result": {"success": True, "metrics": {"target": {"mae_inverse": 0.08}}},
        },
        {
            "iteration": 2,
            "success": True,
            "target": {"mae_inverse": 0.09},
            "primary_component": "temporal_memory",
            "active_components": ["temporal_memory"],
            "result": {"success": True, "metrics": {"target": {"mae_inverse": 0.09}}},
        },
    ]

    audit = build_run_evidence_audit(graph_state, history, "mae_inverse")
    metrics = audit["metrics"]

    assert metrics["improvement_rate"] == 0.5
    assert metrics["invalid_edit_rate"] == 0.5
    assert metrics["routing_stability"] == 1.0
    assert metrics["evidence_alignment"] == 1.0
    assert metrics["budget_efficiency"]["attempts_to_best"] == 1
    assert audit["tables"]["attempts"][0]["branch_mode"] == "last_parent"
    assert audit["tables"]["attempts"][0]["trust_before_mean"] == 0.0
    assert audit["tables"]["relations"][0]["attempt_count"] == 2
    assert audit["tables"]["components"][0]["component"] == "temporal_memory"
    assert {
        "active_memory_reconstruction",
        "test_time_adaptation",
        "experience_reuse",
        "graph_branch_level_search",
        "domain_native_harness",
        "auditable_trajectories",
    } == {row["claim"] for row in audit["tables"]["method_evidence"]}
