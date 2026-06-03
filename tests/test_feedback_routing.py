from forge.feedback import encode_feedback
from forge.routing import route_feedback


def test_shape_failure_routes_to_prediction_head():
    result = {
        "success": False,
        "error_type": "shape",
        "error_message": "RuntimeError: size mismatch",
        "traceback": "shape mismatch in output",
        "paths": {},
    }
    feedback = encode_feedback(result)
    route = route_feedback(feedback)
    assert route["primary_component"] == "prediction_head"


def test_cold_start_success_routes_to_factor_or_regularization():
    result = {
        "success": True,
        "metrics": {
            "inverse": {"mae": 0.01, "rmse": 0.02, "mape": 0.3},
            "train": {"final_train_loss": 0.2, "final_val_loss": 0.3, "early_stopped": False},
            "target": {"mae_inverse": 0.01},
        },
        "paths": {},
    }
    feedback = encode_feedback(result)
    route = route_feedback(feedback)
    assert "factor_fusion" in route["active_components"]


def test_routing_always_includes_primary_component():
    feedback = {"features": {"run_success": 1.0}}
    route = route_feedback(feedback)
    assert route["primary_component"] in route["active_components"]
