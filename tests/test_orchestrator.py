import pytest

from forge.config import load_json, save_json
from forge.orchestrator import GraphOrchestrator


def test_orchestrator_records_stage_artifacts_and_events(tmp_path):
    orch = GraphOrchestrator.open(tmp_path)
    iter_dir = tmp_path / "iter_000"
    model_path = iter_dir / "model.py"
    iter_dir.mkdir()
    model_path.write_text("class ForgeModel: pass\n", encoding="utf-8")

    orch.ensure_iteration(0, iter_dir, model_path)
    with orch.stage(0, "prepare"):
        orch.record_artifact(0, "model", model_path, kind="python_source")
    orch.finish_iteration(0)

    state = load_json(tmp_path / "task_graph.json")
    record = state["iterations"]["iter_000"]
    assert record["status"] == "completed"
    assert record["stages"]["prepare"]["status"] == "succeeded"
    assert record["artifacts"]["model"]["path"] == str(model_path)
    assert state["events_count"] >= 3
    assert (tmp_path / "graph_events.jsonl").exists()


def test_orchestrator_marks_failed_stage(tmp_path):
    orch = GraphOrchestrator.open(tmp_path)
    iter_dir = tmp_path / "iter_000"
    orch.ensure_iteration(0, iter_dir)

    with pytest.raises(RuntimeError):
        with orch.stage(0, "evaluate"):
            raise RuntimeError("boom")

    state = load_json(tmp_path / "task_graph.json")
    record = state["iterations"]["iter_000"]
    assert record["status"] == "failed"
    assert record["stages"]["evaluate"]["status"] == "failed"
    assert "RuntimeError: boom" in record["stages"]["evaluate"]["error"]


def test_orchestrator_rejects_unknown_stage(tmp_path):
    orch = GraphOrchestrator.open(tmp_path)
    orch.ensure_iteration(0, tmp_path / "iter_000")
    with pytest.raises(Exception) as exc_info:
        with orch.stage(0, "unknown"):
            pass
    assert "Unknown orchestration stage" in str(exc_info.value)


def test_history_rows_resolves_legacy_relative_artifact_paths(tmp_path):
    run_root = tmp_path / "demo_run"
    orch = GraphOrchestrator.open(run_root)
    iter_dir = run_root / "iter_000"
    result_path = iter_dir / "result.json"
    iter_dir.mkdir(parents=True)
    save_json(
        {
            "success": True,
            "metrics": {"target": {"mae_inverse": 0.1}},
            "run_dir": str(iter_dir),
        },
        result_path,
    )

    orch.ensure_iteration(0, iter_dir)
    record = orch.state["iterations"]["iter_000"]
    record["artifacts"]["result"] = {
        "path": f"legacy/prefix/{run_root.name}/iter_000/result.json",
        "kind": "file",
    }
    orch.save()

    rows = orch.history_rows("mae_inverse")
    assert rows[0]["iteration"] == 0
    assert rows[0]["target_metric_value"] == 0.1
