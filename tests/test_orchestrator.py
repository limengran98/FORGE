import pytest

from forge.config import load_json
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

