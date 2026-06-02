import sys
import types
from types import SimpleNamespace

from fi.simulate.agent.wrapper import SimulationArtifact, SimulationEvent
from fi.simulate.evaluation.ai_eval import evaluate_agent_report, evaluate_report
from fi.simulate.simulation.models import Persona, TestCaseResult as SimTestCaseResult, TestReport as SimTestReport


def test_evaluate_report_maps_trajectory_sources(monkeypatch):
    captured_inputs = []

    class EvalResult:
        output = 1.0
        reason = "ok"
        score = 1.0

    class BatchResult:
        eval_results = [EvalResult()]

    class FakeEvaluator:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, *, eval_templates, inputs, model_name):
            captured_inputs.append(inputs)
            return BatchResult()

    fake_module = types.ModuleType("fi.evals")
    fake_module.Evaluator = FakeEvaluator
    monkeypatch.setitem(sys.modules, "fi.evals", fake_module)

    report = SimTestReport(
        results=[
            SimTestCaseResult(
                persona=Persona(
                    persona={"name": "Avery"},
                    situation="Avery needs help.",
                    outcome="Issue resolved.",
                ),
                transcript="User: hi\nAgent: hello",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=[{"name": "lookup"}],
                artifacts=[SimulationArtifact(type="screenshot", uri="file:///tmp/a.png")],
                events=[SimulationEvent(type="browser_action", name="click")],
                metadata={"engine": "local_text"},
            )
        ]
    )

    evaluate_report(
        report,
        eval_specs=[
            {
                "template": "agent_trajectory",
                "map": {
                    "input": "messages",
                    "tools": "tool_calls",
                    "artifacts": "artifacts",
                    "events": "events",
                    "metadata": "metadata",
                    "persona": "persona",
                },
            }
        ],
    )

    assert "hi" in captured_inputs[0]["input"]
    assert "lookup" in captured_inputs[0]["tools"]
    assert "screenshot" in captured_inputs[0]["artifacts"]
    assert "browser_action" in captured_inputs[0]["events"]
    assert "local_text" in captured_inputs[0]["metadata"]
    assert "Avery" in captured_inputs[0]["persona"]


def test_evaluate_agent_report_attaches_local_scores(monkeypatch):
    captured = {}

    def fake_evaluate_agent_report(report, *, config, threshold):
        captured["config"] = config
        captured["threshold"] = threshold
        return SimpleNamespace(
            score=0.82,
            passed=True,
            threshold=threshold,
            summary={"case_count": 1, "metric_averages": {"trajectory_score": 0.9}},
            cases=[
                SimpleNamespace(
                    score=0.82,
                    passed=True,
                    metrics=[
                        SimpleNamespace(
                            name="trajectory_score",
                            score=0.9,
                            reason="ok",
                            details={},
                        )
                    ],
                    findings=[],
                )
            ],
        )

    fake_evals = types.ModuleType("fi.evals")
    fake_metrics = types.ModuleType("fi.evals.metrics")
    fake_agents = types.ModuleType("fi.evals.metrics.agents")
    fake_agents.evaluate_agent_report = fake_evaluate_agent_report
    monkeypatch.setitem(sys.modules, "fi.evals", fake_evals)
    monkeypatch.setitem(sys.modules, "fi.evals.metrics", fake_metrics)
    monkeypatch.setitem(sys.modules, "fi.evals.metrics.agents", fake_agents)

    report = SimTestReport(
        results=[
            SimTestCaseResult(
                persona=Persona(
                    persona={"name": "Avery"},
                    situation="Avery needs checkout help.",
                    outcome="Checkout resolved.",
                ),
                transcript="User: help\nAgent: resolved",
                messages=[{"role": "assistant", "content": "resolved"}],
                metadata={"engine": "local_text"},
            )
        ]
    )

    evaluation = evaluate_agent_report(
        report,
        config={"required_tools": ["search_order"]},
        threshold=0.75,
    )

    assert evaluation.score == 0.82
    assert captured["config"]["required_tools"] == ["search_order"]
    assert captured["threshold"] == 0.75
    assert report.results[0].evaluation["agent_report"]["score"] == 0.82
    assert report.results[0].evaluation["agent_report"]["case_score"] == 0.82
    assert report.results[0].evaluation["agent_report"]["metrics"][0]["name"] == "trajectory_score"
    assert report.results[0].metadata["agent_report_summary"]["passed"] is True
