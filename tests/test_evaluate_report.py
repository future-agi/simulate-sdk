import sys
import types

from fi.simulate.agent.wrapper import SimulationArtifact, SimulationEvent
from fi.simulate.evaluation.ai_eval import evaluate_report
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
