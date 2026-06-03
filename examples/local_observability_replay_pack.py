"""
Run a local Future AGI-style observability replay pack.

Requires:
    pip install agent-simulate ai-evaluation

This turns failed regression rows into a trace artifact with status/list/inspect
tools, then scores replay-pack coverage and quality with ai-evaluation.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    ObservabilityReplayEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


REPLAY_CASES = [
    {
        "id": "policy_regression",
        "input": {
            "observability": {
                "run_id": "run_policy_failed",
                "source": "futureagi",
                "framework": "langgraph",
                "score": 0.2,
                "passed": False,
                "metrics": {
                    "policy_adherence": 0.2,
                    "framework_trace_coverage": 0.67,
                },
                "trace_signals": ["agent", "model"],
                "raw": {
                    "trace_id": "trace_policy_failed",
                    "agent_report_evaluation": {"summary": {"metric_averages": {"policy_adherence": 0.2}}},
                },
            }
        },
        "expected": {
            "required_metrics": {"policy_adherence": 0.85, "framework_trace_coverage": 1.0},
            "required_trace_signals": ["agent", "model", "tool"],
        },
        "tags": ["policy"],
    },
    {
        "id": "memory_passed",
        "observability": {
            "run_id": "run_memory_passed",
            "source": "futureagi",
            "framework": "langgraph",
            "score": 0.95,
            "passed": True,
            "metrics": {
                "policy_adherence": 0.95,
                "framework_trace_coverage": 1.0,
                "memory_correctness": 0.95,
            },
            "trace_signals": ["agent", "model", "tool", "memory"],
            "raw": {"trace_id": "trace_memory_passed"},
        },
        "expected_response": {
            "required_metrics": {
                "policy_adherence": 0.85,
                "framework_trace_coverage": 1.0,
                "memory_correctness": 0.85,
            },
            "required_trace_signals": ["agent", "model", "tool"],
        },
        "tags": ["memory"],
    },
]


async def replay_agent(input):
    return AgentResponse(
        content="I inspected the observability replay pack and identified the failed policy and trace cases.",
        tool_calls=[
            {"id": "status", "name": "observability_replay_status", "arguments": {}},
            {
                "id": "failed",
                "name": "list_observability_replay_cases",
                "arguments": {"failed_only": True},
            },
            {
                "id": "case",
                "name": "inspect_observability_replay_case",
                "arguments": {"id": "policy_regression"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="observability-replay-pack",
        dataset=[
            Persona(
                persona={"name": "Asha", "segment": "refunds"},
                situation="Asha needs failed Future AGI regression rows replayed locally.",
                outcome="The replay pack preserves metric, trace, and raw evidence.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=replay_agent,
        environment=ObservabilityReplayEnvironment(
            REPLAY_CASES,
            name="futureagi-refund-observability-regressions",
            source="futureagi",
            framework="langgraph",
            required_trace_signals=["agent", "model", "tool"],
        ),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_observability_replay": ["replay_pack", "case", "failure", "metric", "trace_signal", "raw"],
            "observability_replay_quality": {
                "min_case_count": 2,
                "min_failed_case_count": 1,
                "required_metrics": ["policy_adherence", "framework_trace_coverage"],
                "required_failed_metrics": ["policy_adherence", "framework_trace_coverage"],
                "required_trace_signals": ["agent", "model", "tool"],
                "required_tags": ["policy", "missing_signal:tool"],
                "expected_case_ids": ["policy_regression", "memory_passed"],
                "require_raw_evidence": True,
            },
        },
        threshold=0.85,
    )
    state = report.results[0].metadata["environment_state"]["observability_replay_pack"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("summary:", state["summary"])
    print("observability_replay_coverage:", metrics.get("observability_replay_coverage"))
    print("observability_replay_quality:", metrics.get("observability_replay_quality"))


if __name__ == "__main__":
    asyncio.run(main())
