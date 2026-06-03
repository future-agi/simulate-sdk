"""
Replay framework memory/skill traces and score cross-trial quality.

The fixture mirrors LangGraph/LangChain stream events that expose memory writes,
memory recalls, and skill-library updates. It runs offline with no framework
install, model, or API key.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    load_langgraph_event_stream,
)


LANGGRAPH_EVENTS = [
    {
        "seq": 1,
        "method": "updates",
        "params": {
            "namespace": ["refund_graph:run_1", "memory_node:task_1"],
            "data": {
                "memory_operation": "write",
                "memory_key": "order_id",
                "memory_value": "ord_123",
            },
        },
    },
    {
        "seq": 2,
        "method": "updates",
        "params": {
            "namespace": ["refund_graph:run_1", "memory_node:task_1"],
            "data": {
                "memory_operation": "write",
                "memory_key": "policy_version",
                "memory_value": "2026-05",
            },
        },
    },
    {
        "seq": 3,
        "method": "values",
        "params": {
            "namespace": ["refund_graph:run_1", "memory_node:task_1"],
            "data": {"memory_operation": "recall", "memory_key": "order_id"},
        },
    },
    {
        "seq": 4,
        "method": "values",
        "params": {
            "namespace": ["refund_graph:run_1", "memory_node:task_1"],
            "data": {"memory_operation": "recall", "memory_key": "policy_version"},
        },
    },
    {
        "seq": 5,
        "method": "updates",
        "params": {
            "namespace": ["refund_graph:run_1", "skill_node:task_2"],
            "data": {
                "skill_name": "refund_policy_check",
                "skill_steps": ["lookup", "verify", "respond"],
            },
        },
    },
]


async def trace_inspector_agent(input):
    return AgentResponse(
        content="Refund context persisted and the refund_policy_check skill remained available.",
        tool_calls=[
            {"id": "status", "name": "framework_trace_status", "arguments": {}},
            {"id": "memory", "name": "list_framework_spans", "arguments": {"signal": "memory"}},
            {"id": "skill", "name": "list_framework_spans", "arguments": {"signal": "skill"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="cross-trial-memory-skill",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam starts a refund workflow for order ord_123.",
                outcome="The agent persists refund memory and reusable skill evidence.",
            ),
            Persona(
                persona={"name": "Riya", "risk_profile": "standard"},
                situation="Riya resumes the same refund workflow from memory.",
                outcome="The agent recalls refund memory and reuses the skill evidence.",
            ),
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=trace_inspector_agent,
        environment=load_langgraph_event_stream({"events": LANGGRAPH_EVENTS}),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["framework_trace_status", "list_framework_spans"],
            "available_tools": ["framework_trace_status", "list_framework_spans"],
            "required_framework_trace": ["memory", "skill"],
            "expected_cross_trial_memory": {
                "required_keys": ["order_id", "policy_version"],
                "required_recall_keys": ["order_id", "policy_version"],
                "forbidden_keys": ["raw_user_secret"],
                "min_precision": 1.0,
                "min_recall": 1.0,
                "min_trials_present": 2,
                "require_persistence": True,
            },
            "expected_cross_trial_skills": [
                {
                    "name": "refund_policy_check",
                    "required_steps": ["lookup", "verify", "respond"],
                    "min_trials_present": 2,
                    "require_persistent_after_first": True,
                }
            ],
        },
        threshold=0.85,
    )

    metrics = evaluation.summary["metric_averages"]
    cross_trial = evaluation.summary["cross_trial_memory_skill"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("framework_trace_coverage:", metrics.get("framework_trace_coverage"))
    print("cross_trial_memory_skill:", cross_trial["score"])


if __name__ == "__main__":
    asyncio.run(main())
