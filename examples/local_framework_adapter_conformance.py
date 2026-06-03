"""
Certify a custom framework trace adapter locally.

Requires:
    pip install agent-simulate ai-evaluation

This checks whether custom framework records normalize into the semantic
channels and field mappings that downstream evaluation and optimization need.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    FrameworkTraceEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


RECORDS = [
    {
        "id": "model_1",
        "name": "custom llm generation",
        "type": "llm",
        "input": "order 123",
        "output": "Call search_order and store the result.",
        "usage": {"total_tokens": 48},
    },
    {
        "id": "tool_1",
        "name": "custom tool call",
        "type": "tool",
        "tool_name": "search_order",
        "input": {"order_id": "123"},
    },
    {
        "id": "memory_1",
        "name": "memory_update case_summary",
        "type": "memory_update",
        "memory_operation": "write",
        "memory_key": "case_summary",
        "memory_value": "order 123 resolved",
    },
    {
        "id": "state_1",
        "method": "updates",
        "params": {"data": {"case": {"status": "resolved"}}},
    },
]

ADAPTER_SPEC = {
    "required_signals": ["model", "tool", "memory", "state", "cost"],
    "required_mappings": {
        "model": ["input", "output", "cost"],
        "tool": ["tool_name", "input"],
        "memory": ["memory.operation", "memory.key"],
        "state": ["state"],
    },
}


async def adapter_auditor(input):
    return AgentResponse(
        content="Custom framework adapter conformance inspected.",
        tool_calls=[
            {"id": "status", "name": "framework_trace_status", "arguments": {}},
            {"id": "model", "name": "list_framework_spans", "arguments": {"signal": "model"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="framework-adapter-conformance",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs a custom framework adapter certified before optimization.",
                outcome="The adapter captures model, tool, memory, state, and cost mappings.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=adapter_auditor,
        environment=FrameworkTraceEnvironment(
            framework="custom_runtime",
            events=RECORDS,
            adapter_spec=ADAPTER_SPEC,
        ),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_framework_trace": [
                "adapter_conformance",
                "model",
                "tool",
                "memory",
                "state",
                "cost",
            ],
            "framework_adapter_conformance": ADAPTER_SPEC,
        },
        threshold=0.85,
    )
    state = report.results[0].metadata["environment_state"]["framework_trace"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("adapter_conformance:", state["adapter_conformance"])
    print("framework_trace_coverage:", metrics.get("framework_trace_coverage"))
    print("framework_adapter_conformance:", metrics.get("framework_adapter_conformance"))


if __name__ == "__main__":
    asyncio.run(main())
