"""
Replay a local LangGraph/LangChain event stream and score transcript quality.

The fixture mirrors LangGraph/LangChain `stream_events(..., version="v3")`
protocol events: messages, tool calls, state updates, and final values. It runs
offline with no framework import, no model, and no API keys.

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
        "method": "messages",
        "params": {
            "namespace": ["refund_graph:run_1", "support_agent:task_1"],
            "data": {"node": "support_agent", "text": "I will look up order ord_123."},
        },
    },
    {
        "seq": 2,
        "method": "tools",
        "params": {
            "namespace": ["refund_graph:run_1", "support_agent:task_1"],
            "data": {"event": "tool-start", "tool_name": "lookup_order", "input": {"order_id": "ord_123"}},
        },
    },
    {
        "seq": 3,
        "method": "tools",
        "params": {
            "namespace": ["refund_graph:run_1", "support_agent:task_1"],
            "data": {
                "event": "tool-finish",
                "tool_name": "issue_refund",
                "output": {"status": "resolved"},
            },
        },
    },
    {
        "seq": 4,
        "method": "updates",
        "params": {
            "namespace": ["refund_graph:run_1", "policy_node:task_2"],
            "data": {"case": {"status": "resolved", "approval": "captured"}},
        },
    },
    {
        "seq": 5,
        "method": "values",
        "params": {
            "namespace": ["refund_graph:run_1", "support_agent:task_1"],
            "data": {"final_output": "Refund approved for order ord_123."},
        },
    },
]


async def trace_inspector_agent(input):
    return AgentResponse(
        content="Refund approved for order ord_123 after policy and tool checks.",
        tool_calls=[
            {"id": "status", "name": "framework_trace_status", "arguments": {}},
            {"id": "tools", "name": "list_framework_spans", "arguments": {"signal": "tool"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="langgraph-event-stream-replay",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs a LangGraph refund agent run inspected before optimization.",
                outcome="Refund approved for order ord_123.",
            )
        ],
    )
    environment = load_langgraph_event_stream(
        {"events": LANGGRAPH_EVENTS},
        metadata={"source": "local stream_events fixture"},
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=trace_inspector_agent,
        environment=environment,
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["framework_trace_status", "list_framework_spans"],
            "available_tools": ["framework_trace_status", "list_framework_spans"],
            "required_framework_trace": ["model", "tool", "state"],
            "framework_transcript_quality": {
                "required_event_methods": ["messages", "tools", "updates"],
                "required_nodes": ["support_agent", "policy_node"],
                "required_subgraphs": ["refund_graph"],
                "expected_tool_sequence": ["lookup_order", "issue_refund"],
                "expected_state": {"case": {"status": "resolved", "approval": "captured"}},
                "output_contains": ["Refund approved for order ord_123"],
            },
            "metric_weights": {"framework_transcript_quality": 4.0},
            "success_criteria": ["Refund approved for order ord_123"],
        },
        threshold=0.85,
    )

    averages = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("framework_trace_coverage:", averages.get("framework_trace_coverage"))
    print("framework_transcript_quality:", averages.get("framework_transcript_quality"))


if __name__ == "__main__":
    asyncio.run(main())
