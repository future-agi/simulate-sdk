"""
Replay a local multi-agent framework transcript.

This uses AutoGen-style group-chat records, but the same loader pattern works
for CrewAI event logs and OpenAI Agents SDK trace/span exports. No framework
install, model, or API key is required.

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
    load_autogen_groupchat_transcript,
)


def autogen_records():
    return [
        {
            "type": "TextMessage",
            "source": "PlanningAgent",
            "content": "1. WebSearchAgent: search order 123. 2. DataAnalystAgent: verify refund.",
            "signals": ["agent", "model"],
        },
        {
            "type": "ToolCallRequestEvent",
            "source": "WebSearchAgent",
            "content": [
                {
                    "id": "call_search",
                    "name": "search_policy",
                    "arguments": {"order_id": "123"},
                }
            ],
            "signals": ["tool"],
        },
        {
            "type": "ToolCallExecutionEvent",
            "source": "WebSearchAgent",
            "content": [{"name": "search_policy", "content": "Order 123 policy found."}],
            "signals": ["tool"],
        },
        {
            "type": "handoff_span",
            "handoff_from": "triage_agent",
            "handoff_to": "refund_agent",
            "task": "order 123 refund policy escalation",
            "signals": ["handoff"],
        },
        {
            "type": "TextMessage",
            "source": "DataAnalystAgent",
            "content": "The refund is policy-compliant. TERMINATE",
            "termination": "TERMINATE",
            "signals": ["agent", "model"],
        },
    ]


async def transcript_inspector_agent(input):
    return AgentResponse(
        content="The multi-agent transcript includes planning, search, handoff, analysis, and termination.",
        tool_calls=[
            {"id": "status", "name": "framework_trace_status", "arguments": {}},
            {"id": "tools", "name": "list_framework_spans", "arguments": {"signal": "tool"}},
            {"id": "handoff", "name": "list_framework_spans", "arguments": {"signal": "handoff"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="multi-agent-framework-transcript",
        dataset=[
            Persona(
                persona={"name": "Anika", "risk_profile": "standard"},
                situation="Anika needs a refund decision audited from a multi-agent trace.",
                outcome="The trace shows planning, search, handoff, analyst verification, and termination.",
            )
        ],
    )
    environment = load_autogen_groupchat_transcript({"events": autogen_records()})
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=transcript_inspector_agent,
        environment=environment,
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["framework_trace_status", "list_framework_spans"],
            "available_tools": ["framework_trace_status", "list_framework_spans"],
            "required_framework_trace": ["agent", "tool", "handoff"],
            "framework_transcript_quality": {
                "required_speakers": ["PlanningAgent", "WebSearchAgent", "DataAnalystAgent"],
                "expected_speaker_sequence": ["PlanningAgent", "WebSearchAgent", "DataAnalystAgent"],
                "min_turns": 3,
                "expected_messages": [
                    {"speaker": "PlanningAgent", "contains": ["WebSearchAgent", "DataAnalystAgent"]},
                    {"speaker": "DataAnalystAgent", "contains": ["refund is policy-compliant"]},
                ],
                "expected_handoffs": [
                    {
                        "from_agent": "triage_agent",
                        "to_agent": "refund_agent",
                        "task_contains": ["order 123"],
                    }
                ],
                "required_tools_by_speaker": {"WebSearchAgent": ["search_policy"]},
                "termination_contains": ["TERMINATE"],
            },
            "metric_weights": {"framework_transcript_quality": 5.0, "framework_trace_coverage": 2.0},
        },
        threshold=0.85,
    )

    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("framework_trace_coverage:", metrics.get("framework_trace_coverage"))
    print("framework_transcript_quality:", metrics.get("framework_transcript_quality"))


if __name__ == "__main__":
    asyncio.run(main())
