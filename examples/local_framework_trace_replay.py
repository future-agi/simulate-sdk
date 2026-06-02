"""
Run a local framework trace replay simulation.

This models native trace evidence from orchestration frameworks such as OpenAI
Agents, LangGraph/LangChain, CrewAI, AutoGen, LiveKit, or Pipecat. The trace is
loaded from an OTLP-style TraceAI export, normalized into framework spans, and
scored without importing the framework, starting a room, or calling a model.

Requires:
    pip install agent-simulate ai-evaluation
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


async def trace_inspector_agent(input):
    return AgentResponse(
        content=(
            "Framework trace inspected. The run includes agent, model, tool, "
            "handoff, guardrail, retrieval, memory, browser, and voice spans."
        ),
        tool_calls=[
            {"id": "status", "name": "framework_trace_status", "arguments": {}},
            {"id": "tools", "name": "list_framework_spans", "arguments": {"signal": "tool"}},
            {"id": "handoff", "name": "inspect_framework_span", "arguments": {"id": "handoff_1"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="framework-trace-replay",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs a framework run inspected before optimization.",
                outcome="Framework trace inspected.",
            )
        ],
    )
    trace_export = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "support-agent"}},
                        {"key": "futureagi.project", "value": {"stringValue": "orders"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "traceAI.autoinstrumentation", "version": "0.1.0"},
                        "spans": [
                            {
                                "traceId": "trace_1",
                                "spanId": "agent_1",
                                "name": "AutoGen AssistantAgent support_agent",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "1014000000",
                                "attributes": [
                                    {"key": "fi.span.kind", "value": {"stringValue": "AGENT"}},
                                    {"key": "input.value", "value": {"stringValue": "order 123"}},
                                ],
                            },
                            {
                                "traceId": "trace_1",
                                "spanId": "model_1",
                                "parentSpanId": "agent_1",
                                "name": "DSPy Predict gpt_route",
                                "attributes": [
                                    {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                                    {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "96"}},
                                    {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "24"}},
                                ],
                            },
                            {
                                "traceId": "trace_1",
                                "spanId": "retrieval_1",
                                "parentSpanId": "agent_1",
                                "name": "LlamaIndex retriever policy_vector_search",
                                "attributes": [
                                    {"key": "gen_ai.operation.name", "value": {"stringValue": "retrieve"}}
                                ],
                            },
                            {"traceId": "trace_1", "spanId": "memory_1", "name": "memory_update case_summary"},
                            {"traceId": "trace_1", "spanId": "tool_1", "name": "MCP tool call search_order"},
                            {"traceId": "trace_1", "spanId": "handoff_1", "name": "handoff_span policy_specialist"},
                            {"traceId": "trace_1", "spanId": "guardrail_1", "name": "guardrail_span pii_check"},
                            {"traceId": "trace_1", "spanId": "browser_1", "name": "computer_use browser_click"},
                            {"traceId": "trace_1", "spanId": "voice_1", "name": "LiveKit transcription_span caller_audio"},
                        ],
                    }
                ],
            }
        ]
    }
    environment = FrameworkTraceEnvironment.from_export(
        framework="mixed_agent_stack",
        export=trace_export,
        metadata={"source": "local otlp fixture", "storage": "futureagi"},
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
            "required_tools": [
                "framework_trace_status",
                "list_framework_spans",
                "inspect_framework_span",
            ],
            "available_tools": [
                "framework_trace_status",
                "list_framework_spans",
                "inspect_framework_span",
            ],
            "required_artifact_types": ["trace"],
            "required_framework_trace": [
                "agent",
                "model",
                "tool",
                "handoff",
                "guardrail",
                "retrieval",
                "memory",
                "browser",
                "voice",
                "latency",
                "cost",
            ],
            "success_criteria": ["framework trace inspected"],
        },
        threshold=0.85,
    )

    result = report.results[0]
    trace_state = result.metadata["environment_state"]["framework_trace"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("signals:", trace_state["signals"])
    print("framework_trace_coverage:", evaluation.summary["metric_averages"]["framework_trace_coverage"])


if __name__ == "__main__":
    asyncio.run(main())
