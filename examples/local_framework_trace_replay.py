"""
Run a local framework trace replay simulation.

This models native trace evidence from orchestration frameworks such as OpenAI
Agents, LangGraph/LangChain, CrewAI, AutoGen, LiveKit, or Pipecat. The trace is
normalized into framework spans and scored without importing the framework,
starting a room, or calling a model.

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
    environment = FrameworkTraceEnvironment(
        framework="mixed_agent_stack",
        spans=[
            {"id": "agent_1", "name": "langgraph node support_agent", "type": "agent", "duration_ms": 14},
            {"id": "model_1", "name": "generation_span gpt_route", "type": "llm", "usage": {"tokens": 120}},
            {"id": "retrieval_1", "name": "retriever policy_vector_search", "type": "retrieval"},
            {"id": "memory_1", "name": "memory_update case_summary", "type": "memory"},
            {"id": "tool_1", "name": "function_span search_order", "type": "tool"},
            {"id": "handoff_1", "name": "handoff_span policy_specialist", "type": "handoff"},
            {"id": "guardrail_1", "name": "guardrail_span pii_check", "type": "guardrail"},
            {"id": "browser_1", "name": "computer_use browser_click", "type": "browser"},
            {"id": "voice_1", "name": "transcription_span caller_audio", "type": "voice"},
        ],
        metadata={"source": "local fixture"},
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
