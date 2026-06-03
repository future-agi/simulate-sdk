"""
Replay and score a framework portability matrix locally.

Requires:
    pip install agent-simulate ai-evaluation

Use this after capability/probe certification when an agent configuration must
move between frameworks or protocols without losing runtime, tools, memory,
streaming, lifecycle, orchestration, security, observability, or export paths.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    FrameworkPortabilityEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    normalize_framework_portability_matrix,
)


PORTABILITY_MATRIX = normalize_framework_portability_matrix(
    name="langgraph-to-openai-agents-portability",
    source_framework="langgraph",
    target_framework="openai_agents",
    version="2026-06",
    mappings=[
        {"id": "invoke", "source": "graph.invoke", "target": "Runner.run", "category": "runtime", "status": "mapped", "evidence": ["dry run"]},
        {"id": "tool_discovery", "source": "tools/list", "target": "Agents SDK tools", "category": "tools", "status": "mapped", "evidence": ["schema map"]},
        {"id": "tool_call", "source": "ToolNode", "target": "function tool", "category": "tools", "status": "mapped", "evidence": ["call/result replay"]},
        {"id": "short_term_state", "source": "graph state", "target": "session state", "category": "memory", "status": "mapped", "evidence": ["state projection"]},
        {"id": "streaming_events", "source": "astream_events", "target": "run stream events", "category": "streaming", "status": "mapped", "evidence": ["chunk replay"]},
        {"id": "checkpoint_resume", "source": "checkpointer", "target": "session resume", "category": "lifecycle", "status": "mapped", "evidence": ["resume replay"]},
        {"id": "handoff", "source": "graph route", "target": "agent handoff", "category": "orchestration", "status": "mapped", "evidence": ["route map"]},
        {"id": "guardrail", "source": "policy node", "target": "guardrail", "category": "security", "status": "mapped", "evidence": ["policy gate"]},
        {"id": "otel_trace", "source": "otel spans", "target": "tracing processor", "category": "observability", "status": "mapped", "evidence": ["span map"]},
        {"id": "futureagi_export", "source": "dataset export", "target": "Future AGI row", "category": "exports", "status": "mapped", "evidence": ["export row"]},
    ],
    constraints=["preserve tool schemas", "preserve trace ids"],
)


async def portability_auditor(input):
    return AgentResponse(
        content="Framework portability mappings preserve runtime, tools, memory, streaming, lifecycle, orchestration, security, observability, and exports.",
        tool_calls=[
            {"id": "status", "name": "framework_portability_status", "arguments": {}},
            {"id": "tools", "name": "list_framework_portability_mappings", "arguments": {"category": "tools", "status": "mapped"}},
            {"id": "checkpoint", "name": "inspect_framework_portability_mapping", "arguments": {"id": "checkpoint_resume"}},
            {"id": "gaps", "name": "list_framework_portability_gaps", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="framework-portability-matrix",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs a migration certificate before moving an agent config to a new framework.",
                outcome="The migration preserves runtime, tools, memory, streaming, lifecycle, orchestration, security, observability, and export mappings.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=portability_auditor,
        environment=FrameworkPortabilityEnvironment(PORTABILITY_MATRIX),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_framework_portability": [
                "framework_portability",
                "invoke",
                "tool_discovery",
                "tool_call",
                "short_term_state",
                "streaming_events",
                "checkpoint_resume",
                "handoff",
                "guardrail",
                "otel_trace",
                "futureagi_export",
            ],
            "framework_portability_quality": {
                "source_framework": "langgraph",
                "target_framework": "openai_agents",
                "required_mappings": [
                    "invoke",
                    "tool_discovery",
                    "tool_call",
                    "short_term_state",
                    "streaming_events",
                    "checkpoint_resume",
                    "handoff",
                    "guardrail",
                    "otel_trace",
                    "futureagi_export",
                ],
                "required_categories": ["runtime", "tools", "memory", "streaming", "lifecycle", "orchestration", "security", "observability", "exports"],
                "min_mapped_mappings": 10,
                "min_mapping_rate": 0.9,
                "min_required_mapping_rate": 0.9,
                "max_missing_mappings": 0,
                "max_blocked_mappings": 0,
                "require_evidence": True,
                "require_tools": True,
                "require_memory": True,
                "require_streaming": True,
                "require_lifecycle": True,
                "require_orchestration": True,
                "require_security": True,
                "require_observability": True,
                "require_exports": True,
                "require_runtime": True,
            },
        },
        threshold=0.9,
    )
    state = report.results[0].metadata["environment_state"]["framework_portability_matrix"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("summary:", state["summary"])
    print("framework_portability_coverage:", metrics.get("framework_portability_coverage"))
    print("framework_portability_quality:", metrics.get("framework_portability_quality"))


if __name__ == "__main__":
    asyncio.run(main())
