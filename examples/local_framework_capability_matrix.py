"""
Replay and score a framework capability matrix locally.

Requires:
    pip install agent-simulate ai-evaluation

This certifies whether a framework has the task surfaces and capability
evidence needed before optimizer search changes prompts, tools, memory,
streaming, orchestration, lifecycle, security, observability, or exports.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    FrameworkCapabilityEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    normalize_framework_capability_matrix,
)


MATRIX = normalize_framework_capability_matrix(
    name="langgraph-capability-certificate",
    framework="langgraph",
    version="1.0",
    task_surfaces=["support_chat", "refund_workflow", "browser_research"],
    integrations=["futureagi", "mcp", "otel"],
    capabilities=[
        {"name": "tool_calling", "category": "tools", "status": "supported", "evidence": ["tools/list", "tools/call"]},
        {"name": "mcp_tool_session", "category": "tools", "status": "supported", "evidence": ["mcp tool session"]},
        {"name": "long_term_memory", "category": "memory", "status": "supported", "evidence": ["memory store adapter"]},
        {"name": "streaming_deltas", "category": "streaming", "status": "supported", "evidence": ["stream_events"]},
        {"name": "checkpoint_resume", "category": "lifecycle", "status": "supported", "evidence": ["checkpoint replay"]},
        {"name": "workflow_graph", "category": "orchestration", "status": "supported", "evidence": ["graph nodes and edges"]},
        {"name": "policy_guardrails", "category": "security", "status": "supported", "evidence": ["policy gate"]},
        {"name": "otel_trace_export", "category": "observability", "status": "supported", "evidence": ["OTel spans"]},
        {"name": "futureagi_export", "category": "exports", "status": "supported", "evidence": ["Future AGI regression row"]},
    ],
)


async def capability_auditor(input):
    return AgentResponse(
        content="Framework capability matrix certified tools, memory, streaming, lifecycle, orchestration, security, observability, and exports.",
        tool_calls=[
            {"id": "status", "name": "framework_capability_status", "arguments": {}},
            {
                "id": "tools",
                "name": "list_framework_capabilities",
                "arguments": {"category": "tools", "status": "supported"},
            },
            {
                "id": "checkpoint",
                "name": "inspect_framework_capability",
                "arguments": {"name": "checkpoint_resume"},
            },
            {"id": "surfaces", "name": "list_framework_task_surfaces", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="framework-capability-certificate",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs a framework certified before routing production agents through it.",
                outcome="The framework proves core capabilities, task surfaces, integrations, and evidence.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=capability_auditor,
        environment=FrameworkCapabilityEnvironment(MATRIX),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_framework_capabilities": [
                "framework_capability",
                "tool_calling",
                "long_term_memory",
                "streaming_deltas",
                "checkpoint_resume",
                "workflow_graph",
                "policy_guardrails",
                "otel_trace_export",
                "futureagi_export",
            ],
            "framework_capability_quality": {
                "framework": "langgraph",
                "required_capabilities": [
                    "tool_calling",
                    "long_term_memory",
                    "streaming_deltas",
                    "checkpoint_resume",
                    "workflow_graph",
                    "policy_guardrails",
                    "otel_trace_export",
                    "futureagi_export",
                ],
                "required_categories": ["tools", "memory", "streaming", "lifecycle", "orchestration", "security", "observability", "exports"],
                "required_task_surfaces": ["support_chat", "refund_workflow", "browser_research"],
                "required_integrations": ["futureagi", "mcp"],
                "min_supported_capabilities": 8,
                "min_support_rate": 0.85,
                "require_evidence": True,
                "max_missing_capabilities": 0,
                "require_tools": True,
                "require_memory": True,
                "require_streaming": True,
                "require_lifecycle": True,
                "require_orchestration": True,
                "require_security": True,
                "require_observability": True,
                "require_exports": True,
            },
        },
        threshold=0.9,
    )
    state = report.results[0].metadata["environment_state"]["framework_capability_matrix"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("summary:", state["summary"])
    print("framework_capability_coverage:", metrics.get("framework_capability_coverage"))
    print("framework_capability_quality:", metrics.get("framework_capability_quality"))


if __name__ == "__main__":
    asyncio.run(main())
