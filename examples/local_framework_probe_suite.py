"""
Replay and score framework adapter smoke probes locally.

Requires:
    pip install agent-simulate ai-evaluation

Use this after a capability matrix: declarations say what a framework should
support, while probes prove the adapter actually passed invoke, tool, memory,
streaming, lifecycle, orchestration, security, observability, and export checks.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    FrameworkProbeEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    normalize_framework_probe_suite,
)


PROBE_SUITE = normalize_framework_probe_suite(
    name="langgraph-adapter-probes",
    framework="langgraph",
    version="1.0",
    probes=[
        {"id": "invoke", "operation": "invoke", "category": "runtime", "status": "passed", "evidence": ["ainvoke dry run"], "latency_ms": 18},
        {"id": "list_tools", "operation": "list_tools", "category": "tools", "status": "passed", "evidence": ["tools/list"]},
        {"id": "tool_call", "operation": "tool_call", "category": "tools", "status": "passed", "evidence": ["lookup_policy result"]},
        {"id": "write_memory", "operation": "write_memory", "category": "memory", "status": "passed", "evidence": ["memory write"]},
        {"id": "read_memory", "operation": "read_memory", "category": "memory", "status": "passed", "evidence": ["memory read"]},
        {"id": "stream", "operation": "stream", "category": "streaming", "status": "passed", "evidence": ["stream chunk"]},
        {"id": "checkpoint_save", "operation": "checkpoint_save", "category": "lifecycle", "status": "passed", "evidence": ["checkpoint"]},
        {"id": "checkpoint_resume", "operation": "checkpoint_resume", "category": "lifecycle", "status": "passed", "evidence": ["resume"]},
        {"id": "handoff", "operation": "handoff", "category": "orchestration", "status": "passed", "evidence": ["handoff contract"]},
        {"id": "guardrail", "operation": "guardrail", "category": "security", "status": "passed", "evidence": ["policy gate"]},
        {"id": "trace_export", "operation": "trace_export", "category": "observability", "status": "passed", "evidence": ["OTel span"]},
        {"id": "export", "operation": "export", "category": "exports", "status": "passed", "evidence": ["Future AGI row"]},
    ],
)


async def probe_auditor(input):
    return AgentResponse(
        content="Framework adapter probes passed across runtime, tools, memory, streaming, lifecycle, orchestration, security, observability, and exports.",
        tool_calls=[
            {"id": "status", "name": "framework_probe_status", "arguments": {}},
            {"id": "tools", "name": "list_framework_probes", "arguments": {"category": "tools", "status": "passed"}},
            {"id": "checkpoint", "name": "inspect_framework_probe", "arguments": {"id": "checkpoint_resume"}},
            {"id": "failures", "name": "list_framework_probe_failures", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="framework-probe-suite",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs framework adapter smoke probes before rollout.",
                outcome="The adapter proves invoke, tools, memory, streaming, lifecycle, orchestration, security, observability, and export operations.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=probe_auditor,
        environment=FrameworkProbeEnvironment(PROBE_SUITE),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_framework_probes": [
                "framework_probe",
                "invoke",
                "list_tools",
                "tool_call",
                "write_memory",
                "read_memory",
                "stream",
                "checkpoint_save",
                "checkpoint_resume",
                "handoff",
                "guardrail",
                "trace_export",
                "export",
            ],
            "framework_probe_quality": {
                "framework": "langgraph",
                "required_operations": [
                    "invoke",
                    "list_tools",
                    "tool_call",
                    "write_memory",
                    "read_memory",
                    "stream",
                    "checkpoint_save",
                    "checkpoint_resume",
                    "handoff",
                    "guardrail",
                    "trace_export",
                    "export",
                ],
                "required_categories": ["tools", "memory", "streaming", "lifecycle", "orchestration", "security", "observability", "exports"],
                "min_passed_probes": 12,
                "min_required_pass_rate": 1.0,
                "max_failed_probes": 0,
                "max_blocked_probes": 0,
                "require_evidence": True,
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
    state = report.results[0].metadata["environment_state"]["framework_probe_suite"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("summary:", state["summary"])
    print("framework_probe_coverage:", metrics.get("framework_probe_coverage"))
    print("framework_probe_quality:", metrics.get("framework_probe_quality"))


if __name__ == "__main__":
    asyncio.run(main())
