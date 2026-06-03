"""
Run a local orchestration graph trace replay.

Requires:
    pip install agent-simulate ai-evaluation

The fixture mirrors graph/runtime evidence from LangGraph, LiveKit, Pipecat, or
OpenTelemetry-style GenAI traces without importing those frameworks.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    OrchestrationTraceEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


TRACE_RECORDS = [
    {
        "id": "workflow",
        "name": "invoke_workflow refund_graph",
        "attributes": {
            "gen_ai.operation.name": "invoke_workflow",
            "gen_ai.workflow.name": "refund_graph",
        },
        "duration_ms": 8,
    },
    {
        "id": "route_policy",
        "name": "handoff triage to policy",
        "node": "triage_agent",
        "route_from": "triage_agent",
        "route_to": "policy_agent",
        "type": "handoff",
        "latency_ms": 12,
    },
    {
        "id": "policy_error",
        "name": "policy_agent tool timeout",
        "node": "policy_agent",
        "event": "error",
        "error": {"message": "rate limit", "recoverable": True},
        "attempt": 1,
        "latency_ms": 40,
    },
    {
        "id": "policy_retry",
        "name": "policy_agent retry succeeded",
        "node": "policy_agent",
        "event": "retry",
        "status": "success",
        "attempt": 2,
        "recovered": True,
        "latency_ms": 35,
        "usage": {"total_tokens": 80},
    },
    {
        "id": "refund_tool",
        "name": "execute_tool issue_refund",
        "node": "refund_tool",
        "route_from": "policy_agent",
        "route_to": "refund_tool",
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "issue_refund",
        },
        "latency_ms": 30,
    },
    {
        "id": "final_state",
        "method": "updates",
        "params": {
            "namespace": ["refund_graph:run_1", "final_node:task_1"],
            "data": {"case": {"status": "resolved"}},
        },
    },
]


async def orchestration_agent(input):
    return AgentResponse(
        content="I inspected the workflow graph, retry, recovery, and route evidence.",
        tool_calls=[
            {"id": "status", "name": "orchestration_trace_status", "arguments": {}},
            {"id": "retry", "name": "list_orchestration_steps", "arguments": {"signal": "retry"}},
            {"id": "node", "name": "inspect_orchestration_node", "arguments": {"id": "policy_agent"}},
            {
                "id": "edge",
                "name": "inspect_orchestration_edge",
                "arguments": {"from": "triage_agent", "to": "policy_agent"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="orchestration-graph-trace",
        dataset=[
            Persona(
                persona={"name": "Riya", "risk_profile": "standard"},
                situation="Riya needs a refund workflow inspected before optimization.",
                outcome="The workflow graph includes route, retry, recovery, state, latency, and cost evidence.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=orchestration_agent,
        environment=OrchestrationTraceEnvironment(
            framework="langgraph",
            records=TRACE_RECORDS,
            state={"case": {"status": "resolved"}},
        ),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "orchestration_trace_status",
                "list_orchestration_steps",
                "inspect_orchestration_node",
                "inspect_orchestration_edge",
            ],
            "available_tools": [
                "orchestration_trace_status",
                "list_orchestration_steps",
                "inspect_orchestration_node",
                "inspect_orchestration_edge",
            ],
            "required_artifact_types": ["trace"],
            "required_orchestration_trace": [
                "workflow",
                "node",
                "route",
                "handoff",
                "tool",
                "retry",
                "recovered",
                "latency",
                "cost",
                "state",
            ],
            "orchestration_trace_quality": {
                "required_nodes": ["triage_agent", "policy_agent", "refund_tool"],
                "required_step_types": ["workflow", "tool", "retry"],
                "expected_routes": [
                    {"from": "triage_agent", "to": "policy_agent", "type": "handoff"},
                    {"from": "policy_agent", "to": "refund_tool"},
                ],
                "min_retry_count": 1,
                "require_recovered_errors": True,
                "expected_recovered_errors": [{"node": "policy_agent"}],
                "max_total_latency_ms": 150,
                "max_step_latency_ms": 50,
                "max_total_cost": 100,
                "max_error_count": 1,
                "required_terminal_status": "success",
                "expected_state": {"case": {"status": "resolved"}},
            },
            "metric_weights": {
                "orchestration_trace_coverage": 4.0,
                "orchestration_flow_quality": 5.0,
            },
        },
        threshold=0.85,
    )

    trace = report.results[0].metadata["environment_state"]["orchestration_trace"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("signals:", trace["signals"])
    print("retry_count:", trace["summary"]["retry_count"])
    print("recovered_failures:", trace["summary"]["recovered_failures"])
    print("orchestration_trace_coverage:", metrics.get("orchestration_trace_coverage"))
    print("orchestration_flow_quality:", metrics.get("orchestration_flow_quality"))


if __name__ == "__main__":
    asyncio.run(main())
