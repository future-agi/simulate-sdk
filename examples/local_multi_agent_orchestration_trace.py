"""
Replay a multi-agent orchestration control loop from framework trace events.

Requires:
    pip install agent-simulate ai-evaluation

This turns framework-level spawn, delegation, communication, aggregation, and
stop decisions into a deterministic orchestration trace contract.
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
        "id": "spawn_policy",
        "name": "spawn policy_agent",
        "node": "coordinator",
        "route_from": "coordinator",
        "route_to": "policy_agent",
        "type": "spawn",
        "latency_ms": 10,
    },
    {
        "id": "delegate_policy",
        "name": "delegate refund policy review",
        "node": "coordinator",
        "delegator": "coordinator",
        "delegate_to": "policy_agent",
        "task": "review refund policy",
        "latency_ms": 12,
    },
    {
        "id": "delegate_retrieval",
        "name": "delegate evidence retrieval",
        "node": "coordinator",
        "delegator": "coordinator",
        "delegate_to": "retrieval_agent",
        "task": "retrieve order evidence",
        "latency_ms": 12,
    },
    {
        "id": "agent_message",
        "name": "message retrieval_agent to policy_agent",
        "node": "retrieval_agent",
        "sender": "retrieval_agent",
        "receiver": "policy_agent",
        "type": "message",
        "latency_ms": 8,
    },
    {
        "id": "consensus",
        "name": "aggregate policy and evidence consensus",
        "node": "coordinator",
        "type": "aggregate",
        "status": "success",
        "usage": {"total_tokens": 64},
    },
    {
        "id": "stop",
        "name": "terminate orchestration after consensus",
        "node": "coordinator",
        "type": "stop",
        "status": "success",
        "state": {"decision": {"status": "approved"}},
    },
]


async def audit_agent(input):
    return AgentResponse(
        content="The multi-agent orchestration spawned specialists, delegated work, aggregated consensus, and stopped.",
        tool_calls=[
            {"id": "status", "name": "orchestration_trace_status", "arguments": {}},
            {"id": "spawn", "name": "list_orchestration_steps", "arguments": {"signal": "spawn"}},
            {
                "id": "delegate",
                "name": "inspect_orchestration_edge",
                "arguments": {"from": "coordinator", "to": "policy_agent"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="multi-agent-orchestration-control",
        dataset=[
            Persona(
                persona={"name": "Anika", "risk_profile": "standard"},
                situation="Anika needs a multi-agent refund decision audited from framework trace events.",
                outcome="The trace proves spawn, delegation, communication, aggregation, stop, and final state.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=audit_agent,
        environment=OrchestrationTraceEnvironment(
            framework="autogen",
            records=TRACE_RECORDS,
            state={"decision": {"status": "approved"}},
        ),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_orchestration_trace": [
                "agent",
                "spawn",
                "delegate",
                "communicate",
                "aggregate",
                "stop",
                "state",
            ],
            "orchestration_trace_quality": {
                "required_nodes": ["coordinator", "policy_agent", "retrieval_agent"],
                "required_step_types": ["spawn", "delegate", "communicate", "aggregate", "stop"],
                "expected_routes": [
                    {"from": "coordinator", "to": "policy_agent", "type": "delegate"},
                    {"from": "coordinator", "to": "retrieval_agent", "type": "delegate"},
                ],
                "min_agent_count": 3,
                "min_spawn_count": 1,
                "min_delegation_count": 2,
                "min_communication_count": 1,
                "require_aggregation": True,
                "require_stop_decision": True,
                "required_terminal_status": "success",
                "expected_state": {"decision": {"status": "approved"}},
            },
        },
        threshold=0.85,
    )

    state = report.results[0].metadata["environment_state"]["orchestration_trace"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("orchestration_summary:", state["summary"])
    print("orchestration_trace_coverage:", metrics.get("orchestration_trace_coverage"))
    print("orchestration_flow_quality:", metrics.get("orchestration_flow_quality"))


if __name__ == "__main__":
    asyncio.run(main())
