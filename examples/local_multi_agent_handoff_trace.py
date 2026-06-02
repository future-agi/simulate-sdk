"""
Run a local multi-agent handoff trace simulation.

This models team evidence an orchestration framework should produce: roles,
handoff contracts, delegated work, room messages, review requests, final
reconciliation, and contract-quality checks. No agent framework, cloud service,
or model key is required.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    MultiAgentRoomEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


async def coordinated_support_agent(input):
    return AgentResponse(
        content=(
            "Refund decision resolved through multi-agent coordination. I handed "
            "off policy review, requested QA review, reconciled the decision, "
            "and preserved the handoff trace."
        ),
        tool_calls=[
            {
                "id": "handoff",
                "name": "handoff",
                "arguments": {
                    "to": "policy_specialist",
                    "task": "Check refund eligibility for order 123.",
                    "context": {"order_id": "123", "policy_version": "v2"},
                    "reason": "Refund policy expertise required.",
                },
            },
            {
                "id": "message",
                "name": "send_room_message",
                "arguments": {
                    "to": "room",
                    "message": "Policy specialist is checking order 123 refund eligibility.",
                },
            },
            {
                "id": "review",
                "name": "request_review",
                "arguments": {
                    "reviewer": "qa_reviewer",
                    "target": "refund decision",
                    "criteria": ["policy alignment", "customer-safe wording"],
                },
            },
            {
                "id": "reconcile",
                "name": "reconcile",
                "arguments": {
                    "summary": "Refund is eligible after policy review and QA check.",
                    "accepted_source": "policy_specialist",
                    "conflicts": [],
                },
            },
            {"id": "status", "name": "room_status", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="multi-agent-handoff-trace",
        dataset=[
            Persona(
                persona={"name": "Mira", "risk_profile": "standard"},
                situation="Mira needs a refund decision that requires specialist review.",
                outcome="Refund decision resolved through multi-agent coordination.",
            )
        ],
    )
    environment = MultiAgentRoomEnvironment(
        {
            "support_agent": {"role": "frontline", "goal": "coordinate resolution"},
            "policy_specialist": {"role": "policy", "goal": "verify refund eligibility"},
            "qa_reviewer": {"role": "quality", "goal": "review final customer response"},
        },
        handoff_contracts={
            "policy_specialist": {
                "required_output": "eligibility decision with cited policy",
                "sla_turns": 1,
                "require_reason": True,
                "required_context_keys": ["order_id", "policy_version"],
                "required_task_terms": ["refund", "eligibility"],
            },
            "qa_reviewer": {
                "required_output": "customer-safe final answer review",
                "sla_turns": 1,
            },
        },
        expected_handoffs=[
            {
                "to": "policy_specialist",
                "task_contains": ["refund", "eligibility"],
                "reason_contains": ["policy"],
                "context_keys": ["order_id", "policy_version"],
                "contract_matched": True,
            }
        ],
        expected_reviews=[
            {
                "reviewer": "qa_reviewer",
                "target_contains": ["refund"],
                "criteria": ["policy alignment", "customer-safe wording"],
            }
        ],
        expected_reconciliation={
            "accepted_source": "policy_specialist",
            "summary_contains": ["eligible"],
            "conflicts_empty": True,
        },
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=coordinated_support_agent,
        environment=environment,
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "handoff",
                "send_room_message",
                "request_review",
                "reconcile",
                "room_status",
            ],
            "available_tools": [
                "handoff",
                "send_room_message",
                "request_review",
                "reconcile",
                "room_status",
            ],
            "required_artifact_types": ["trace"],
            "required_multi_agent_trace": [
                "role",
                "contract",
                "handoff",
                "message",
                "review",
                "reconciliation",
            ],
            "required_multi_agent_roles": ["support_agent", "policy_specialist", "qa_reviewer"],
            "expected_multi_agent_handoffs": [
                {
                    "to": "policy_specialist",
                    "task_contains": ["refund", "eligibility"],
                    "reason_contains": ["policy"],
                    "context_keys": ["order_id", "policy_version"],
                    "contract_matched": True,
                }
            ],
            "expected_multi_agent_reviews": [
                {
                    "reviewer": "qa_reviewer",
                    "target_contains": ["refund"],
                    "criteria": ["policy alignment", "customer-safe wording"],
                }
            ],
            "expected_multi_agent_reconciliation": {
                "accepted_source": "policy_specialist",
                "summary_contains": ["eligible"],
                "conflicts_empty": True,
            },
            "success_criteria": ["refund decision resolved through multi-agent coordination"],
        },
        threshold=0.85,
    )

    result = report.results[0]
    room_state = result.metadata["environment_state"]["multi_agent"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("handoffs:", room_state["handoffs"])
    print("reviews:", room_state["reviews"])
    print("multi_agent_trace_coverage:", evaluation.summary["metric_averages"]["multi_agent_trace_coverage"])
    print(
        "multi_agent_coordination_quality:",
        evaluation.summary["metric_averages"]["multi_agent_coordination_quality"],
    )


if __name__ == "__main__":
    asyncio.run(main())
