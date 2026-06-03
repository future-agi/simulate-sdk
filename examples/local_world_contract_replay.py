"""
Run a local world-contract state-machine replay.

Requires:
    pip install agent-simulate ai-evaluation

Use this pattern when a task is defined by valid world transitions, invariants,
and final state rather than by one framework trace shape.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    WorldContractEnvironment,
    evaluate_agent_report,
)


def refund_world():
    return WorldContractEnvironment(
        name="refund_world",
        actors=[{"id": "support_agent", "role": "agent"}, {"id": "customer", "role": "user"}],
        resources=[{"id": "case"}, {"id": "refund_policy"}],
        initial_state={
            "case": {
                "status": "open",
                "identity_verified": False,
                "policy_checked": False,
                "refund_issued": False,
            }
        },
        transitions=[
            {
                "id": "verify_identity",
                "actor": "support_agent",
                "resource": "case",
                "required": True,
                "effects": {"case.identity_verified": True},
                "postconditions": {"case.identity_verified": True},
                "signals": ["identity", "milestone"],
            },
            {
                "id": "check_policy",
                "actor": "support_agent",
                "resource": "refund_policy",
                "required": True,
                "preconditions": {"case.identity_verified": True},
                "effects": {"case.policy_checked": True},
                "postconditions": {"case.policy_checked": True},
                "signals": ["policy", "milestone"],
            },
            {
                "id": "issue_refund",
                "actor": "support_agent",
                "resource": "case",
                "required": True,
                "preconditions": {"case.identity_verified": True, "case.policy_checked": True},
                "effects": {"case.refund_issued": True, "case.status": "resolved"},
                "postconditions": {"case.refund_issued": True, "case.status": "resolved"},
                "signals": ["tool", "milestone"],
            },
        ],
        invariants=[
            {
                "id": "refund_requires_identity",
                "when": {"case.refund_issued": True},
                "must": {"case.identity_verified": True},
            },
            {
                "id": "refund_requires_policy",
                "when": {"case.refund_issued": True},
                "must": {"case.policy_checked": True},
            },
        ],
        success_conditions=[
            {"id": "refund_resolved", "must": {"case.status": "resolved", "case.refund_issued": True}}
        ],
        policy_gates=[{"id": "identity_gate", "must": {"case.identity_verified": True}}],
        adversarial_surfaces=[{"id": "user_message", "type": "prompt_injection"}],
    )


async def world_agent(input):
    return AgentResponse(
        content="I verified identity, checked policy, issued the refund, and inspected the world contract.",
        tool_calls=[
            {"id": "status", "name": "world_contract_status", "arguments": {}},
            {"id": "identity", "name": "apply_world_transition", "arguments": {"id": "verify_identity"}},
            {"id": "policy", "name": "apply_world_transition", "arguments": {"id": "check_policy"}},
            {"id": "refund", "name": "apply_world_transition", "arguments": {"id": "issue_refund"}},
            {"id": "invariant", "name": "inspect_world_invariant", "arguments": {"id": "refund_requires_identity"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="world-contract-refund",
        dataset=[
            Persona(
                persona={"name": "Riya", "risk_profile": "standard"},
                situation="Riya needs a refund, but the agent must follow world rules before issuing it.",
                outcome="Identity is verified, policy is checked, refund is issued, and the case is resolved.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=world_agent,
        environment=refund_world(),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "world_contract_status",
                "apply_world_transition",
                "inspect_world_invariant",
            ],
            "available_tools": [
                "world_contract_status",
                "list_world_transitions",
                "apply_world_transition",
                "inspect_world_invariant",
            ],
            "required_world_contract": [
                "actor",
                "resource",
                "transition",
                "completed_transition",
                "required_transition",
                "invariant",
                "success_condition",
                "policy",
                "adversarial_surface",
                "state",
                "success",
            ],
            "world_contract_quality": {
                "required_actors": ["support_agent", "customer"],
                "required_resources": ["case", "refund_policy"],
                "required_transitions": [
                    {"id": "verify_identity", "status": "success"},
                    {"id": "check_policy", "status": "success"},
                    {"id": "issue_refund", "status": "success"},
                ],
                "min_completed_transitions": 3,
                "require_all_required_transitions": True,
                "require_all_invariants_pass": True,
                "required_invariants": ["refund_requires_identity", "refund_requires_policy"],
                "required_success_conditions": ["refund_resolved"],
                "max_violation_count": 0,
                "max_forbidden_transitions": 0,
                "required_terminal_status": "success",
                "expected_state": {
                    "case": {
                        "status": "resolved",
                        "identity_verified": True,
                        "policy_checked": True,
                        "refund_issued": True,
                    }
                },
            },
            "metric_weights": {"world_contract_coverage": 4.0, "world_contract_quality": 5.0},
        },
        threshold=0.85,
    )

    world = report.results[0].metadata["environment_state"]["world_contract"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("terminal_status:", world["summary"]["terminal_status"])
    print("completed_required_transition_count:", world["summary"]["completed_required_transition_count"])
    print("world_contract_coverage:", metrics.get("world_contract_coverage"))
    print("world_contract_quality:", metrics.get("world_contract_quality"))


if __name__ == "__main__":
    asyncio.run(main())
