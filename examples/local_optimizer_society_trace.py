"""
Replay an optimizer society trace locally.

Requires:
    pip install agent-simulate ai-evaluation

This audits optimizer deliberation itself: roles, proposals, diagnostics,
role-credit, critique, synthesis, steward simplification, and best-candidate
selection are carried as a trace artifact and scored locally. Governance checks
cover role diversity, mediator review, contract gates, rollback, and locality.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    OptimizerTraceEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    normalize_optimizer_society_trace,
)


TRACE = normalize_optimizer_society_trace(
    name="role-graph-optimizer",
    optimizer="SocietyAgentOptimizer",
    roles=[
        {"name": "sutradhara", "proposal_kind": "specialist", "archetype": "orchestrator"},
        {"name": "vidura", "proposal_kind": "adversary", "archetype": "prudent_critic"},
        {"name": "sangha", "proposal_kind": "coverage_synthesis", "archetype": "collective_synthesis"},
        {"name": "dharma_steward", "proposal_kind": "steward", "archetype": "minimal_process_guardian"},
    ],
    proposals=[
        {"candidate_id": "seed", "role": "seed", "round": 0, "score": 0.2, "patch": {}},
        {
            "candidate_id": "handoff_patch",
            "role": "sutradhara",
            "round": 1,
            "score": 0.55,
            "patch": {"multi_agent.handoff.contract": "explicit_policy"},
            "role_kind": "specialist",
            "role_archetype": "orchestrator",
        },
        {
            "candidate_id": "security_patch",
            "role": "vidura",
            "round": 1,
            "score": 0.72,
            "patch": {"security.adversarial_review": "red_team"},
            "role_kind": "adversary",
            "role_archetype": "prudent_critic",
        },
        {
            "candidate_id": "synthesis_patch",
            "role": "sangha",
            "round": 2,
            "score": 1.0,
            "patch": {
                "multi_agent.handoff.contract": "explicit_policy",
                "security.adversarial_review": "red_team",
            },
            "role_kind": "coverage_synthesis",
            "role_archetype": "collective_synthesis",
        },
        {
            "candidate_id": "steward_patch",
            "role": "dharma_steward",
            "round": 3,
            "score": 0.97,
            "patch": {"multi_agent.handoff.contract": "explicit_policy"},
            "role_kind": "steward",
            "role_archetype": "minimal_process_guardian",
        },
    ],
    rounds=[{"round": 1}, {"round": 2}, {"round": 3}],
    diagnostics=[{"component": "multi_agent", "failure_mode": "coordination_failure"}],
    search_paths=["multi_agent.handoff.contract", "security.adversarial_review"],
    governance={
        "checks": [
            {"name": "role_diversity", "passed": True},
            {"name": "mediator_review", "passed": True},
            {"name": "contract_gate", "passed": True},
            {"name": "rollback_check", "passed": True},
            {"name": "search_locality", "passed": True},
        ]
    },
    best_candidate_id="synthesis_patch",
    final_score=1.0,
)


async def trace_auditor(input):
    return AgentResponse(
        content="Optimizer society trace inspected with role credit, critique, synthesis, and steward evidence.",
        tool_calls=[
            {"id": "status", "name": "optimizer_trace_status", "arguments": {}},
            {"id": "governance", "name": "inspect_optimizer_governance", "arguments": {}},
            {"id": "role", "name": "inspect_optimizer_role", "arguments": {"role": "sangha"}},
            {
                "id": "proposal",
                "name": "list_optimizer_proposals",
                "arguments": {"role": "sangha", "min_score": 0.9},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="optimizer-society-trace",
        dataset=[
            Persona(
                persona={"name": "Mira", "risk_profile": "standard"},
                situation="Mira needs the optimizer deliberation audited before promoting a multi-agent strategy.",
                outcome="The trace preserves role, proposal, credit, critique, synthesis, and steward evidence.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=trace_auditor,
        environment=OptimizerTraceEnvironment(TRACE),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_optimizer_trace": [
                "optimizer_trace",
                "role",
                "role_graph",
                "proposal",
                "evaluation",
                "score",
                "credit",
                "diagnostic",
                "search_path",
                "critique",
                "synthesis",
                "steward",
                "governance",
                "role_diversity",
                "mediator_review",
                "contract_gate",
                "rollback_check",
                "search_locality",
                "best_candidate",
            ],
            "optimizer_trace_quality": {
                "required_roles": ["sutradhara", "vidura", "sangha", "dharma_steward"],
                "min_role_count": 4,
                "min_proposal_count": 5,
                "min_round_count": 3,
                "min_credit_entries": 4,
                "required_archetypes": ["collective_synthesis", "prudent_critic"],
                "required_search_paths": ["multi_agent.handoff.contract"],
                "required_governance_signals": ["role_diversity", "mediator_review", "contract_gate", "rollback_check", "search_locality"],
                "min_governance_checks": 5,
                "min_governance_pass_rate": 1.0,
                "min_best_score": 0.99,
                "required_best_role": "sangha",
                "require_role_graph": True,
                "require_diagnostics": True,
                "require_critique": True,
                "require_synthesis": True,
                "require_steward": True,
                "require_governance": True,
                "require_role_diversity": True,
                "require_mediator": True,
                "require_contract_gate": True,
                "require_rollback": True,
                "require_locality": True,
                "max_duplicate_candidate_count": 0,
            },
        },
        threshold=0.9,
    )
    state = report.results[0].metadata["environment_state"]["optimizer_society_trace"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("summary:", state["summary"])
    print("optimizer_trace_coverage:", metrics.get("optimizer_trace_coverage"))
    print("optimizer_trace_quality:", metrics.get("optimizer_trace_quality"))


if __name__ == "__main__":
    asyncio.run(main())
