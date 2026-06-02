"""
Run a local autonomy-loop trace simulation.

This models the scaffold evidence an autonomous agent should produce: observe,
orient, plan, act, verify, reflect, memory, feedback, skill-library updates, and
quality checks for plan/verifier/reflection/memory/skill/stop behavior. No cloud
service, model key, browser, or media runtime is required.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    AutonomyLoopEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


async def autonomy_loop_agent(input):
    return AgentResponse(
        content=(
            "The autonomous support case is resolved with verified loop evidence. "
            "I observed context, oriented around policy, planned, acted, verified, "
            "reflected, wrote memory, and stored a reusable skill."
        ),
        tool_calls=[
            {
                "id": "observe",
                "name": "record_observation",
                "arguments": {"signals": ["order 123", "refund request", "policy gate"]},
            },
            {
                "id": "orient",
                "name": "orient_strategy",
                "arguments": {"strategy": "use policy and order evidence before final answer"},
            },
            {
                "id": "plan",
                "name": "propose_plan",
                "arguments": {"steps": ["lookup order", "check policy", "verify result"]},
            },
            {
                "id": "act",
                "name": "record_action",
                "arguments": {"action": "lookup order and check refund policy"},
            },
            {
                "id": "verify",
                "name": "verify_outcome",
                "arguments": {
                    "passed": True,
                    "checks": ["order exists", "policy allows refund"],
                    "should_stop": True,
                },
            },
            {
                "id": "reflect",
                "name": "reflect",
                "arguments": {"lesson": "policy gates should precede irreversible actions"},
            },
            {
                "id": "memory",
                "name": "write_memory",
                "arguments": {"order_id": "123", "resolution": "refund eligible"},
            },
            {
                "id": "skill",
                "name": "store_skill",
                "arguments": {
                    "name": "refund_policy_resolution",
                    "steps": ["observe", "orient", "plan", "act", "verify", "reflect"],
                },
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="autonomy-loop-trace",
        dataset=[
            Persona(
                persona={"name": "Isha", "risk_profile": "standard"},
                situation="Isha needs a refund decision that follows policy.",
                outcome="The autonomous support case is resolved with verified loop evidence.",
            )
        ],
    )
    environment = AutonomyLoopEnvironment(
        goal="Resolve a support case with explicit monitor-control evidence.",
        feedback={
            "verify": {"score": 1.0, "source": "policy_check"},
            "reflect": {"lesson_quality": "usable"},
        },
        prior_memory={"refund_cases": "ask for order id and verify policy"},
        policy={"irreversible_actions_require_verification": True},
        expected_plan={"required_steps": ["lookup", "policy", "verify"], "min_steps": 3},
        expected_verification={
            "required_checks": ["order exists", "policy allows refund"],
            "passed_required": True,
            "min_score": 1.0,
        },
        expected_reflection={"required_terms": ["policy", "irreversible"], "min_length": 20},
        expected_memory={"required_keys": ["order_id", "resolution"]},
        expected_skills=[
            {
                "name": "refund_policy_resolution",
                "required_steps": ["observe", "verify", "reflect"],
            }
        ],
        expected_stop={"should_stop": True},
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=autonomy_loop_agent,
        environment=environment,
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "record_observation",
                "orient_strategy",
                "propose_plan",
                "record_action",
                "verify_outcome",
                "reflect",
                "write_memory",
                "store_skill",
            ],
            "available_tools": [
                "record_observation",
                "orient_strategy",
                "propose_plan",
                "record_action",
                "verify_outcome",
                "reflect",
                "write_memory",
                "store_skill",
                "autonomy_status",
            ],
            "required_artifact_types": ["trace"],
            "required_autonomy_loop": [
                "observe",
                "orient",
                "plan",
                "act",
                "verify",
                "reflect",
                "memory",
                "feedback",
                "skill",
                "policy",
            ],
            "expected_autonomy_plan": {"required_steps": ["lookup", "policy", "verify"], "min_steps": 3},
            "expected_autonomy_verification": {
                "required_checks": ["order exists", "policy allows refund"],
                "passed_required": True,
                "min_score": 1.0,
            },
            "expected_autonomy_reflection": {"required_terms": ["policy", "irreversible"], "min_length": 20},
            "expected_autonomy_memory": {"required_keys": ["order_id", "resolution"]},
            "expected_autonomy_skills": [
                {
                    "name": "refund_policy_resolution",
                    "required_steps": ["observe", "verify", "reflect"],
                }
            ],
            "expected_autonomy_stop": {"should_stop": True},
            "success_criteria": ["autonomous support case is resolved with verified loop evidence"],
        },
        threshold=0.85,
    )

    result = report.results[0]
    autonomy_state = result.metadata["environment_state"]["autonomy_loop"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("stages_observed:", autonomy_state["stages_observed"])
    print("skills:", sorted(autonomy_state["skills"].keys()))
    print("autonomy_loop_coverage:", evaluation.summary["metric_averages"]["autonomy_loop_coverage"])
    print("autonomy_loop_quality:", evaluation.summary["metric_averages"]["autonomy_loop_quality"])


if __name__ == "__main__":
    asyncio.run(main())
