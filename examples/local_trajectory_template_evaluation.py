"""
Run a generated trajectory-template task locally.

Requires:
    pip install agent-simulate ai-evaluation

The synthetic bundle includes the scenario, tool schemas, mocked API
environment, image artifact fixture, and ai-evaluation config. No model keys,
cloud services, LiveKit room, browser runtime, or real media files are needed.
"""

import asyncio

from fi.simulate import AgentResponse, SimulationEvent, SyntheticDataGenerator, TestRunner
from fi.simulate.evaluation import evaluate_agent_report


async def main():
    bundle = SyntheticDataGenerator().generate_trajectory_template_task(
        "refund trajectory",
        seed=31,
    )

    async def agent(input):
        task = input.persona["trajectory_template_task"]
        return AgentResponse(
            content=(
                "Approval confirmed. Refund approved for ord_123 within policy. "
                "Receipt total is 19.99."
            ),
            tool_calls=[
                {
                    "id": "call_lookup_order",
                    "name": task["lookup_tool"],
                    "arguments": task["lookup_arguments"],
                },
                {
                    "id": "call_issue_refund",
                    "name": task["action_tool"],
                    "arguments": task["action_arguments"],
                },
            ],
            artifacts=bundle.make_artifacts(),
            events=[
                SimulationEvent(
                    type="browser_action",
                    name="navigate",
                    payload={"action": "navigate", "url": task["browser_url"]},
                )
            ],
            memory_updates={
                "order_id": "ord_123",
                "resolution": "refund approved",
            },
        )

    report = await TestRunner().run_test(
        scenario=bundle.scenario,
        agent_callback=agent,
        environment=bundle.make_environment(),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config=bundle.agent_report_config,
        threshold=0.9,
    )
    metrics = evaluation.summary["metric_averages"]

    print("scenario:", bundle.scenario.name)
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("agent_goal_accuracy:", metrics["agent_goal_accuracy"])
    print("tool_call_f1:", metrics["tool_call_f1"])
    print("policy_adherence:", metrics["policy_adherence"])
    print("memory_correctness:", metrics["memory_correctness"])
    print("multimodal_faithfulness:", metrics["multimodal_faithfulness"])


if __name__ == "__main__":
    asyncio.run(main())
