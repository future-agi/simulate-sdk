"""
Generate a self-contained tool task, run it locally, and score it.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import AgentResponse, SyntheticDataGenerator, TestRunner, evaluate_agent_report


async def main():
    bundle = SyntheticDataGenerator().generate_tool_task(
        "order fulfillment",
        seed=8,
        target_status="shipped",
    )

    async def agent(input):
        task = input.persona["tool_task"]
        return AgentResponse(
            content="Order 123 has status shipped in the simulated system.",
            tool_calls=[
                {
                    "id": "call_update_order",
                    "name": task["tool"],
                    "arguments": task["arguments"],
                }
            ],
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
        threshold=0.85,
    )

    print("scenario:", bundle.scenario.name)
    print("tool:", bundle.tool_name)
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("metrics:", evaluation.summary["metric_averages"])


if __name__ == "__main__":
    asyncio.run(main())
