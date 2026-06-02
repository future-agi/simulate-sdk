"""
Run a local simulation and score the report with ai-evaluation agent metrics.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    ScriptedAgentWrapper,
    SimulationEvent,
    SyntheticDataGenerator,
    TestRunner,
    evaluate_agent_report,
)


async def main():
    scenario = SyntheticDataGenerator().generate(
        "checkout support",
        num_personas=1,
        seed=3,
        task="checkout support case",
    )
    agent = ScriptedAgentWrapper(
        [
            AgentResponse(
                content=(
                    "First I will search the order because I need its status. "
                    "Checkout support case resolved within policy."
                ),
                tool_calls=[
                    {
                        "id": "call_search_order",
                        "name": "search_order",
                        "arguments": {"order_id": "123"},
                    }
                ],
                tool_responses=[
                    {
                        "role": "tool",
                        "tool_call_id": "call_search_order",
                        "content": "checkout support case resolved",
                    }
                ],
                memory_updates={"order_id": "123", "status": "resolved"},
                state={"case": {"resolved": True}},
                events=[
                    SimulationEvent(
                        type="browser_action",
                        name="review_checkout",
                        payload={
                            "url": "https://shop.example.com/checkout",
                            "action": "review checkout status",
                        },
                    )
                ],
            )
        ]
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=agent,
        modality="cua",
        max_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["search_order"],
            "available_tools": ["search_order"],
            "allowed_domains": ["shop.example.com"],
            "memory_allowed_keys": ["order_id", "status"],
            "expected_state": {"case": {"resolved": True}},
            "success_criteria": ["checkout support case resolved"],
        },
    )

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("attached:", report.results[0].evaluation["agent_report"]["case_score"])


if __name__ == "__main__":
    asyncio.run(main())
