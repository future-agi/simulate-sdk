"""
Run a self-contained agent simulation with local environment adapters.

This models the world around the agent: mocked APIs/tools, browser/CUA state,
files, and a multi-agent handoff room. No cloud service, browser process, model
provider, or LiveKit room is required.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    BrowserEnvironment,
    FileEnvironment,
    MultiAgentRoomEnvironment,
    Scenario,
    Persona,
    ToolMockEnvironment,
    TestRunner,
    evaluate_agent_report,
)


async def checkout_agent(input):
    return AgentResponse(
        content=(
            "First I will search the order, review the checkout page, read the "
            "policy, and hand off to the policy specialist because this needs "
            "a verified resolution. Checkout support case resolved within policy."
        ),
        tool_calls=[
            {
                "id": "call_search",
                "name": "search_order",
                "arguments": {"order_id": "ord_123"},
            },
            {
                "id": "call_browser",
                "name": "browser_click",
                "arguments": {
                    "url": "https://shop.example.com/checkout",
                    "action": "review checkout status",
                },
            },
            {
                "id": "call_policy",
                "name": "read_file",
                "arguments": {"path": "refund-policy.md"},
            },
            {
                "id": "call_handoff",
                "name": "handoff",
                "arguments": {
                    "to": "policy_specialist",
                    "task": "confirm checkout policy",
                },
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="environment-backed-checkout",
        dataset=[
            Persona(
                persona={"name": "Avery", "risk_profile": "standard"},
                situation="Avery needs checkout support for order ord_123.",
                outcome="Checkout support case resolved within policy.",
            )
        ],
    )
    environments = [
        ToolMockEnvironment(
            {
                "search_order": lambda args, ctx: {
                    "content": "Order ord_123 is paid and ready for checkout review.",
                    "result": {"status": "ready"},
                    "state_updates": {"case": {"resolved": True}},
                }
            }
        ),
        BrowserEnvironment(
            url="https://shop.example.com/checkout",
            dom="<button id='review'>Review checkout</button>",
            allowed_domains=["shop.example.com"],
        ),
        FileEnvironment({"refund-policy.md": "Refunds require policy approval."}),
        MultiAgentRoomEnvironment(["support_agent", "policy_specialist"]),
    ]

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=checkout_agent,
        environment=environments,
        max_turns=1,
        modality="cua",
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "search_order",
                "browser_click",
                "read_file",
                "handoff",
            ],
            "available_tools": [
                "search_order",
                "browser_click",
                "read_file",
                "handoff",
            ],
            "allowed_domains": ["shop.example.com"],
            "expected_state": {"case": {"resolved": True}},
            "success_criteria": ["checkout support case resolved"],
        },
    )

    result = report.results[0]
    print("score:", evaluation.score)
    print("tools:", [call.get("name") for call in result.tool_calls])
    print("state:", result.metadata["environment_state"])
    print("events:", [event.type for event in result.events])


if __name__ == "__main__":
    asyncio.run(main())
