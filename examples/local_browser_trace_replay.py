"""
Run a local browser/CUA trace replay simulation.

This models the evidence a computer-use agent should produce: DOM snapshots,
screenshots, action replay, console logs, and network requests. No browser
process, cloud service, or model key is required.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    BrowserEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


async def checkout_browser_agent(input):
    return AgentResponse(
        content=(
            "Checkout confirmed with browser trace evidence. I inspected the "
            "page, clicked confirm, and captured browser evidence."
        ),
        tool_calls=[
            {"id": "call_snapshot", "name": "browser_snapshot", "arguments": {}},
            {
                "id": "call_click",
                "name": "browser_click",
                "arguments": {
                    "url": "https://shop.example.com/checkout/done",
                    "action": "click confirm order",
                },
            },
            {"id": "call_console", "name": "browser_console", "arguments": {}},
            {"id": "call_network", "name": "browser_network", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="browser-trace-replay",
        dataset=[
            Persona(
                persona={"name": "Riley", "risk_profile": "standard"},
                situation="Riley needs a checkout confirmation in the browser.",
                outcome="Checkout confirmed with browser trace evidence.",
            )
        ],
    )
    environment = BrowserEnvironment(
        url="https://shop.example.com/checkout",
        allowed_domains=["shop.example.com"],
        snapshots=[
            {
                "id": "checkout",
                "url": "https://shop.example.com/checkout",
                "dom": "<button id='confirm'>Confirm order</button>",
                "screenshot_uri": "file:///fixtures/checkout.png",
            },
            {
                "id": "done",
                "url": "https://shop.example.com/checkout/done",
                "dom": "<main>Order confirmed</main>",
                "screenshot_uri": "file:///fixtures/checkout-done.png",
            },
        ],
        console_logs=[{"level": "info", "message": "checkout page hydrated"}],
        network_log=[
            {"url": "https://shop.example.com/api/checkout", "method": "POST", "status": 200}
        ],
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=checkout_browser_agent,
        environment=environment,
        modality="cua",
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "browser_snapshot",
                "browser_click",
                "browser_console",
                "browser_network",
            ],
            "available_tools": [
                "browser_snapshot",
                "browser_click",
                "browser_console",
                "browser_network",
            ],
            "allowed_domains": ["shop.example.com"],
            "required_artifact_types": ["browser_dom", "screenshot", "trace"],
            "required_browser_trace": ["dom", "screenshot", "action", "console", "network"],
            "success_criteria": ["checkout confirmed"],
        },
        threshold=0.85,
    )

    result = report.results[0]
    browser_state = result.metadata["environment_state"]["browser"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("artifacts:", [artifact.type for artifact in result.artifacts])
    print("action_replay:", browser_state["action_replay"])
    print("browser_trace_coverage:", evaluation.summary["metric_averages"]["browser_trace_coverage"])


if __name__ == "__main__":
    asyncio.run(main())
