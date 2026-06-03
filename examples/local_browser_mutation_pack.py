"""
Run a structured browser mutation-pack resilience check locally.

Requires:
    pip install agent-simulate ai-evaluation

Use this pattern for stale selectors, DOM drift, storage drift, runtime faults,
network latency, overlays, and actionability changes that browser/CUA agents
must detect and handle.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    BrowserEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    normalize_browser_mutation_pack,
)


MUTATION_PACK = normalize_browser_mutation_pack(
    mutations=[
        {
            "id": "confirm_selector_drift",
            "type": "selector_alias",
            "url": "https://shop.example.com/checkout",
            "selector": "#confirm",
            "alternate_selector": "#confirm-now",
            "old_text": "id='confirm'",
            "new_text": "id='confirm-now'",
            "next_url": "https://shop.example.com/done",
            "success_dom": "<main><h1>Done</h1><p>Order confirmed.</p></main>",
            "success_state_updates": {"checkout": {"status": "confirmed"}},
        },
        {
            "id": "cart_storage_drift",
            "type": "storage_drift",
            "local_storage": {
                "https://shop.example.com": {"cart_version": "mutated"}
            },
        },
        {
            "id": "hydration_runtime_warning",
            "type": "runtime_error",
            "message": "Recoverable hydration warning after DOM mutation.",
        },
        {
            "id": "checkout_api_latency",
            "type": "network_latency",
            "request_url": "https://shop.example.com/api/checkout",
            "latency_ms": 240,
        },
    ],
    url="https://shop.example.com/checkout",
)


async def resilient_agent(input):
    return AgentResponse(
        content=(
            "I detected browser mutations, refreshed the snapshot, rechecked storage "
            "and runtime state, then used the fallback selector."
        ),
        tool_calls=[
            {"id": "mutations", "name": "browser_mutations", "arguments": {}},
            {"id": "refresh", "name": "browser_refresh_snapshot", "arguments": {}},
            {"id": "storage", "name": "browser_storage", "arguments": {}},
            {"id": "runtime", "name": "browser_runtime", "arguments": {}},
            {
                "id": "fallback_click",
                "name": "browser_click",
                "arguments": {"selector": "#confirm-now", "action": "click confirm"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="browser-mutation-pack",
        dataset=[
            Persona(
                persona={"name": "Mira", "channel": "browser"},
                situation="Mira needs checkout completed despite a mutated browser world.",
                outcome="The agent detects mutations, avoids the stale selector, and confirms checkout.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=resilient_agent,
        environment=BrowserEnvironment(
            url="https://shop.example.com/checkout",
            dom="<main><button id='confirm'>Confirm</button></main>",
            allowed_domains=["shop.example.com"],
            mutation_pack=MUTATION_PACK,
        ),
        max_turns=1,
        min_turns=1,
        modality="cua",
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "browser_mutations",
                "browser_refresh_snapshot",
                "browser_storage",
                "browser_runtime",
                "browser_click",
            ],
            "required_browser_trace": [
                "browser_mutation_pack",
                "dom_mutation",
                "storage_state",
                "runtime_event",
                "performance_timing",
                "actionability",
            ],
            "required_browser_mutations": [
                "confirm_selector_drift",
                "cart_storage_drift",
                "hydration_runtime_warning",
                "checkout_api_latency",
            ],
            "browser_mutation_resilience": {
                "required_types": [
                    "selector_alias",
                    "storage_drift",
                    "runtime_error",
                    "network_latency",
                ],
                "required_mitigations": [
                    "browser_mutations",
                    "browser_refresh_snapshot",
                    "browser_storage",
                    "browser_runtime",
                    "selector_fallback",
                ],
                "expected_actions": [
                    {
                        "selector": "#confirm-now",
                        "success": True,
                        "mutation_id": "confirm_selector_drift",
                    }
                ],
                "expected_storage": {
                    "local_storage": {
                        "https://shop.example.com": {"cart_version": "mutated"}
                    }
                },
                "max_runtime_errors": 1,
            },
            "expected_browser_state": {"checkout.status": "confirmed"},
            "metric_weights": {"browser_mutation_resilience": 5.0, "browser_action_outcome": 2.0},
        },
        threshold=0.85,
    )

    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("mutation_count:", MUTATION_PACK["summary"]["mutation_count"])
    print("browser_mutation_resilience:", metrics.get("browser_mutation_resilience"))
    print("browser_action_outcome:", metrics.get("browser_action_outcome"))


if __name__ == "__main__":
    asyncio.run(main())
