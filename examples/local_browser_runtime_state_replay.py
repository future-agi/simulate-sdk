"""
Run a local browser/CUA replay with storage-state and runtime capture hooks.

Requires:
    pip install agent-simulate ai-evaluation

The example simulates a checkout click that mutates cookies, localStorage, and
sessionStorage while recording a page error and resource timing entry.
"""

import asyncio

from fi.simulate import AgentResponse, BrowserEnvironment, Persona, Scenario, TestRunner, evaluate_agent_report


async def browser_agent(input):
    return AgentResponse(
        content="I inspected browser storage/runtime state and confirmed checkout.",
        tool_calls=[
            {"id": "storage", "name": "browser_storage", "arguments": {}},
            {"id": "runtime", "name": "browser_runtime", "arguments": {}},
            {
                "id": "click",
                "name": "browser_click",
                "arguments": {"selector": "#confirm", "action": "click confirm"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="browser-runtime-state",
        dataset=[
            Persona(
                persona={"name": "Mira", "risk_profile": "standard"},
                situation="Mira needs checkout confirmed with browser runtime evidence.",
                outcome="Storage state, runtime events, and performance timing are captured.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=browser_agent,
        environment=BrowserEnvironment(
            url="https://shop.example.com/checkout",
            dom="<button id='confirm'>Confirm</button>",
            screenshot_uri="file:///tmp/checkout.png",
            allowed_domains=["shop.example.com"],
            storage_state={
                "cookies": [{"name": "session", "value": "before", "domain": "shop.example.com", "path": "/"}],
                "origins": [
                    {
                        "origin": "https://shop.example.com",
                        "localStorage": [{"name": "cart_id", "value": "cart_123"}],
                    }
                ],
            },
            actions=[
                {
                    "id": "confirm_runtime_capture",
                    "tool_names": ["browser_click"],
                    "selector": "#confirm",
                    "next_url": "https://shop.example.com/done",
                    "dom": "<main>Done</main>",
                    "cookies": {"checkout_session": "confirmed"},
                    "local_storage": {"https://shop.example.com": {"checkout_status": "confirmed"}},
                    "session_storage": {"https://shop.example.com": {"last_action": "confirm"}},
                    "runtime_events": [
                        {
                            "type": "page_error",
                            "level": "error",
                            "message": "Recoverable hydration mismatch handled.",
                        }
                    ],
                    "performance_entries": [
                        {
                            "name": "https://shop.example.com/api/checkout",
                            "entry_type": "resource",
                            "duration": 120,
                        }
                    ],
                }
            ],
        ),
        max_turns=1,
        min_turns=1,
        modality="cua",
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["browser_storage", "browser_runtime", "browser_click"],
            "available_tools": ["browser_storage", "browser_runtime", "browser_click"],
            "required_artifact_types": ["trace", "screenshot"],
            "required_browser_trace": [
                "storage_state",
                "cookie",
                "local_storage",
                "session_storage",
                "runtime_error",
                "performance_entry",
                "performance_timing",
            ],
            "expected_browser_storage": {
                "cookies": {"checkout_session": "confirmed"},
                "local_storage": {"https://shop.example.com": {"checkout_status": "confirmed"}},
                "session_storage": {"https://shop.example.com": {"last_action": "confirm"}},
            },
            "expected_browser_runtime_events": [
                {"type": "page_error", "message_contains": "hydration mismatch"}
            ],
            "max_browser_performance_duration_ms": 150,
        },
        threshold=0.85,
    )

    result = report.results[0]
    browser = result.metadata["environment_state"]["browser"]
    metrics = evaluation.summary["metric_averages"]
    cookies = {cookie["name"]: cookie["value"] for cookie in browser["storage_state"]["cookies"]}
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("checkout_session:", cookies.get("checkout_session"))
    print("runtime_error_count:", browser["runtime_summary"]["error_count"])
    print("max_duration_ms:", browser["runtime_summary"]["max_duration_ms"])
    print("browser_trace_coverage:", metrics.get("browser_trace_coverage"))
    print("browser_action_outcome:", metrics.get("browser_action_outcome"))
    print("browser_grounding_quality:", metrics.get("browser_grounding_quality"))


if __name__ == "__main__":
    asyncio.run(main())
