"""
Run a local browser/CUA trace replay from HAR, OpenAI CUA, and Browser Use data.

Requires:
    pip install agent-simulate ai-evaluation

The fixture is deterministic JSON shaped like real exports:
- HAR `log.entries` with response bodies.
- OpenAI Computer Use `computer_call` and `computer_call_output` items.
- Browser Use-style URLs, screenshots, model actions, and action results.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    load_browser_trace_export,
)


def build_browser_trace_export():
    return {
        "provider": "browser_use",
        "urls": ["https://shop.example.com/checkout"],
        "screenshot_paths": ["/tmp/browser-use-checkout.png"],
        "model_actions": [{"click": {"index": 1, "x": 190, "y": 450}}],
        "action_results": [{"success": True}],
        "log": {
            "entries": [
                {
                    "startedDateTime": "2026-06-03T10:00:00Z",
                    "request": {
                        "method": "GET",
                        "url": "https://shop.example.com/api/cart",
                    },
                    "response": {
                        "status": 200,
                        "content": {
                            "mimeType": "application/json",
                            "text": "{\"cart\":\"ready\"}",
                        },
                    },
                }
            ]
        },
        "events": [
            {
                "type": "computer_call",
                "id": "cu_confirm",
                "call_id": "call_confirm",
                "actions": [
                    {"type": "screenshot"},
                    {"type": "click", "button": "left", "x": 190, "y": 450},
                ],
                "pending_safety_checks": [
                    {
                        "id": "sc_prompt_injection",
                        "code": "malicious_instructions",
                        "message": "Hostile page instruction detected.",
                    }
                ],
                "status": "completed",
            },
            {
                "type": "computer_call_output",
                "call_id": "call_confirm",
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "file:///tmp/openai-cua-after.png",
                },
                "current_url": "https://shop.example.com/checkout",
            },
        ],
    }


async def browser_agent(input):
    return AgentResponse(
        content="I inspected the imported trace, confirmed the cart API body, and clicked checkout.",
        tool_calls=[
            {"id": "network", "name": "browser_network", "arguments": {}},
            {"id": "click", "name": "computer_click", "arguments": {"x": 190, "y": 450, "action": "click"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="browser-cua-trace-replay",
        dataset=[
            Persona(
                persona={"name": "Mira", "risk_profile": "standard"},
                situation="Mira needs checkout completed from imported browser/CUA trace evidence.",
                outcome="The agent uses HAR resource bodies and replays the CUA click safely.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=browser_agent,
        environment=load_browser_trace_export(
            build_browser_trace_export(),
            provider="browser_use",
            allowed_domains=["shop.example.com"],
        ),
        max_turns=1,
        min_turns=1,
        modality="cua",
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["browser_network", "computer_click"],
            "available_tools": ["browser_network", "computer_click"],
            "required_artifact_types": ["trace", "screenshot"],
            "required_browser_trace": [
                "har",
                "resource_body",
                "actionability",
                "actionability_timeline",
                "openai_cua_trace",
                "browser_use_trace",
                "action",
                "network",
                "screenshot",
                "prompt_injection_surface",
            ],
            "success_criteria": ["replays the CUA click"],
        },
        threshold=0.85,
    )

    result = report.results[0]
    browser = result.metadata["environment_state"]["browser"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("resource_body:", browser["resource_bodies"][0]["body"])
    print("action_success:", browser["action_replay"][-1]["success"])
    print("browser_trace_coverage:", metrics.get("browser_trace_coverage"))


if __name__ == "__main__":
    asyncio.run(main())
