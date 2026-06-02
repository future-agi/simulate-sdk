"""
Run a local Playwright trace replay simulation.

This builds a tiny Playwright-style trace.zip fixture with DOM snapshots,
screenshot resources, an action timeline, and a video resource. The browser
environment imports the archive, applies stale-screenshot and layout-shift
perturbations, then evaluates whether the agent refreshes and clicks the shifted
control.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio
import json
import tempfile
import zipfile
from pathlib import Path

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    load_playwright_trace_export,
)


def build_trace_zip(path: Path) -> None:
    records = [
        {
            "type": "frame-snapshot",
            "snapshot": {
                "id": "checkout_before",
                "url": "https://shop.example.com/checkout",
                "html": "<button id='confirm'>Confirm</button>",
                "screenshotSha1": "before.png",
            },
        },
        {
            "type": "before",
            "callId": "call_confirm",
            "apiName": "locator.click",
            "pageUrl": "https://shop.example.com/checkout",
            "params": {
                "selector": "#confirm",
                "boundingBox": {"x": 160, "y": 380, "width": 180, "height": 54},
            },
        },
        {
            "type": "frame-snapshot",
            "snapshot": {
                "id": "checkout_current",
                "url": "https://shop.example.com/checkout",
                "html": "<aside>Banner</aside><button id='confirm'>Confirm</button>",
                "screenshotSha1": "after.png",
            },
        },
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trace.trace", "\n".join(json.dumps(record) for record in records))
        archive.writestr("resources/before.png", b"before")
        archive.writestr("resources/after.png", b"after")
        archive.writestr("resources/checkout.webm", b"video")


async def browser_agent(input):
    return AgentResponse(
        content="I refreshed the stale screenshot and clicked the shifted confirm button.",
        tool_calls=[
            {"id": "refresh", "name": "browser_refresh_snapshot", "arguments": {}},
            {
                "id": "click",
                "name": "computer_click",
                "arguments": {"selector": "#confirm", "action": "locator.click", "x": 190, "y": 475},
            },
        ],
    )


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "checkout-trace.zip"
        build_trace_zip(trace_path)
        scenario = Scenario(
            name="playwright-trace-replay",
            dataset=[
                Persona(
                    persona={"name": "Sam", "risk_profile": "standard"},
                    situation="Sam needs a captured browser checkout trace replayed.",
                    outcome="The shifted confirm button is clicked after stale visual state is refreshed.",
                )
            ],
        )
        environment = load_playwright_trace_export(
            trace_path,
            allowed_domains=["shop.example.com"],
            perturbations=[
                {
                    "id": "banner_shift",
                    "type": "layout_shift",
                    "score": 0.18,
                    "affected_regions": ["call_confirm_target"],
                    "delta": {"y": 70},
                },
                {
                    "id": "stale_before",
                    "type": "stale_screenshot",
                    "snapshot_id": "checkout_before",
                },
            ],
        )
        report = await TestRunner().run_test(
            scenario=scenario,
            agent_callback=browser_agent,
            environment=environment,
            max_turns=1,
            min_turns=1,
            modality="cua",
        )
        evaluation = evaluate_agent_report(
            report,
            config={
                "required_tools": ["browser_refresh_snapshot", "computer_click"],
                "available_tools": ["browser_refresh_snapshot", "computer_click"],
                "required_artifact_types": ["trace", "screenshot", "video"],
                "required_browser_trace": [
                    "playwright_trace",
                    "video",
                    "action",
                    "coordinate_region",
                    "layout_shift",
                    "stale_screenshot",
                    "perturbation",
                ],
                "expected_browser_regions": [
                    {"name": "call_confirm_target", "bounds": [160, 450, 180, 54], "selector": "#confirm"}
                ],
                "expected_browser_perturbations": [
                    {"id": "banner_shift", "type": "layout_shift", "affected_regions": ["call_confirm_target"]},
                    {"id": "stale_before", "type": "stale_screenshot"},
                ],
                "allow_stale_browser_screenshot": False,
                "max_browser_layout_shift_score": 0.1,
                "success_criteria": ["clicked the shifted confirm button"],
            },
            threshold=0.85,
        )

        result = report.results[0]
        browser = result.metadata["environment_state"]["browser"]
        print("score:", evaluation.score)
        print("passed:", evaluation.passed)
        print("snapshot:", browser["snapshot"]["id"])
        print("browser_trace_coverage:", evaluation.summary["metric_averages"]["browser_trace_coverage"])
        print("browser_grounding_quality:", evaluation.summary["metric_averages"]["browser_grounding_quality"])


if __name__ == "__main__":
    asyncio.run(main())
