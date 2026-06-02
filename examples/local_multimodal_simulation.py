"""
Run a self-contained multimodal/CUA-style simulation without LiveKit, cloud runs,
or model provider keys.

This is the smallest loop for testing a framework adapter:
1. Generate synthetic scenarios locally.
2. Wrap any agent/framework object.
3. Run multi-turn simulation.
4. Keep transcript, messages, tool calls, artifacts, and events for evaluation.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    ScriptedAgentWrapper,
    SimulationArtifact,
    SimulationEvent,
    SyntheticDataGenerator,
    TestRunner,
    wrap_framework,
)


class BrowserUseLikeAgent:
    """Stand-in for Browser Use, Playwright, OpenAI CUA, or another browser agent."""

    def run(self, payload):
        assert payload["modality"] == "cua"
        return {
            "content": "Clicked submit, confirmed the success state, and captured the final screen.",
            "artifacts": [
                {
                    "type": "screenshot",
                    "uri": "file:///tmp/final-screen.png",
                    "mime_type": "image/png",
                    "role": "assistant",
                }
            ],
            "events": [
                {
                    "type": "browser_action",
                    "name": "click",
                    "payload": {"selector": "#submit"},
                }
            ],
        }


async def main():
    scenario = SyntheticDataGenerator().generate(
        "browser checkout support",
        num_personas=2,
        seed=11,
        task="checkout support flow",
    )

    browser_agent = wrap_framework("computer_use", BrowserUseLikeAgent())

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=browser_agent,
        modality="cua",
        max_turns=2,
        artifacts=[
            SimulationArtifact(
                type="browser_dom",
                data="<button id='submit'>Submit</button>",
                mime_type="text/html",
                role="user",
            )
        ],
        events=[
            SimulationEvent(
                type="environment",
                name="browser_fixture_loaded",
                payload={"url": "https://example.test/checkout"},
            )
        ],
    )

    for result in report.results:
        print("=" * 80)
        print(result.transcript)
        print("tool calls:", result.tool_calls)
        print("artifacts:", [artifact.type for artifact in result.artifacts])
        print("events:", [event.type for event in result.events])

    # Deterministic mock agents are useful for regression tests.
    smoke_report = await TestRunner().run_test(
        topic="refund support",
        agent_callback=ScriptedAgentWrapper(
            [
                AgentResponse(
                    content="I can resolve the refund within policy.",
                    memory_updates={"last_policy": "refund"},
                )
            ]
        ),
        max_turns=1,
    )
    print("smoke transcript:", smoke_report.results[0].transcript)


if __name__ == "__main__":
    asyncio.run(main())
