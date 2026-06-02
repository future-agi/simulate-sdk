"""
Run a local voice + image simulation and score required media artifacts.

This models multimodal worlds without LiveKit, a browser process, model keys,
or real media generation. Audio and image fixtures are carried as artifacts,
while voice/image tool calls produce normalized events and state updates.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    ImageEnvironment,
    Persona,
    Scenario,
    TestRunner,
    VoiceEnvironment,
    evaluate_agent_report,
)


async def receipt_voice_agent(input):
    return AgentResponse(
        content=(
            "I inspected the receipt image, transcribed the caller, and answered "
            "with low-latency voice output. Receipt support case resolved."
        ),
        tool_calls=[
            {
                "id": "call_image",
                "name": "inspect_image",
                "arguments": {"id": "receipt"},
            },
            {
                "id": "call_stt",
                "name": "transcribe_audio",
                "arguments": {"id": "caller_1"},
            },
            {
                "id": "call_speak",
                "name": "speak",
                "arguments": {
                    "text": "The receipt shows order 123 and the support case is resolved.",
                    "latency_ms": 420,
                },
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="voice-image-receipt-support",
        dataset=[
            Persona(
                persona={"name": "Maya", "risk_profile": "standard"},
                situation="Maya shared a receipt image and a voice note about order 123.",
                outcome="Receipt support case resolved.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=receipt_voice_agent,
        environment=[
            ImageEnvironment(
                {
                    "receipt": {
                        "uri": "file:///fixtures/receipt.png",
                        "description": "Receipt image for order 123",
                        "labels": ["receipt", "order"],
                    }
                }
            ),
            VoiceEnvironment(
                [
                    {
                        "id": "caller_1",
                        "transcript": "Please check the receipt for order 123.",
                        "audio_uri": "file:///fixtures/caller.wav",
                    }
                ],
                sample_rate_hz=24000,
            ),
        ],
        max_turns=1,
        min_turns=1,
        modality="voice",
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["inspect_image", "transcribe_audio", "speak"],
            "available_tools": ["inspect_image", "transcribe_audio", "speak"],
            "required_artifact_types": ["image", "audio"],
            "max_voice_latency_ms": 1000,
            "success_criteria": ["receipt support case resolved"],
        },
    )

    result = report.results[0]
    print("score:", evaluation.score)
    print("artifacts:", [artifact.type for artifact in result.artifacts])
    print("tools:", [call.get("name") for call in result.tool_calls])
    print("state:", result.metadata["environment_state"])


if __name__ == "__main__":
    asyncio.run(main())
