"""
Run a local voice replay simulation with routing and interruption checks.

This models LiveKit/Pipecat-style realtime evidence without starting a room,
pipeline, model, or media service. The report carries audio artifacts, VAD/STT
events, TTS latency, barge-in handling, call routing, and a voice trace artifact.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    VoiceEnvironment,
    evaluate_agent_report,
)


async def routed_voice_agent(input):
    return AgentResponse(
        content=(
            "Voice support call resolved with replay evidence. I routed the "
            "caller to billing, transcribed the request, responded, and handled "
            "the interruption."
        ),
        tool_calls=[
            {"id": "call_status", "name": "voice_status", "arguments": {}},
            {
                "id": "call_route",
                "name": "route_call",
                "arguments": {"route": "billing", "reason": "billing support request"},
            },
            {
                "id": "call_stt",
                "name": "transcribe_audio",
                "arguments": {"id": "caller_1"},
            },
            {
                "id": "call_tts",
                "name": "speak",
                "arguments": {
                    "text": "I routed you to billing and can help with order 123.",
                    "latency_ms": 420,
                },
            },
            {"id": "call_stop", "name": "stop_speaking", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="voice-replay-routing",
        dataset=[
            Persona(
                persona={"name": "Noah", "risk_profile": "standard"},
                situation="Noah called about a billing issue and interrupted mid-response.",
                outcome="Voice support call resolved with replay evidence.",
            )
        ],
    )
    environment = VoiceEnvironment(
        [
            {
                "id": "caller_1",
                "speaker": "user",
                "transcript": "I need billing help for order 123.",
                "audio_uri": "file:///fixtures/noah.wav",
                "barge_in": True,
                "start_ms": 0,
                "end_ms": 1800,
            }
        ],
        sample_rate_hz=24000,
        latency_profile={"stt": [120, 180], "tts": [420]},
        event_replay=[
            {"name": "vad_start", "timestamp_ms": 0},
            {"name": "stt_partial", "payload": {"transcript": "I need billing"}},
            {"name": "vad_end", "timestamp_ms": 1800},
        ],
        routes={
            "default": {"kind": "agent", "name": "front_desk"},
            "billing": {"kind": "queue", "name": "billing_specialist"},
        },
        initial_route="default",
        allow_interruptions=True,
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=routed_voice_agent,
        environment=environment,
        modality="voice",
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "voice_status",
                "route_call",
                "transcribe_audio",
                "speak",
                "stop_speaking",
            ],
            "available_tools": [
                "voice_status",
                "route_call",
                "transcribe_audio",
                "speak",
                "stop_speaking",
            ],
            "required_artifact_types": ["audio", "trace"],
            "required_voice_trace": [
                "audio",
                "vad",
                "stt",
                "tts",
                "interruption",
                "route",
                "latency",
            ],
            "max_voice_latency_ms": 1000,
            "success_criteria": ["voice support call resolved with replay evidence"],
        },
        threshold=0.85,
    )

    result = report.results[0]
    voice_state = result.metadata["environment_state"]["voice"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("artifacts:", [artifact.type for artifact in result.artifacts])
    print("route_history:", voice_state["route_history"])
    print("voice_trace_coverage:", evaluation.summary["metric_averages"]["voice_trace_coverage"])


if __name__ == "__main__":
    asyncio.run(main())
