"""
Replay WebRTC getStats-style voice quality evidence locally.

This models LiveKit/WebRTC RTP, track, codec, audio-level, jitter, and packet
loss evidence without starting a room or calling an external service.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    load_voice_export,
)


VOICE_EXPORT = {
    "framework": "livekit",
    "events": [
        {
            "id": "caller_1",
            "event": "user_input_transcribed",
            "transcript": "Billing issue for order 123.",
            "speaker_id": "caller",
            "is_final": True,
            "timestamp_ms": 180,
        }
    ],
    "webrtc_stats": [
        {
            "id": "inbound_audio_1",
            "type": "inbound-rtp",
            "kind": "audio",
            "trackIdentifier": "caller-track",
            "codecId": "codec_opus",
            "packetsReceived": 1000,
            "packetsLost": 5,
            "jitter": 0.012,
            "audioLevel": 0.18,
            "totalAudioEnergy": 4.2,
        },
        {
            "id": "remote_inbound_audio_1",
            "type": "remote-inbound-rtp",
            "kind": "audio",
            "fractionLost": 0.004,
            "jitter": 0.006,
        },
        {
            "id": "codec_opus",
            "type": "codec",
            "mimeType": "audio/opus",
            "payloadType": 111,
        },
    ],
    "speaker_segments": [
        {"id": "seg_caller", "speaker": "caller", "start_ms": 0, "end_ms": 900},
        {"id": "seg_agent", "speaker": "agent", "start_ms": 940, "end_ms": 1300},
    ],
}


async def webrtc_stats_agent(input):
    return AgentResponse(
        content="Voice call routed after inspecting WebRTC stats.",
        tool_calls=[
            {"id": "route", "name": "route_call", "arguments": {"route": "billing"}},
            {"id": "stt", "name": "transcribe_audio", "arguments": {"id": "caller_1"}},
            {"id": "status", "name": "voice_status", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="voice-webrtc-stats-replay",
        dataset=[
            Persona(
                persona={"name": "Noah", "risk_profile": "standard"},
                situation="Noah called about billing; the replay includes WebRTC quality stats.",
                outcome="The call is routed, transcribed, and passes WebRTC quality gates.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=webrtc_stats_agent,
        environment=load_voice_export(
            VOICE_EXPORT,
            framework="livekit",
            routes={"default": {"agent": "support"}, "billing": {"agent": "billing"}},
        ),
        modality="voice",
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["route_call", "transcribe_audio", "voice_status"],
            "available_tools": ["route_call", "transcribe_audio", "voice_status"],
            "required_artifact_types": ["trace"],
            "expected_voice_route": "billing",
            "expected_voice_transcript_contains": ["order 123"],
            "required_voice_speakers": ["caller", "agent"],
            "max_voice_jitter_ms": 20,
            "max_voice_packet_loss_pct": 1.0,
            "required_voice_trace": [
                "livekit_export",
                "webrtc",
                "rtp",
                "track",
                "codec",
                "audio_level",
                "jitter",
                "packet_loss",
                "diarization",
            ],
        },
        threshold=0.9,
    )

    voice_state = report.results[0].metadata["environment_state"]["voice"]
    inbound = voice_state["webrtc_stats"][0]
    metrics = evaluation.summary["metric_averages"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("track_id:", inbound["track_id"])
    print("jitter_ms:", inbound["jitter_ms"])
    print("packet_loss_pct:", inbound["packet_loss_pct"])
    print("audio_level:", inbound["audio_level"])
    print("quality:", voice_state["perceptual_metrics"]["overall"])
    print("voice_trace_coverage:", metrics["voice_trace_coverage"])
    print("voice_interaction_quality:", metrics["voice_interaction_quality"])


if __name__ == "__main__":
    asyncio.run(main())
