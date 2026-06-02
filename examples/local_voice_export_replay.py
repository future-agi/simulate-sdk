"""
Replay a local voice/realtime export with waveform, diarization, and quality checks.

Requires:
    pip install agent-simulate ai-evaluation

This is offline and deterministic. The export mixes LiveKit-style session
events with Pipecat-style frames plus self-contained waveform metadata,
speaker segments, and WebRTC-style quality counters.
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
            "language": "en",
            "is_final": True,
            "timestamp_ms": 160,
        },
        {"event": "agent_state_changed", "old_state": "thinking", "new_state": "speaking"},
        {"event": "overlapping_speech", "overlap_ms": 140, "probability": 0.73},
    ],
    "frames": [
        {"id": "raw_in", "frame_type": "InputAudioRawFrame", "sample_rate": 24000, "num_frames": 4800},
        {"id": "pc_transcript", "frame_type": "TranscriptionFrame", "text": "Billing issue for order 123."},
        {"id": "pc_tts", "frame_type": "OutputAudioRawFrame", "num_frames": 2400},
        {"id": "pc_interrupt", "frame_type": "InterruptionFrame", "timestamp_ms": 1185},
    ],
    "recordings": [
        {
            "id": "caller_wave",
            "speaker": "caller",
            "duration_ms": 1700,
            "sample_rate_hz": 24000,
            "snr_db": 32,
            "mos": 4.3,
            "clipping_ratio": 0.002,
            "jitter_ms": 18,
            "packet_loss_pct": 0.4,
        }
    ],
    "speaker_segments": [
        {"id": "seg_caller", "speaker": "caller", "start_ms": 0, "end_ms": 1700, "confidence": 0.96},
        {"id": "seg_agent", "speaker": "agent", "start_ms": 760, "end_ms": 1220, "confidence": 0.93},
    ],
    "perceptual_metrics": {
        "overall": {
            "snr_db": 32,
            "mos": 4.3,
            "clipping_ratio": 0.002,
            "jitter_ms": 18,
            "packet_loss_pct": 0.4,
        }
    },
}


async def export_replay_agent(input):
    return AgentResponse(
        content="Voice export replay handled with billing route and quality evidence.",
        tool_calls=[
            {"id": "route", "name": "route_call", "arguments": {"route": "billing"}},
            {"id": "stt", "name": "transcribe_audio", "arguments": {"id": "caller_1"}},
            {
                "id": "tts",
                "name": "speak",
                "arguments": {"text": "I can help with billing for order 123.", "latency_ms": 420},
            },
            {"id": "stop", "name": "stop_speaking", "arguments": {}},
            {"id": "status", "name": "voice_status", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="voice-export-replay",
        dataset=[
            Persona(
                persona={"name": "Noah", "risk_profile": "standard"},
                situation="Noah called about billing; the replay includes audio, speakers, and quality counters.",
                outcome="The call is routed, transcribed, answered, and passes voice export quality gates.",
            )
        ],
    )
    environment = load_voice_export(
        VOICE_EXPORT,
        framework="livekit",
        sample_rate_hz=24000,
        routes={
            "default": {"kind": "agent", "name": "front_desk"},
            "billing": {"kind": "queue", "name": "billing_specialist"},
        },
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=export_replay_agent,
        environment=environment,
        modality="voice",
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["route_call", "transcribe_audio", "speak", "stop_speaking", "voice_status"],
            "available_tools": ["route_call", "transcribe_audio", "speak", "stop_speaking", "voice_status"],
            "required_artifact_types": ["audio", "trace"],
            "expected_voice_route": "billing",
            "expected_voice_transcript_contains": ["order 123"],
            "required_voice_frame_types": ["InputAudioRawFrame", "TranscriptionFrame", "OutputAudioRawFrame"],
            "required_voice_speakers": ["caller", "agent"],
            "min_voice_snr_db": 25,
            "min_voice_mos": 4.0,
            "max_voice_clipping_ratio": 0.01,
            "max_voice_jitter_ms": 30,
            "max_voice_packet_loss_pct": 1.0,
            "required_voice_trace": [
                "livekit_export",
                "frame",
                "waveform",
                "diarization",
                "perceptual",
                "snr",
                "mos",
                "clipping",
                "jitter",
                "packet_loss",
            ],
        },
        threshold=0.9,
    )

    voice_state = report.results[0].metadata["environment_state"]["voice"]
    metrics = evaluation.summary["metric_averages"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("route_history:", voice_state["route_history"])
    print("speakers:", sorted({segment["speaker"] for segment in voice_state["diarization"]}))
    print("quality:", voice_state["perceptual_metrics"]["overall"])
    print("voice_trace_coverage:", metrics["voice_trace_coverage"])
    print("voice_interaction_quality:", metrics["voice_interaction_quality"])


if __name__ == "__main__":
    asyncio.run(main())
