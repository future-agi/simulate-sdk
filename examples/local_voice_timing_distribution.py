"""
Run a local voice replay with timing-distribution evidence.

This models LiveKit/Pipecat-style stage metrics without starting a room,
pipeline, model, or media service. The trace carries VAD, end-of-utterance,
STT, LLM, TTS, and full-turn timing samples plus computed p95 summaries.

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


TIMING_DISTRIBUTION = {
    "stage_order": ["vad", "eou", "stt", "llm", "tts", "turn"],
    "stages": {
        "vad": {"samples_ms": [18, 20, 22], "source": "vad_metrics"},
        "eou": {"samples_ms": [95, 105, 110], "source": "eou_metrics"},
        "stt": {"samples_ms": [170, 190, 210], "source": "stt_metrics"},
        "llm": {"samples_ms": [240, 260, 280], "source": "llm_metrics"},
        "tts": {"samples_ms": [280, 300, 320], "source": "tts_metrics"},
        "turn": {"samples_ms": [780, 820, 840, 860], "source": "session_metrics"},
    },
}


async def timing_agent(input):
    return AgentResponse(
        content="Voice call handled with inspected timing evidence.",
        tool_calls=[
            {"id": "timing", "name": "voice_timing", "arguments": {}},
            {"id": "stt", "name": "transcribe_audio", "arguments": {"id": "caller_1"}},
            {
                "id": "tts",
                "name": "speak",
                "arguments": {"text": "I can help with billing for order 123."},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="voice-timing-distribution",
        dataset=[
            Persona(
                persona={"name": "Noah", "risk_profile": "standard"},
                situation="Noah calls about billing for order 123.",
                outcome="The call is transcribed and answered with timing evidence.",
            )
        ],
    )
    environment = VoiceEnvironment(
        [
            {
                "id": "caller_1",
                "speaker": "caller",
                "transcript": "Billing issue for order 123.",
                "audio_uri": "file:///fixtures/noah.wav",
                "start_ms": 0,
                "end_ms": 1700,
            }
        ],
        sample_rate_hz=24000,
        latency_profile={"stt": [170, 190, 210], "tts": [280, 300, 320]},
        timing_distribution=TIMING_DISTRIBUTION,
        frame_replay=[
            {"frame_type": "UserStartedSpeakingFrame", "timestamp_ms": 20, "vad_ms": 20},
            {"frame_type": "UserStoppedSpeakingFrame", "timestamp_ms": 1700, "eou_delay_ms": 105},
            {"frame_type": "TranscriptionFrame", "timestamp_ms": 1900, "stt_latency_ms": 190},
            {"frame_type": "TTSStartedFrame", "timestamp_ms": 2460, "ttft_ms": 300},
        ],
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=timing_agent,
        environment=environment,
        modality="voice",
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["voice_timing", "transcribe_audio", "speak"],
            "available_tools": ["voice_timing", "transcribe_audio", "speak"],
            "required_artifact_types": ["audio", "trace"],
            "required_voice_trace": [
                "timing_distribution",
                "timing_stage",
                "vad",
                "eou",
                "stt",
                "llm",
                "tts",
                "turn",
                "latency",
            ],
            "voice_timing_distribution": {
                "required_stages": ["vad", "eou", "stt", "llm", "tts", "turn"],
                "required_order": ["vad", "eou", "stt", "llm", "tts", "turn"],
                "min_samples_per_stage": 3,
                "max_stage_p95_ms": {
                    "vad": 24,
                    "eou": 112,
                    "stt": 212,
                    "llm": 282,
                    "tts": 322,
                    "turn": 870,
                },
                "max_turn_p95_ms": 870,
            },
            "metric_weights": {"voice_timing_distribution_quality": 5.0},
        },
        threshold=0.85,
    )

    timing = report.results[0].metadata["environment_state"]["voice"]["timing_distribution"]
    metrics = evaluation.summary["metric_averages"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("timing_stages:", timing["stage_order"])
    print("turn_p95_ms:", timing["stages"]["turn"]["p95_ms"])
    print("voice_timing_distribution_quality:", metrics.get("voice_timing_distribution_quality"))
    print("voice_trace_coverage:", metrics.get("voice_trace_coverage"))


if __name__ == "__main__":
    asyncio.run(main())
