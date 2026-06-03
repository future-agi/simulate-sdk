"""
Replay a voice export with a real local WAV recording.

The example writes a small WAV fixture, loads it through `load_voice_export`,
and scores decoded media metadata: sample rate, duration, RMS/peak level, and
clipping. No room, media server, model, or API key is required.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio
import math
import struct
import tempfile
import wave
from pathlib import Path

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    load_voice_export,
)


def write_wav(path: Path, *, sample_rate_hz: int = 24000, duration_ms: int = 900) -> None:
    sample_count = int(sample_rate_hz * duration_ms / 1000)
    samples = [
        int(math.sin(2 * math.pi * 440 * index / sample_rate_hz) * 10000)
        for index in range(sample_count)
    ]
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def voice_export(path: Path):
    return {
        "framework": "livekit",
        "events": [
            {
                "id": "caller_1",
                "event": "user_input_transcribed",
                "transcript": "Billing issue for order 123.",
                "speaker_id": "caller",
            }
        ],
        "recordings": [{"id": "caller_wav", "speaker": "caller", "path": str(path)}],
        "speaker_segments": [
            {"speaker": "caller", "start_ms": 0, "end_ms": 900},
            {"speaker": "agent", "start_ms": 920, "end_ms": 1400},
        ],
    }


async def media_replay_agent(input):
    return AgentResponse(
        content="Decoded media replay inspected and routed to billing.",
        tool_calls=[
            {"id": "route", "name": "route_call", "arguments": {"route": "billing"}},
            {"id": "stt", "name": "transcribe_audio", "arguments": {"id": "caller_1"}},
            {"id": "status", "name": "voice_status", "arguments": {}},
        ],
    )


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "caller.wav"
        write_wav(wav_path)
        scenario = Scenario(
            name="voice-media-decode",
            dataset=[
                Persona(
                    persona={"name": "Noah", "risk_profile": "standard"},
                    situation="Noah called about billing; the export includes a captured WAV file.",
                    outcome="The call media is decoded, routed, and passes local quality gates.",
                )
            ],
        )
        report = await TestRunner().run_test(
            scenario=scenario,
            agent_callback=media_replay_agent,
            environment=load_voice_export(
                voice_export(wav_path),
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
                "required_artifact_types": ["audio", "trace"],
                "expected_voice_route": "billing",
                "expected_voice_transcript_contains": ["order 123"],
                "required_voice_speakers": ["caller", "agent"],
                "max_voice_clipping_ratio": 0.01,
                "min_voice_sample_rate_hz": 16000,
                "min_voice_duration_ms": 750,
                "max_voice_duration_ms": 1500,
                "min_voice_rms_db": -35,
                "max_voice_peak_db": -0.1,
                "required_voice_trace": [
                    "livekit_export",
                    "waveform",
                    "media",
                    "diarization",
                    "sample_rate",
                    "duration",
                    "rms",
                    "peak",
                    "clipping",
                ],
            },
            threshold=0.85,
        )

        voice_state = report.results[0].metadata["environment_state"]["voice"]
        waveform = voice_state["waveforms"][0]
        metrics = evaluation.summary["metric_averages"]

        print("score:", evaluation.score)
        print("passed:", evaluation.passed)
        print("decoded:", waveform["decoded_audio"])
        print("duration_ms:", waveform["duration_ms"])
        print("rms_db:", waveform["rms_db"])
        print("peak_db:", waveform["peak_db"])
        print("voice_interaction_quality:", metrics.get("voice_interaction_quality"))


if __name__ == "__main__":
    asyncio.run(main())
