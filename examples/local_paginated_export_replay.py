"""
Replay authenticated paginated framework and voice exports locally.

The source specs are inline pages, but the same loader options work with HTTP
URLs, headers, bearer/API-key auth, next-url pagination, and cursor pagination.

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
    load_framework_trace_export,
    load_voice_export,
)


FRAMEWORK_EXPORT = {
    "auth": {"type": "bearer", "token": "redacted"},
    "pagination": {"enabled": True, "cursor_path": "pagination.next_cursor"},
    "pages": [
        {
            "records": [
                {
                    "name": "Future AGI support agent",
                    "span_id": "agent_span",
                    "attributes": {"fi.span.kind": "AGENT"},
                }
            ],
            "pagination": {"next_cursor": "page_2"},
        },
        {
            "records": [
                {
                    "name": "OpenAI chat completion",
                    "span_id": "model_span",
                    "attributes": {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.usage.input_tokens": 48,
                    },
                },
                {
                    "name": "MCP tool search_order",
                    "span_id": "tool_span",
                    "attributes": {"gen_ai.operation.name": "execute_tool"},
                },
            ]
        },
    ],
}

VOICE_EXPORT = {
    "framework": "livekit",
    "auth": {"type": "api_key", "header": "X-FI-Key", "token": "redacted"},
    "pagination": {"enabled": True, "next_url_path": "links.next"},
    "pages": [
        {
            "events": [
                {
                    "event": "user_input_transcribed",
                    "id": "caller_1",
                    "transcript": "Billing issue for order 123.",
                    "speaker_id": "caller",
                }
            ],
            "links": {"next": "page_2"},
        },
        {
            "frames": [
                {"id": "input", "frame_type": "InputAudioRawFrame", "sample_rate": 24000},
                {
                    "id": "transcript",
                    "frame_type": "TranscriptionFrame",
                    "text": "Billing issue for order 123.",
                },
            ],
            "recordings": [
                {"id": "caller_wave", "speaker": "caller", "duration_ms": 1000, "sample_rate_hz": 24000}
            ],
        },
    ],
}


async def export_agent(input):
    return AgentResponse(
        content="Paginated exports inspected with trace and voice evidence.",
        tool_calls=[
            {"id": "trace", "name": "framework_trace_status", "arguments": {}},
            {"id": "spans", "name": "list_framework_spans", "arguments": {"signal": "model"}},
            {"id": "voice", "name": "voice_status", "arguments": {}},
            {"id": "stt", "name": "transcribe_audio", "arguments": {"id": "caller_1"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="paginated-export-replay",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs stored framework and voice exports inspected before optimization.",
                outcome="Paginated exports inspected with authenticated source metadata.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=export_agent,
        environment=[
            load_framework_trace_export(FRAMEWORK_EXPORT, framework="future_agi"),
            load_voice_export(VOICE_EXPORT, framework="livekit"),
        ],
        max_turns=1,
        min_turns=1,
        modality="voice",
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "framework_trace_status",
                "list_framework_spans",
                "voice_status",
                "transcribe_audio",
            ],
            "available_tools": [
                "framework_trace_status",
                "list_framework_spans",
                "voice_status",
                "transcribe_audio",
            ],
            "required_artifact_types": ["trace", "audio"],
            "required_framework_trace": [
                "agent",
                "model",
                "tool",
                "cost",
                "export",
                "export_auth",
                "export_pagination",
            ],
            "required_voice_trace": [
                "livekit_export",
                "stt",
                "frame",
                "waveform",
                "export",
                "export_auth",
                "export_pagination",
            ],
            "metric_weights": {"framework_trace_coverage": 4.0, "voice_trace_coverage": 3.0},
        },
        threshold=0.85,
    )

    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("framework_trace_coverage:", metrics.get("framework_trace_coverage"))
    print("voice_trace_coverage:", metrics.get("voice_trace_coverage"))


if __name__ == "__main__":
    asyncio.run(main())
