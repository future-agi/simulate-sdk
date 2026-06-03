"""
Run a local streaming/session trace replay.

Requires:
    pip install agent-simulate ai-evaluation

The fixture mirrors incremental output from LangChain/LangGraph stream modes,
OpenAI Agents stream events, LiveKit AgentSession events, Pipecat frames, or
OpenTelemetry-style GenAI streaming attributes without importing those runtimes.
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    StreamingTraceEnvironment,
    TestRunner,
    evaluate_agent_report,
)


STREAM_EVENTS = [
    {
        "id": "start",
        "type": "LLMFullResponseStartFrame",
        "timestamp_ms": 1000,
        "source": "pipecat.pipeline",
    },
    {
        "id": "chunk_1",
        "type": "messages",
        "delta": "Refund ",
        "role": "assistant",
        "timestamp_ms": 1120,
        "latency_ms": 120,
        "source": "langgraph:model_node",
    },
    {
        "id": "tool_delta",
        "type": "raw_response_event",
        "data": {
            "type": "response.function_call_arguments.delta",
            "delta": "{\"order_id\":\"ord_123\"",
        },
        "tool_call_chunks": [{"name": "lookup_order", "args": "{\"order_id\":\"ord_123\""}],
        "timestamp_ms": 1148,
    },
    {
        "id": "interruption",
        "event": "user_interruption_detected",
        "payload": {"probability": 0.91},
        "timestamp_ms": 1175,
    },
    {
        "id": "drop",
        "frame_type": "CancelFrame",
        "dropped_count": 1,
        "timestamp_ms": 1180,
    },
    {
        "id": "recovered",
        "event": "agent_false_interruption",
        "status": "resumed",
        "timestamp_ms": 1210,
    },
    {
        "id": "chunk_2",
        "type": "messages",
        "delta": "approved.",
        "gap_ms": 18,
        "timestamp_ms": 1228,
    },
    {
        "id": "usage",
        "event": "session_usage_updated",
        "usage": {"output_tokens": 9},
        "timestamp_ms": 1240,
    },
    {
        "id": "final",
        "event": "response.completed",
        "status": "completed",
        "timestamp_ms": 1250,
    },
]


async def streaming_agent(input):
    return AgentResponse(
        content="I inspected streaming chunks, tool deltas, interruption recovery, and final output.",
        tool_calls=[
            {"id": "status", "name": "streaming_trace_status", "arguments": {}},
            {"id": "chunks", "name": "list_stream_events", "arguments": {"signal": "chunk"}},
            {"id": "tool", "name": "inspect_stream_event", "arguments": {"id": "tool_delta"}},
        ],
    )


async def main():
    scenario = Scenario(
        name="streaming-trace-replay",
        dataset=[
            Persona(
                persona={"name": "Riya", "risk_profile": "standard"},
                situation="Riya needs a streamed refund response inspected before optimization.",
                outcome="The stream includes chunks, tool deltas, recovery, latency, gap, usage, and finalization evidence.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=streaming_agent,
        environment=StreamingTraceEnvironment(
            framework="mixed-realtime",
            events=STREAM_EVENTS,
            state={"response": {"status": "completed"}},
        ),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "streaming_trace_status",
                "list_stream_events",
                "inspect_stream_event",
            ],
            "available_tools": [
                "streaming_trace_status",
                "list_stream_events",
                "inspect_stream_event",
            ],
            "required_artifact_types": ["trace"],
            "required_streaming_trace": [
                "stream",
                "chunk",
                "tool_delta",
                "interruption",
                "recovered",
                "drop",
                "latency",
                "gap",
                "usage",
                "final",
                "state",
            ],
            "streaming_trace_quality": {
                "expected_output_contains": ["Refund approved"],
                "required_chunks": ["Refund ", "approved."],
                "expected_chunk_sequence": ["Refund ", "approved."],
                "expected_tool_deltas": [
                    {"name": "lookup_order", "arguments": {"order_id": "ord_123"}}
                ],
                "min_chunk_count": 2,
                "min_tool_delta_count": 1,
                "max_first_token_latency_ms": 200,
                "max_gap_ms": 50,
                "max_dropped_events": 1,
                "max_error_count": 0,
                "require_completion": True,
                "require_interruption_recovery": True,
                "expected_state": {"response": {"status": "completed"}},
            },
            "metric_weights": {
                "streaming_trace_coverage": 4.0,
                "streaming_interaction_quality": 5.0,
            },
        },
        threshold=0.85,
    )

    trace = report.results[0].metadata["environment_state"]["streaming_trace"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("signals:", trace["signals"])
    print("assembled_text:", trace["summary"]["assembled_text"])
    print("first_token_latency_ms:", trace["summary"].get("first_token_latency_ms"))
    print("streaming_trace_coverage:", metrics.get("streaming_trace_coverage"))
    print("streaming_interaction_quality:", metrics.get("streaming_interaction_quality"))


if __name__ == "__main__":
    asyncio.run(main())
