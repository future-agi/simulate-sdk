"""
Replay and score framework lifecycle/session evidence locally.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    FrameworkLifecycleEnvironment,
    Persona,
    Scenario,
    evaluate_agent_report,
    normalize_framework_lifecycle_trace,
)
from fi.simulate.simulation.engines.local_text import LocalTextEngine


LIFECYCLE_TRACE = normalize_framework_lifecycle_trace(
    name="langgraph-lifecycle",
    framework="langgraph",
    session_id="thread-123",
    phases=[
        {"id": "init", "stage": "initialize", "status": "completed", "state": {"config": "loaded"}},
        {"id": "tools", "stage": "register_tools", "registered_tools": ["search_order", "issue_refund"]},
        {"id": "start", "stage": "start_session", "state_keys": ["thread_id", "messages"]},
        {"id": "invoke", "stage": "invoke", "latency_ms": 42, "state_keys": ["messages"]},
        {"id": "stream", "stage": "stream", "status": "completed"},
        {"id": "checkpoint", "stage": "checkpoint", "checkpoint": {"thread_id": "thread-123", "step": 1}},
        {"id": "retry", "stage": "retry", "retry_of": "invoke", "error": "tool timeout", "recovered": True},
        {"id": "cancel", "stage": "cancel", "status": "cancelled"},
        {"id": "resume", "stage": "resume", "status": "resumed", "state_persisted": True},
        {"id": "shutdown", "stage": "shutdown", "status": "completed"},
    ],
    state={"thread_id": "thread-123", "case": {"status": "resolved"}},
)


async def agent(input):
    return AgentResponse(
        content="Framework lifecycle trace includes setup, tool registration, checkpoint, retry, resume, and cleanup evidence.",
        tool_calls=[
            {"id": "status", "name": "framework_lifecycle_status", "arguments": {}},
            {
                "id": "phases",
                "name": "list_framework_lifecycle_phases",
                "arguments": {"session_id": "thread-123"},
            },
            {
                "id": "session",
                "name": "inspect_framework_session",
                "arguments": {"session_id": "thread-123"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="framework-lifecycle-quality",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs a framework lifecycle certified before rollout.",
                outcome="The framework lifecycle trace proves setup, session persistence, retry, resume, and cleanup.",
            )
        ],
    )
    report = await LocalTextEngine().run(
        scenario=scenario,
        agent_callback=agent,
        environment=FrameworkLifecycleEnvironment(LIFECYCLE_TRACE),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_framework_lifecycle": [
                "framework_lifecycle",
                "initialize",
                "tool_registration",
                "start_session",
                "invocation",
                "streaming",
                "checkpoint",
                "retry",
                "cancellation",
                "resume",
                "cleanup",
                "state_persistence",
                "session",
            ],
            "framework_lifecycle_quality": {
                "framework": "langgraph",
                "required_sessions": ["thread-123"],
                "required_stages": ["initialize", "tool_registration", "start_session", "invoke", "checkpoint", "resume", "shutdown"],
                "min_phase_count": 10,
                "min_tool_registrations": 1,
                "min_invocations": 1,
                "min_recovered_errors": 1,
                "require_streaming": True,
                "require_checkpoint": True,
                "require_retry": True,
                "require_cancellation": True,
                "require_resume": True,
                "require_cleanup": True,
                "require_state_persistence": True,
                "terminal_status": "completed",
                "max_error_count": 1,
            },
            "metric_weights": {
                "framework_lifecycle_coverage": 4.0,
                "framework_lifecycle_quality": 6.0,
            },
        },
        threshold=0.9,
    )
    metrics = evaluation.summary["metric_averages"]
    print("score:", round(evaluation.score, 4))
    print("passed:", evaluation.passed)
    print("summary:", LIFECYCLE_TRACE["summary"])
    print("framework_lifecycle_coverage:", metrics.get("framework_lifecycle_coverage"))
    print("framework_lifecycle_quality:", metrics.get("framework_lifecycle_quality"))


if __name__ == "__main__":
    asyncio.run(main())
