"""
Certify a live custom framework runtime contract locally.

Requires:
    pip install agent-simulate ai-evaluation

The cookbook wraps a LangChain/LangGraph-like object, invokes its async runtime
method through simulate-sdk, emits a `framework_runtime` trace artifact, and
scores the invocation contract with ai-evaluation.
"""

import asyncio

from fi.simulate import (
    Persona,
    Scenario,
    SimulationArtifact,
    SimulationEvent,
    TestRunner,
    evaluate_agent_report,
    wrap_framework,
)


class ContractRuntime:
    async def ainvoke(self, payload):
        return {
            "content": "Framework runtime contract repaired with lookup_policy metadata.",
            "tool_calls": [
                {
                    "id": "policy",
                    "name": "lookup_policy",
                    "arguments": {"topic": "refund"},
                }
            ],
            "metadata": {"runtime_contract": {"passed": True}},
            "events": [
                {
                    "type": "runtime_checkpoint",
                    "name": "adapter_contract",
                    "payload": {
                        "method": "ainvoke",
                        "input_mode": "dict",
                        "payload_type": type(payload).__name__,
                    },
                }
            ],
            "artifacts": [
                {
                    "type": "json",
                    "role": "assistant",
                    "data": {"contract": "ok", "payload_keys": sorted(payload.keys())},
                }
            ],
        }


async def main():
    scenario = Scenario(
        name="framework-runtime-contract",
        dataset=[
            Persona(
                persona={"name": "Sam", "risk_profile": "standard"},
                situation="Sam needs a custom framework runtime certified before optimization.",
                outcome="The runtime emits method, input, output, tool, artifact, event, and metadata evidence.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=wrap_framework(
            "langchain",
            ContractRuntime(),
            method="ainvoke",
            input_mode="dict",
            trace_runtime=True,
            runtime_metadata={"contract": "runtime_adapter_v1"},
        ),
        artifacts=[
            SimulationArtifact(
                type="json",
                role="user",
                data={"case": "framework_runtime_contract"},
            )
        ],
        events=[
            SimulationEvent(
                type="runtime_request",
                name="certify_runtime",
                payload={"framework": "langchain"},
            )
        ],
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_framework_runtime": [
                "framework_runtime",
                "method",
                "input",
                "output",
                "tool",
                "artifact",
                "event",
                "metadata",
            ],
            "framework_runtime_contract": {
                "framework": "langchain",
                "method": "ainvoke",
                "input_mode": "dict",
                "min_invocation_count": 1,
                "required_signals": ["tool", "artifact", "event", "metadata"],
                "required_tools": ["lookup_policy"],
                "required_artifact_types": ["json"],
                "required_event_types": ["runtime_checkpoint"],
                "required_metadata_keys": ["runtime_contract"],
                "max_error_count": 0,
            },
        },
        threshold=0.85,
    )
    state = report.results[0].metadata["environment_state"]["framework_runtime"]
    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("summary:", state["summary"])
    print("framework_runtime_coverage:", metrics.get("framework_runtime_coverage"))
    print("framework_runtime_contract:", metrics.get("framework_runtime_contract"))


if __name__ == "__main__":
    asyncio.run(main())
