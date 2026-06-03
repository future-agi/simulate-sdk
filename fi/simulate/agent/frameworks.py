from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from fi.simulate.agent.generic import GenericAgentWrapper, InputMode
from fi.simulate.agent.wrapper import AgentWrapper


@dataclass(frozen=True)
class FrameworkAdapterSpec:
    """Import-free adapter preset for a common agent/orchestration framework."""

    name: str
    method: Optional[str]
    input_mode: InputMode
    modality: str = "text"
    notes: str = ""


FRAMEWORK_PRESETS: Dict[str, FrameworkAdapterSpec] = {
    # Text/chat orchestration
    "callable": FrameworkAdapterSpec("callable", None, "agent_input", notes="Plain Python callable."),
    "langchain": FrameworkAdapterSpec("langchain", "ainvoke", "dict", notes="LangChain Runnable/Chain."),
    "langgraph": FrameworkAdapterSpec("langgraph", "ainvoke", "dict", notes="LangGraph compiled graph."),
    "llamaindex": FrameworkAdapterSpec("llamaindex", "achat", "text", notes="LlamaIndex chat/query engines."),
    "crewai": FrameworkAdapterSpec("crewai", "kickoff", "dict", notes="CrewAI Crew kickoff."),
    "autogen": FrameworkAdapterSpec("autogen", "run", "text", notes="AutoGen AgentChat style task run."),
    "semantic_kernel": FrameworkAdapterSpec("semantic_kernel", "invoke", "dict", notes="Semantic Kernel function/agent."),
    "openai_agents": FrameworkAdapterSpec("openai_agents", "run", "text", notes="OpenAI Agents SDK runner/agent."),
    "pydantic_ai": FrameworkAdapterSpec("pydantic_ai", "run", "text", notes="PydanticAI agent."),
    "haystack": FrameworkAdapterSpec("haystack", "run", "dict", notes="Haystack pipeline."),
    # Voice and realtime
    "livekit": FrameworkAdapterSpec("livekit", "respond", "text", modality="voice", notes="LiveKit agent/session shim."),
    "pipecat": FrameworkAdapterSpec("pipecat", "process", "dict", modality="voice", notes="Pipecat pipeline/processor shim."),
    "vapi": FrameworkAdapterSpec("vapi", "respond", "dict", modality="voice", notes="Webhook/local adapter shim."),
    "retell": FrameworkAdapterSpec("retell", "respond", "dict", modality="voice", notes="Webhook/local adapter shim."),
    # Computer-use / browser / multimodal
    "computer_use": FrameworkAdapterSpec("computer_use", "run", "dict", modality="cua", notes="Browser or desktop CUA runner."),
    "browser_use": FrameworkAdapterSpec("browser_use", "run", "dict", modality="cua", notes="Browser automation agent."),
    "playwright": FrameworkAdapterSpec("playwright", "run", "dict", modality="cua", notes="Playwright-backed agent harness."),
    "vision_agent": FrameworkAdapterSpec("vision_agent", "run", "dict", modality="image", notes="Image or multimodal agent."),
}


def supported_frameworks() -> list[str]:
    """Return names accepted by wrap_framework."""

    return sorted(FRAMEWORK_PRESETS)


def wrap_framework(
    framework: str,
    agent: Any,
    *,
    method: str | None = None,
    input_mode: InputMode | None = None,
    system_prompt: str | None = None,
    output_key: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_runtime: bool = False,
    runtime_metadata: Optional[Dict[str, Any]] = None,
) -> AgentWrapper:
    """
    Wrap a known framework by name without importing that framework.

    Presets are intentionally thin. They encode the most common method/payload
    shape while leaving escape hatches for custom method, input_mode, and
    output_key.
    """

    key = framework.lower().replace("-", "_")
    if key not in FRAMEWORK_PRESETS:
        raise ValueError(
            f"Unsupported framework preset '{framework}'. Use GenericAgentWrapper "
            f"or one of: {', '.join(supported_frameworks())}."
        )

    spec = FRAMEWORK_PRESETS[key]
    return GenericAgentWrapper(
        agent,
        method=method or spec.method,
        input_mode=input_mode or spec.input_mode,
        output_key=output_key,
        system_prompt=system_prompt,
        metadata={"framework": spec.name, "modality": spec.modality, **(metadata or {})},
        trace_runtime=trace_runtime,
        runtime_metadata=runtime_metadata,
    )
