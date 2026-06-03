import inspect
import json
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

from fi.simulate.agent.wrapper import (
    AgentInput,
    AgentResponse,
    AgentWrapper,
    SimulationArtifact,
    SimulationEvent,
)

InputMode = Literal["auto", "agent_input", "dict", "messages", "text"]


class GenericAgentWrapper(AgentWrapper):
    """
    Framework-neutral adapter for agent objects, callables, and orchestration SDKs.

    The wrapper intentionally depends on conventions instead of optional imports:
    LangChain/LangGraph expose invoke/ainvoke, AutoGen and OpenAI-style runners often
    expose run/arun, voice stacks usually expose send/respond/chat, and plain Python
    agents are just callables. Users can override method/input_mode when a framework
    has a custom shape.
    """

    def __init__(
        self,
        agent: Any,
        *,
        method: str | Callable[..., Any] | None = None,
        input_mode: InputMode = "auto",
        output_key: str | None = None,
        system_prompt: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent = agent
        self.method = method
        self.input_mode = input_mode
        self.output_key = output_key
        self.system_prompt = system_prompt
        self.metadata = metadata or {}

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        method = self._resolve_method()
        method_name = getattr(method, "__name__", None) or (
            self.method if isinstance(self.method, str) else None
        )
        payload = self._build_payload(input, method_name=method_name)

        if payload is _NO_PAYLOAD:
            raw = method()
        else:
            raw = method(payload)

        if inspect.isawaitable(raw):
            raw = await raw

        if _is_async_stream(raw):
            raw = await self._coerce_async_stream(raw)
        elif _is_sync_stream(raw):
            raw = self._coerce_sync_stream(raw)

        return self._coerce_response(raw)

    def _resolve_method(self) -> Callable[..., Any]:
        if isinstance(self.agent, AgentWrapper):
            return self.agent.call

        if callable(self.method):
            return self.method

        if isinstance(self.method, str):
            candidate = getattr(self.agent, self.method, None)
            if callable(candidate):
                return candidate
            raise AttributeError(f"Agent does not expose method '{self.method}'.")

        for name in ("call", "ainvoke", "invoke", "arun", "run", "send", "respond", "chat"):
            candidate = getattr(self.agent, name, None)
            if callable(candidate):
                return candidate

        if callable(self.agent):
            return self.agent

        raise TypeError(
            "GenericAgentWrapper needs a callable agent or an object exposing one "
            "of call/ainvoke/invoke/arun/run/send/respond/chat."
        )

    def _build_payload(self, input: AgentInput, *, method_name: str | None) -> Any:
        mode = self.input_mode
        if mode == "auto":
            mode = self._infer_input_mode(method_name)

        if mode == "agent_input":
            return input

        messages = self._messages_with_system(input.messages)
        latest_text = _message_content(input.new_message) if input.new_message else ""

        if mode == "messages":
            return messages
        if mode == "text":
            return latest_text
        if mode == "dict":
            return {
                "messages": messages,
                "input": latest_text,
                "thread_id": input.thread_id,
                "execution_id": input.execution_id,
                "turn_index": input.turn_index,
                "scenario_name": input.scenario_name,
                "persona": input.persona,
                "situation": input.situation,
                "expected_outcome": input.expected_outcome,
                "modality": input.modality,
                "artifacts": [_model_to_dict(artifact) for artifact in input.artifacts],
                "events": [_model_to_dict(event) for event in input.events],
                "memory": input.memory,
                "tools": input.tools,
                "metadata": {**input.metadata, **self.metadata},
            }

        return input

    def _infer_input_mode(self, method_name: str | None) -> InputMode:
        if method_name in {"ainvoke", "invoke"}:
            return "dict"
        if method_name in {"arun", "run", "send", "respond", "chat"}:
            return "text"
        return "agent_input"

    def _messages_with_system(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = [dict(message) for message in messages]
        if not self.system_prompt:
            return normalized
        if normalized and normalized[0].get("role") == "system":
            return normalized
        return [{"role": "system", "content": self.system_prompt}, *normalized]

    def _coerce_response(self, raw: Any) -> str | AgentResponse:
        if isinstance(raw, AgentResponse):
            return raw
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")

        content = self._extract_content(raw)
        tool_calls = self._extract_tool_calls(raw)
        tool_responses = self._extract_tool_responses(raw)
        artifacts = self._extract_artifacts(raw)
        events = self._extract_events(raw)
        metadata = self._extract_metadata(raw)
        if self.metadata:
            metadata = {**metadata, **self.metadata}

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            artifacts=artifacts,
            events=events,
            metadata=metadata or None,
        )

    async def _coerce_async_stream(self, raw: Any) -> AgentResponse:
        chunks: List[Any] = []
        async for chunk in raw:
            chunks.append(chunk)
        return self._coerce_stream_chunks(chunks)

    def _coerce_sync_stream(self, raw: Any) -> AgentResponse:
        return self._coerce_stream_chunks(list(raw))

    def _coerce_stream_chunks(self, chunks: List[Any]) -> AgentResponse:
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_responses: List[Dict[str, Any]] = []
        artifacts: List[SimulationArtifact] = []
        events: List[SimulationEvent] = []

        for index, chunk in enumerate(chunks, start=1):
            text = _stream_chunk_text(chunk)
            if text:
                content_parts.append(text)
            tool_calls.extend(self._extract_tool_calls(chunk) or [])
            tool_responses.extend(self._extract_tool_responses(chunk) or [])
            artifacts.extend(self._extract_artifacts(chunk))
            events.extend(self._extract_events(chunk))
            events.append(_stream_chunk_event(chunk, index=index, text=text))

        trace = _streaming_trace_from_chunks(chunks, self.metadata)
        state: Dict[str, Any] = {}
        if trace.get("events"):
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=trace,
                    metadata={
                        "kind": "streaming_trace",
                        "framework": trace.get("framework", "generic"),
                        "source": "generic_agent_wrapper",
                    },
                )
            )
            state["streaming_trace"] = trace

        metadata = {
            "streaming": {
                "chunk_count": len(chunks),
                "content_part_count": len(content_parts),
                "signals": list(trace.get("signals", [])),
                "summary": dict(trace.get("summary", {})),
            },
            **self.metadata,
        }
        return AgentResponse(
            content="".join(content_parts),
            tool_calls=tool_calls or None,
            tool_responses=tool_responses or None,
            artifacts=artifacts,
            events=events,
            state=state or None,
            metadata=metadata,
        )

    def _extract_content(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")

        if isinstance(raw, dict):
            if self.output_key and self.output_key in raw:
                return _stringify(raw[self.output_key])
            for key in (
                "content",
                "output",
                "response",
                "text",
                "final_output",
                "answer",
                "result",
            ):
                if key in raw and raw[key] is not None:
                    return _stringify(raw[key])
            if "message" in raw:
                return _message_content(raw["message"])
            if "messages" in raw:
                return _last_message_content(raw["messages"])
            if "choices" in raw:
                return _choices_content(raw["choices"])

        for attr in ("content", "output", "response", "text", "final_output", "answer"):
            if hasattr(raw, attr):
                value = getattr(raw, attr)
                if value is not None:
                    return _stringify(value)

        if hasattr(raw, "message"):
            return _message_content(getattr(raw, "message"))
        if hasattr(raw, "messages"):
            return _last_message_content(getattr(raw, "messages"))
        if isinstance(raw, (list, tuple)):
            return _last_message_content(raw)

        return str(raw)

    def _extract_tool_calls(self, raw: Any) -> Optional[List[Dict[str, Any]]]:
        return _extract_list_field(
            raw,
            ("tool_calls", "toolCalls", "tool_call_chunks", "toolCallChunks"),
        )

    def _extract_tool_responses(self, raw: Any) -> Optional[List[Dict[str, Any]]]:
        return _extract_list_field(raw, ("tool_responses", "toolResponses", "tool_outputs", "toolOutputs"))

    def _extract_metadata(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            value = raw.get("metadata")
            return dict(value) if isinstance(value, dict) else {}
        value = getattr(raw, "metadata", None)
        return dict(value) if isinstance(value, dict) else {}

    def _extract_artifacts(self, raw: Any) -> List[SimulationArtifact]:
        values = _extract_list_field(raw, ("artifacts", "media", "attachments"))
        artifacts: List[SimulationArtifact] = []
        for value in values or []:
            try:
                artifacts.append(SimulationArtifact(**value))
            except Exception:
                continue
        return artifacts

    def _extract_events(self, raw: Any) -> List[SimulationEvent]:
        values = _extract_list_field(raw, ("events", "trajectory", "spans"))
        events: List[SimulationEvent] = []
        for value in values or []:
            try:
                events.append(SimulationEvent(**value))
            except Exception:
                continue
        return events


class _NoPayload:
    pass


_NO_PAYLOAD = _NoPayload()


def wrap_agent(
    agent: Any,
    *,
    method: str | Callable[..., Any] | None = None,
    input_mode: InputMode = "auto",
    output_key: str | None = None,
    system_prompt: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentWrapper:
    """Return an AgentWrapper for an existing AgentWrapper, object, or callable."""

    if isinstance(agent, AgentWrapper) and method is None and input_mode == "auto":
        return agent
    return GenericAgentWrapper(
        agent,
        method=method,
        input_mode=input_mode,
        output_key=output_key,
        system_prompt=system_prompt,
        metadata=metadata,
    )


def _extract_list_field(raw: Any, names: Iterable[str]) -> Optional[List[Dict[str, Any]]]:
    value = None
    if isinstance(raw, dict):
        for name in names:
            value = raw.get(name)
            if value is not None:
                break
    else:
        for name in names:
            if hasattr(raw, name):
                value = getattr(raw, name)
                break
    if not isinstance(value, list):
        return None
    return [dict(item) for item in value if isinstance(item, dict)] or None


def _is_async_stream(value: Any) -> bool:
    if isinstance(value, (AgentResponse, str, bytes, dict, list, tuple)):
        return False
    return inspect.isasyncgen(value) or hasattr(value, "__anext__") or hasattr(value, "__aiter__")


def _is_sync_stream(value: Any) -> bool:
    if isinstance(value, (AgentResponse, str, bytes, dict, list, tuple)):
        return False
    return inspect.isgenerator(value) or hasattr(value, "__next__")


def _stream_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, dict):
        for key in (
            "content",
            "delta",
            "text",
            "transcript",
            "output",
            "response",
            "final_output",
        ):
            value = chunk.get(key)
            if value is not None:
                return _stringify(value)
        for key in ("message", "chunk"):
            if key in chunk:
                return _message_content(chunk[key])
        if "choices" in chunk:
            return _choices_content(chunk["choices"])
        for key in ("data", "payload"):
            value = chunk.get(key)
            if isinstance(value, dict):
                text = _stream_chunk_text(value)
                if text:
                    return text
    for attr in ("content", "delta", "text", "transcript", "output", "response"):
        if hasattr(chunk, attr):
            value = getattr(chunk, attr)
            if value is not None:
                return _stringify(value)
    if hasattr(chunk, "message"):
        return _message_content(getattr(chunk, "message"))
    if hasattr(chunk, "choices"):
        return _choices_content(getattr(chunk, "choices"))
    return ""


def _stream_chunk_event(chunk: Any, *, index: int, text: str) -> SimulationEvent:
    payload = _stream_chunk_payload(chunk)
    if text:
        payload.setdefault("delta", text)
    return SimulationEvent(
        type=_stream_chunk_event_type(chunk),
        name=_stream_chunk_event_name(chunk, index=index),
        payload=payload,
        timestamp_ms=_stream_chunk_timestamp_ms(chunk),
        metadata={"stream_index": index},
    )


def _stream_chunk_event_type(chunk: Any) -> str:
    value = _stream_chunk_field(chunk, ("type", "event", "frame_type", "method"))
    if value:
        return str(value)
    return "stream_chunk"


def _stream_chunk_event_name(chunk: Any, *, index: int) -> str:
    value = _stream_chunk_field(chunk, ("name", "id", "event_id"))
    if value:
        return str(value)
    return f"stream_chunk_{index}"


def _stream_chunk_timestamp_ms(chunk: Any) -> Optional[int]:
    value = _stream_chunk_field(chunk, ("timestamp_ms", "time_ms"))
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _stream_chunk_payload(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        return dict(chunk)
    if isinstance(chunk, (str, bytes)):
        return {"delta": _stream_chunk_text(chunk)}
    if hasattr(chunk, "model_dump"):
        value = chunk.model_dump()
        return dict(value) if isinstance(value, dict) else {"value": value}
    if hasattr(chunk, "dict"):
        value = chunk.dict()
        return dict(value) if isinstance(value, dict) else {"value": value}
    payload: Dict[str, Any] = {}
    for key in ("id", "type", "event", "name", "content", "delta", "text", "transcript"):
        if hasattr(chunk, key):
            value = getattr(chunk, key)
            if value is not None:
                payload[key] = value
    return payload or {"value": str(chunk)}


def _stream_chunk_field(chunk: Any, names: Iterable[str]) -> Any:
    if isinstance(chunk, dict):
        for name in names:
            value = chunk.get(name)
            if value is not None:
                return value
        for key in ("data", "payload"):
            value = chunk.get(key)
            if isinstance(value, dict):
                nested = _stream_chunk_field(value, names)
                if nested is not None:
                    return nested
    for name in names:
        if hasattr(chunk, name):
            value = getattr(chunk, name)
            if value is not None:
                return value
    return None


def _streaming_trace_from_chunks(chunks: List[Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    from fi.simulate.environment import normalize_streaming_trace_events

    framework = str(metadata.get("framework") or "generic")
    trace_metadata = {
        "source": "generic_agent_wrapper",
        **dict(metadata),
    }
    return normalize_streaming_trace_events(
        framework,
        chunks,
        metadata=trace_metadata,
    )


def _choices_content(choices: Any) -> str:
    if not choices:
        return ""
    first = choices[0]
    if isinstance(first, dict):
        return _message_content(first.get("message") or first.get("delta") or first)
    return _message_content(getattr(first, "message", None) or getattr(first, "delta", None) or first)


def _last_message_content(messages: Any) -> str:
    if not messages:
        return ""
    try:
        return _message_content(list(messages)[-1])
    except TypeError:
        return _message_content(messages)


def _message_content(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        if "content" in message and message["content"] is not None:
            return _stringify(message["content"])
        if "text" in message and message["text"] is not None:
            return _stringify(message["text"])
        if "parts" in message:
            return " ".join(_stringify(part) for part in message["parts"])
    for attr in ("content", "text"):
        if hasattr(message, attr):
            value = getattr(message, attr)
            if value is not None:
                return _stringify(value)
    return str(message)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return str(value)


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)
