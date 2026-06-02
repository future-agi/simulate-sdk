from __future__ import annotations

import copy
import json
from abc import ABC
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from fi.simulate.agent.wrapper import SimulationArtifact, SimulationEvent


class EnvironmentSnapshot(BaseModel):
    """State, tools, artifacts, and events exposed by a simulation environment."""

    tools: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[SimulationArtifact] = Field(default_factory=list)
    events: List[SimulationEvent] = Field(default_factory=list)
    state: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    """Result from executing a tool call inside a local environment."""

    tool_call_id: Optional[str] = None
    tool_name: str
    content: str
    result: Any = None
    success: bool = True
    error: Optional[str] = None
    state_updates: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[SimulationArtifact] = Field(default_factory=list)
    events: List[SimulationEvent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_tool_message(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id or self.tool_name,
            "content": self.content,
        }


class EnvironmentAdapter(ABC):
    """Base class for local simulation environments."""

    name = "environment"

    def reset(self, **context: Any) -> EnvironmentSnapshot:
        return EnvironmentSnapshot()

    def observe(self, **context: Any) -> EnvironmentSnapshot:
        return EnvironmentSnapshot()

    def handle_tool_call(
        self,
        tool_call: Mapping[str, Any],
        **context: Any,
    ) -> Optional[ToolExecutionResult]:
        return None


class ToolMockEnvironment(EnvironmentAdapter):
    """
    Local API/tool mock environment.

    Handlers can return plain values, dictionaries, or ToolExecutionResult. A
    dictionary can include `content`, `result`, `success`, `error`,
    `state_updates`, `artifacts`, and `events`.
    """

    name = "tool_mock"

    def __init__(
        self,
        tools: Mapping[str, Callable[[Dict[str, Any], Dict[str, Any]], Any] | Any],
        *,
        tool_schemas: Optional[Iterable[Dict[str, Any]]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.handlers = dict(tools)
        self.tool_schemas = list(tool_schemas or [])
        self.initial_state = copy.deepcopy(initial_state or {})
        self.state = copy.deepcopy(self.initial_state)

    def reset(self, **context: Any) -> EnvironmentSnapshot:
        self.state = copy.deepcopy(self.initial_state)
        return EnvironmentSnapshot(
            tools=self._tool_specs(),
            state=copy.deepcopy(self.state),
            events=[
                SimulationEvent(
                    type="environment",
                    name="tool_mock_ready",
                    payload={"tools": sorted(self.handlers.keys())},
                )
            ],
        )

    def handle_tool_call(
        self,
        tool_call: Mapping[str, Any],
        **context: Any,
    ) -> Optional[ToolExecutionResult]:
        name = _tool_name(tool_call)
        if not name:
            return None
        if name not in self.handlers:
            return None

        arguments = _tool_arguments(tool_call)
        call_id = _tool_call_id(tool_call)
        handler = self.handlers[name]
        try:
            raw = handler(arguments, context) if callable(handler) else handler
            result = _coerce_tool_result(raw, tool_name=name, tool_call_id=call_id)
        except Exception as exc:
            result = ToolExecutionResult(
                tool_call_id=call_id,
                tool_name=name,
                content=f"Tool {name} failed: {exc}",
                success=False,
                error=str(exc),
            )

        _deep_merge(self.state, result.state_updates)
        result.events.append(
            SimulationEvent(
                type="tool_execution",
                name=name,
                payload={
                    "arguments": arguments,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                },
            )
        )
        return result

    def _tool_specs(self) -> List[Dict[str, Any]]:
        if self.tool_schemas:
            return copy.deepcopy(self.tool_schemas)
        specs = []
        for name in sorted(self.handlers.keys()):
            specs.append(
                {
                    "name": name,
                    "description": f"Mocked tool '{name}' available in the local simulation.",
                    "parameters": {"type": "object", "properties": {}},
                }
            )
        return specs


class BrowserEnvironment(EnvironmentAdapter):
    """Minimal browser/CUA environment with DOM artifacts and domain policy."""

    name = "browser"

    def __init__(
        self,
        *,
        url: str = "https://example.test/",
        dom: str = "<html><body></body></html>",
        screenshot_uri: Optional[str] = None,
        allowed_domains: Optional[Iterable[str]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.initial_url = url
        self.url = url
        self.dom = dom
        self.screenshot_uri = screenshot_uri
        self.allowed_domains = {domain.lower() for domain in allowed_domains or []}
        self.initial_state = copy.deepcopy(state or {})
        self.state = copy.deepcopy(self.initial_state)

    def reset(self, **context: Any) -> EnvironmentSnapshot:
        self.url = self.initial_url
        self.state = copy.deepcopy(self.initial_state)
        artifacts = [
            SimulationArtifact(
                type="browser_dom",
                data=self.dom,
                mime_type="text/html",
                role="environment",
                metadata={"url": self.url},
            )
        ]
        if self.screenshot_uri:
            artifacts.append(
                SimulationArtifact(
                    type="screenshot",
                    uri=self.screenshot_uri,
                    role="environment",
                    metadata={"url": self.url},
                )
            )
        return EnvironmentSnapshot(
            tools=[
                {
                    "name": "browser_navigate",
                    "description": "Navigate the simulated browser to a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
                {
                    "name": "browser_click",
                    "description": "Click an element in the simulated browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "action": {"type": "string"},
                        },
                    },
                },
            ],
            artifacts=artifacts,
            state={"browser": {"url": self.url, **copy.deepcopy(self.state)}},
            events=[
                SimulationEvent(
                    type="environment",
                    name="browser_ready",
                    payload={"url": self.url, "allowed_domains": sorted(self.allowed_domains)},
                )
            ],
        )

    def handle_tool_call(
        self,
        tool_call: Mapping[str, Any],
        **context: Any,
    ) -> Optional[ToolExecutionResult]:
        name = _tool_name(tool_call)
        if name not in {"browser_navigate", "browser_click", "playwright_click", "computer_click"}:
            return None

        arguments = _tool_arguments(tool_call)
        call_id = _tool_call_id(tool_call)
        requested_url = str(arguments.get("url") or self.url)
        action = str(arguments.get("action") or arguments.get("selector") or name)
        allowed, reason = self._allowed_url(requested_url)
        if not allowed:
            return ToolExecutionResult(
                tool_call_id=call_id,
                tool_name=name,
                content=f"Blocked browser action: {reason}",
                result={"url": requested_url, "action": action},
                success=False,
                error=reason,
                events=[
                    SimulationEvent(
                        type="browser_action",
                        name=name,
                        payload={"url": requested_url, "action": action, "blocked": True},
                    )
                ],
            )

        self.url = requested_url
        state_update = {"browser": {"url": self.url, "last_action": action}}
        return ToolExecutionResult(
            tool_call_id=call_id,
            tool_name=name,
            content=f"Browser action completed: {action} at {self.url}",
            result={"url": self.url, "action": action},
            state_updates=state_update,
            events=[
                SimulationEvent(
                    type="browser_action",
                    name=name,
                    payload={"url": self.url, "action": action, "blocked": False},
                )
            ],
        )

    def _allowed_url(self, url: str) -> tuple[bool, str]:
        if not self.allowed_domains:
            return True, ""
        host = urlparse(url).netloc.lower()
        if any(host == domain or host.endswith(f".{domain}") for domain in self.allowed_domains):
            return True, ""
        return False, f"host '{host}' is outside allowed domains"


class FileEnvironment(EnvironmentAdapter):
    """In-memory file environment with read/write/list tools."""

    name = "files"

    def __init__(self, files: Optional[Mapping[str, str]] = None) -> None:
        self.initial_files = dict(files or {})
        self.files = dict(self.initial_files)

    def reset(self, **context: Any) -> EnvironmentSnapshot:
        self.files = dict(self.initial_files)
        return EnvironmentSnapshot(
            tools=[
                {"name": "list_files", "description": "List simulated files."},
                {"name": "read_file", "description": "Read a simulated file."},
                {"name": "write_file", "description": "Write a simulated file."},
            ],
            artifacts=[
                SimulationArtifact(
                    type="file",
                    data={"files": sorted(self.files.keys())},
                    role="environment",
                )
            ],
            state={"files": {"paths": sorted(self.files.keys())}},
        )

    def handle_tool_call(
        self,
        tool_call: Mapping[str, Any],
        **context: Any,
    ) -> Optional[ToolExecutionResult]:
        name = _tool_name(tool_call)
        if name not in {"list_files", "read_file", "write_file"}:
            return None
        arguments = _tool_arguments(tool_call)
        call_id = _tool_call_id(tool_call)
        if name == "list_files":
            result = sorted(self.files.keys())
        elif name == "read_file":
            path = str(arguments.get("path", ""))
            if path not in self.files:
                return ToolExecutionResult(
                    tool_call_id=call_id,
                    tool_name=name,
                    content=f"File not found: {path}",
                    success=False,
                    error="file_not_found",
                )
            result = self.files[path]
        else:
            path = str(arguments.get("path", ""))
            content = str(arguments.get("content", ""))
            self.files[path] = content
            result = {"path": path, "bytes": len(content.encode("utf-8"))}
        return ToolExecutionResult(
            tool_call_id=call_id,
            tool_name=name,
            content=result if isinstance(result, str) else json.dumps(result, default=str),
            result=result,
            state_updates={"files": {"paths": sorted(self.files.keys())}},
            events=[
                SimulationEvent(
                    type="file_action",
                    name=name,
                    payload={"arguments": arguments, "result": result},
                )
            ],
        )


class MultiAgentRoomEnvironment(EnvironmentAdapter):
    """Simple multi-agent room environment for handoff and coordination tests."""

    name = "multi_agent_room"

    def __init__(self, participants: Iterable[str]) -> None:
        self.participants = list(participants)
        self.messages: List[Dict[str, Any]] = []

    def reset(self, **context: Any) -> EnvironmentSnapshot:
        self.messages = []
        return EnvironmentSnapshot(
            tools=[
                {
                    "name": "handoff",
                    "description": "Hand off work to another simulated agent role.",
                },
                {
                    "name": "send_room_message",
                    "description": "Send a message to the simulated multi-agent room.",
                },
            ],
            state={"multi_agent": {"participants": self.participants, "messages": []}},
            events=[
                SimulationEvent(
                    type="multi_agent",
                    name="room_ready",
                    payload={"participants": self.participants},
                )
            ],
        )

    def handle_tool_call(
        self,
        tool_call: Mapping[str, Any],
        **context: Any,
    ) -> Optional[ToolExecutionResult]:
        name = _tool_name(tool_call)
        if name not in {"handoff", "send_room_message"}:
            return None
        arguments = _tool_arguments(tool_call)
        call_id = _tool_call_id(tool_call)
        recipient = arguments.get("to") or arguments.get("role") or "room"
        content = str(arguments.get("message") or arguments.get("task") or "")
        message = {"tool": name, "to": recipient, "message": content}
        self.messages.append(message)
        return ToolExecutionResult(
            tool_call_id=call_id,
            tool_name=name,
            content=f"{name} sent to {recipient}: {content}",
            result=message,
            state_updates={"multi_agent": {"participants": self.participants, "messages": list(self.messages)}},
            events=[
                SimulationEvent(
                    type="multi_agent",
                    name=name,
                    payload=message,
                )
            ],
        )


def coerce_environment_adapters(
    environment: EnvironmentAdapter | Iterable[EnvironmentAdapter] | None,
) -> List[EnvironmentAdapter]:
    if environment is None:
        return []
    if isinstance(environment, EnvironmentAdapter):
        return [environment]
    return list(environment)


def _coerce_tool_result(
    value: Any,
    *,
    tool_name: str,
    tool_call_id: Optional[str],
) -> ToolExecutionResult:
    if isinstance(value, ToolExecutionResult):
        if value.tool_call_id is None:
            value.tool_call_id = tool_call_id
        return value
    if isinstance(value, dict):
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=str(value.get("content", value.get("result", ""))),
            result=value.get("result", value),
            success=bool(value.get("success", True)),
            error=value.get("error"),
            state_updates=dict(value.get("state_updates", {})),
            artifacts=[_coerce_artifact(item) for item in value.get("artifacts", [])],
            events=[_coerce_event(item) for item in value.get("events", [])],
            metadata=dict(value.get("metadata", {})),
        )
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=str(value),
        result=value,
    )


def _tool_name(tool_call: Mapping[str, Any]) -> Optional[str]:
    function = tool_call.get("function")
    if isinstance(function, dict):
        return tool_call.get("name") or function.get("name")
    return tool_call.get("name") or tool_call.get("tool") or tool_call.get("action")


def _tool_arguments(tool_call: Mapping[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function")
    value: Any = tool_call.get("arguments", tool_call.get("args", tool_call.get("input", {})))
    if isinstance(function, dict) and "arguments" in function:
        value = function["arguments"]
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": value}
    return {"value": value}


def _tool_call_id(tool_call: Mapping[str, Any]) -> Optional[str]:
    value = tool_call.get("id") or tool_call.get("tool_call_id") or tool_call.get("call_id")
    return str(value) if value is not None else None


def _coerce_artifact(value: SimulationArtifact | Dict[str, Any]) -> SimulationArtifact:
    if isinstance(value, SimulationArtifact):
        return value
    return SimulationArtifact(**value)


def _coerce_event(value: SimulationEvent | Dict[str, Any]) -> SimulationEvent:
    if isinstance(value, SimulationEvent):
        return value
    return SimulationEvent(**value)


def _deep_merge(target: Dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
