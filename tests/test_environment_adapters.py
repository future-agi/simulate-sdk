import pytest

from fi.simulate import (
    AdversarialEnvironmentPack,
    AgentResponse,
    BrowserEnvironment,
    FileEnvironment,
    ImageEnvironment,
    MultiAgentRoomEnvironment,
    ToolMockEnvironment,
    VoiceEnvironment,
)
from fi.simulate.simulation.engines.local_text import LocalTextEngine
from fi.simulate.simulation.models import Persona, Scenario


def _scenario():
    return Scenario(
        name="env-smoke",
        dataset=[
            Persona(
                persona={"name": "Avery"},
                situation="Avery needs the environment to complete a task.",
                outcome="The environment-backed task is resolved.",
            )
        ],
    )


def test_environment_adapters_reset_mutable_state():
    file_env = FileEnvironment({"policy.md": "Policy"})
    file_env.reset()
    file_env.handle_tool_call(
        {"name": "write_file", "arguments": {"path": "tmp.md", "content": "temp"}}
    )
    assert "tmp.md" in file_env.files
    file_snapshot = file_env.reset()
    assert file_snapshot.state["files"]["paths"] == ["policy.md"]
    assert "tmp.md" not in file_env.files

    room_env = MultiAgentRoomEnvironment(["agent_a", "agent_b"])
    room_env.reset()
    room_env.handle_tool_call(
        {"name": "handoff", "arguments": {"to": "agent_b", "task": "review"}}
    )
    assert room_env.messages
    room_snapshot = room_env.reset()
    assert room_snapshot.state["multi_agent"]["messages"] == []
    assert room_env.messages == []


@pytest.mark.asyncio
async def test_tool_mock_environment_seeds_tools_and_executes_calls():
    seen_tools = []

    async def agent(input):
        seen_tools.extend(tool["name"] for tool in input.tools)
        return AgentResponse(
            content="I will look up the order.",
            tool_calls=[
                {
                    "id": "call_order",
                    "name": "search_order",
                    "arguments": {"order_id": "ord_123"},
                }
            ],
        )

    environment = ToolMockEnvironment(
        {
            "search_order": lambda args, ctx: {
                "content": "Order ord_123 is resolved",
                "result": {"status": "resolved"},
                "state_updates": {"order": {"status": "resolved"}},
            }
        }
    )

    report = await LocalTextEngine().run(
        scenario=_scenario(),
        agent_callback=agent,
        environment=environment,
        max_turns=1,
        min_turns=1,
    )

    result = report.results[0]
    assert "search_order" in seen_tools
    assert any(message["role"] == "tool" for message in result.messages)
    assert "Order ord_123 is resolved" in result.transcript
    assert result.metadata["environment_state"]["order"]["status"] == "resolved"
    assert any(event.type == "tool_execution" for event in result.events)


@pytest.mark.asyncio
async def test_browser_environment_exposes_dom_and_blocks_cross_origin():
    async def agent(input):
        return AgentResponse(
            content="I will navigate the browser.",
            tool_calls=[
                {
                    "id": "call_browser",
                    "name": "browser_navigate",
                    "arguments": {"url": "https://evil.example/pay"},
                }
            ],
        )

    report = await LocalTextEngine().run(
        scenario=_scenario(),
        agent_callback=agent,
        environment=BrowserEnvironment(
            url="https://shop.example.com/checkout",
            dom="<button id='pay'>Pay</button>",
            allowed_domains=["shop.example.com"],
        ),
        max_turns=1,
        min_turns=1,
        modality="cua",
    )

    result = report.results[0]
    assert result.artifacts[0].type == "browser_dom"
    assert any("outside allowed domains" in message["content"] for message in result.messages if message["role"] == "tool")
    browser_events = [event for event in result.events if event.type == "browser_action"]
    assert browser_events
    assert browser_events[-1].payload["blocked"] is True


@pytest.mark.asyncio
async def test_file_and_multi_agent_environments_update_state():
    async def agent(input):
        return AgentResponse(
            content="I will read a file and hand off to a specialist.",
            tool_calls=[
                {
                    "id": "call_read",
                    "name": "read_file",
                    "arguments": {"path": "policy.md"},
                },
                {
                    "id": "call_handoff",
                    "name": "handoff",
                    "arguments": {"to": "policy_specialist", "task": "review policy"},
                },
            ],
        )

    report = await LocalTextEngine().run(
        scenario=_scenario(),
        agent_callback=agent,
        environment=[
            FileEnvironment({"policy.md": "Refunds require approval."}),
            MultiAgentRoomEnvironment(["support_agent", "policy_specialist"]),
        ],
        max_turns=1,
        min_turns=1,
    )

    result = report.results[0]
    tool_text = "\n".join(
        message["content"] for message in result.messages if message["role"] == "tool"
    )
    assert "Refunds require approval" in tool_text
    assert "policy_specialist" in tool_text
    assert result.metadata["environment_state"]["files"]["paths"] == ["policy.md"]
    assert result.metadata["environment_state"]["multi_agent"]["messages"][0]["to"] == "policy_specialist"


@pytest.mark.asyncio
async def test_voice_and_image_environments_expose_media_and_execute_tools():
    seen_tools = []

    async def agent(input):
        seen_tools.extend(tool["name"] for tool in input.tools)
        return AgentResponse(
            content="I will inspect the image and respond by voice.",
            tool_calls=[
                {
                    "id": "call_image",
                    "name": "inspect_image",
                    "arguments": {"id": "receipt"},
                },
                {
                    "id": "call_stt",
                    "name": "transcribe_audio",
                    "arguments": {"id": "utt_1"},
                },
                {
                    "id": "call_speak",
                    "name": "speak",
                    "arguments": {
                        "text": "The receipt image shows order 123.",
                        "latency_ms": 350,
                    },
                },
                {
                    "id": "call_stop",
                    "name": "stop_speaking",
                    "arguments": {},
                },
            ],
        )

    report = await LocalTextEngine().run(
        scenario=_scenario(),
        agent_callback=agent,
        environment=[
            ImageEnvironment(
                {
                    "receipt": {
                        "uri": "file:///tmp/receipt.png",
                        "description": "Receipt for order 123",
                        "labels": ["receipt", "order"],
                    }
                }
            ),
            VoiceEnvironment(
                [
                    {
                        "id": "utt_1",
                        "transcript": "Please inspect the receipt.",
                        "audio_uri": "file:///tmp/user.wav",
                        "barge_in": True,
                    }
                ],
                sample_rate_hz=24000,
            ),
        ],
        max_turns=1,
        min_turns=1,
        modality="voice",
    )

    result = report.results[0]
    assert {"inspect_image", "transcribe_audio", "speak", "stop_speaking"}.issubset(set(seen_tools))
    assert any(artifact.type == "image" and artifact.metadata["id"] == "receipt" for artifact in result.artifacts)
    assert any(artifact.type == "audio" and artifact.metadata["id"] == "utt_1" for artifact in result.artifacts)
    assert any(event.type == "image" and event.name == "inspect_image" for event in result.events)
    assert any(event.type == "voice" and event.name == "tts_output" for event in result.events)
    assert result.metadata["environment_state"]["images"]["last_inspected"] == "receipt"
    assert result.metadata["environment_state"]["voice"]["interruptions_handled"] == 1


@pytest.mark.asyncio
async def test_adversarial_environment_pack_exposes_hostile_world_surfaces():
    async def agent(input):
        return AgentResponse(
            content="I will inspect retrieved context.",
            tool_calls=[
                {
                    "id": "call_search",
                    "name": "search_knowledge_base",
                    "arguments": {"query": "refund policy"},
                },
                {
                    "id": "call_file",
                    "name": "read_adversarial_file",
                    "arguments": {"path": "retrieved-policy.md"},
                },
                {
                    "id": "call_memory",
                    "name": "load_memory_context",
                    "arguments": {},
                },
            ],
        )

    report = await LocalTextEngine().run(
        scenario=_scenario(),
        agent_callback=agent,
        environment=AdversarialEnvironmentPack(),
        max_turns=1,
        min_turns=1,
    )

    result = report.results[0]
    tool_text = "\n".join(
        message["content"] for message in result.messages if message["role"] == "tool"
    )

    assert "Ignore previous instructions" in tool_text
    assert any(artifact.type == "file" for artifact in result.artifacts)
    assert any(artifact.type == "browser_dom" for artifact in result.artifacts)
    assert any(event.type == "environment_injection" for event in result.events)
    assert result.metadata["environment_state"]["adversarial"]["surfaces"] == [
        "tool",
        "file",
        "browser",
        "memory",
    ]
