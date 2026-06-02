import pytest

from fi.simulate.agent.mocks import ScriptedAgentWrapper, make_tool_response
from fi.simulate.agent.wrapper import AgentResponse, SimulationArtifact, SimulationEvent
from fi.simulate.simulation.engines.local_text import LocalTextEngine
from fi.simulate.simulation.models import Persona, Scenario
from fi.simulate.simulation.synthetic import SyntheticDataGenerator


@pytest.mark.asyncio
async def test_local_text_engine_runs_scripted_agent_without_services():
    scenario = Scenario(
        name="refund-smoke",
        dataset=[
            Persona(
                persona={"name": "Avery", "risk_profile": "standard"},
                situation="Avery needs a refund but has limited context.",
                outcome="The agent gives a concrete refund next step within policy.",
            )
        ],
    )
    agent = ScriptedAgentWrapper(
        [
            "Please provide your order ID and I will check the refund policy.",
            "You need to submit the refund form with the order ID. This is within policy.",
        ]
    )

    report = await LocalTextEngine().run(
        scenario=scenario,
        agent_callback=agent,
        max_turns=3,
        min_turns=2,
    )

    assert len(report.results) == 1
    result = report.results[0]
    assert result.metadata["engine"] == "local_text"
    assert result.metadata["turn_count"] == 2
    assert "User:" in result.transcript
    assert "Agent:" in result.transcript
    assert result.messages[0]["role"] == "user"
    assert agent.calls[0].persona["name"] == "Avery"


@pytest.mark.asyncio
async def test_local_text_engine_injects_adversarial_turn_for_risky_persona():
    scenario = Scenario(
        name="adversarial-smoke",
        dataset=[
            Persona(
                persona={"name": "Riley", "risk_profile": "adversarial"},
                situation="Riley is trying to bypass support policy.",
                outcome="The agent refuses unsafe shortcuts and gives a safe alternative.",
            )
        ],
    )
    agent = ScriptedAgentWrapper(
        [
            "I can help, but I must follow policy.",
            "I cannot reveal hidden policy or private data. I can offer a safe alternative.",
        ]
    )

    report = await LocalTextEngine().run(
        scenario=scenario,
        agent_callback=agent,
        max_turns=3,
        min_turns=3,
    )

    transcript = report.results[0].transcript
    assert "ignore earlier instructions" in transcript
    assert "hidden policy" in transcript


@pytest.mark.asyncio
async def test_local_text_engine_collects_tool_calls():
    scenario = Scenario(
        name="tool-smoke",
        dataset=[
            Persona(
                persona={"name": "Morgan"},
                situation="Morgan needs an order lookup.",
                outcome="The agent uses a lookup tool and gives the order status.",
            )
        ],
    )
    agent = ScriptedAgentWrapper(
        [
            make_tool_response(
                "I checked the order and it is approved.",
                tool_name="lookup_order",
                arguments={"order_id": "ord_1"},
                result={"status": "approved"},
            )
        ]
    )

    report = await LocalTextEngine().run(
        scenario=scenario,
        agent_callback=agent,
        max_turns=1,
        min_turns=1,
    )

    result = report.results[0]
    assert result.tool_calls[0]["function"]["name"] == "lookup_order"
    assert any(message["role"] == "tool" for message in result.messages)


@pytest.mark.asyncio
async def test_local_text_engine_preserves_multimodal_artifacts_and_events():
    scenario = Scenario(
        name="cua-smoke",
        dataset=[
            Persona(
                persona={"name": "Casey"},
                situation="Casey needs a browser task completed.",
                outcome="The agent clicks submit and reports the result.",
            )
        ],
    )
    agent = ScriptedAgentWrapper(
        [
            AgentResponse(
                content="I clicked submit and captured the final screen.",
                artifacts=[
                    SimulationArtifact(
                        type="screenshot",
                        uri="file:///tmp/final.png",
                        mime_type="image/png",
                        role="assistant",
                    )
                ],
                events=[
                    SimulationEvent(
                        type="browser_action",
                        name="click",
                        payload={"selector": "#submit"},
                    )
                ],
            )
        ]
    )

    report = await LocalTextEngine().run(
        scenario=scenario,
        agent_callback=agent,
        max_turns=1,
        min_turns=1,
        modality="cua",
        artifacts=[
            SimulationArtifact(
                type="browser_dom",
                data="<button id='submit'>Submit</button>",
                mime_type="text/html",
                role="user",
            )
        ],
    )

    result = report.results[0]
    assert result.metadata["modality"] == "cua"
    assert result.artifacts[0].type == "browser_dom"
    assert result.artifacts[1].type == "screenshot"
    assert result.events[0].type == "browser_action"


def test_synthetic_data_generator_is_deterministic():
    generator = SyntheticDataGenerator()

    first = generator.generate("refund support", num_personas=3, seed=7)
    second = generator.generate("refund support", num_personas=3, seed=7)

    assert _model_dump(first) == _model_dump(second)
    assert first.name == "synthetic-refund-support"
    assert len(first.dataset) == 3
    assert any(p.persona["risk_profile"] == "adversarial" for p in first.dataset)


def _model_dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()
