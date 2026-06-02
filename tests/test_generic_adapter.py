import pytest

from fi.simulate.agent.frameworks import supported_frameworks, wrap_framework
from fi.simulate.agent.generic import GenericAgentWrapper, wrap_agent
from fi.simulate.agent.wrapper import AgentInput, AgentResponse, SimulationArtifact


@pytest.fixture
def agent_input():
    return AgentInput(
        thread_id="thread-1",
        execution_id="exec-1",
        turn_index=1,
        scenario_name="returns",
        persona={"name": "Avery"},
        situation="Avery needs a refund.",
        expected_outcome="Refund is resolved within policy.",
        messages=[
            {"role": "user", "content": "I need a refund."},
            {"role": "assistant", "content": "I can help."},
            {"role": "user", "content": "What is the next step?"},
        ],
        new_message={"role": "user", "content": "What is the next step?"},
        modality="image",
        artifacts=[
            SimulationArtifact(
                type="image",
                uri="file:///tmp/refund-form.png",
                mime_type="image/png",
                role="user",
            )
        ],
        memory={"account_id": "acct-1"},
    )


@pytest.mark.asyncio
async def test_generic_wrapper_adapts_async_runnable(agent_input):
    class RunnableAgent:
        async def ainvoke(self, payload):
            assert payload["input"] == "What is the next step?"
            assert payload["persona"]["name"] == "Avery"
            assert payload["messages"][0]["role"] == "system"
            assert payload["modality"] == "image"
            assert payload["artifacts"][0]["type"] == "image"
            return {
                "output": "Submit the refund form and include the order ID.",
                "metadata": {"framework": "langchain-like"},
            }

    wrapper = GenericAgentWrapper(RunnableAgent(), system_prompt="Stay within policy.")

    response = await wrapper.call(agent_input)

    assert isinstance(response, AgentResponse)
    assert response.content.startswith("Submit the refund form")
    assert response.metadata["framework"] == "langchain-like"


@pytest.mark.asyncio
async def test_generic_wrapper_adapts_sync_message_callable(agent_input):
    def message_agent(messages):
        return f"received {len(messages)} messages"

    wrapper = GenericAgentWrapper(message_agent, input_mode="messages")

    response = await wrapper.call(agent_input)

    assert response == "received 3 messages"


@pytest.mark.asyncio
async def test_generic_wrapper_extracts_autogen_like_task_result(agent_input):
    class Message:
        def __init__(self, content):
            self.content = content

    class TaskResult:
        messages = [Message("user task"), Message("final agent answer")]

    class AutoGenLikeAgent:
        def run(self, task):
            assert task == "What is the next step?"
            return TaskResult()

    response = await wrap_agent(AutoGenLikeAgent()).call(agent_input)

    assert isinstance(response, AgentResponse)
    assert response.content == "final agent answer"


@pytest.mark.asyncio
async def test_generic_wrapper_preserves_tool_calls(agent_input):
    class ToolCallingAgent:
        def invoke(self, payload):
            return {
                "content": "I looked up the refund.",
                "tool_calls": [{"id": "call_1", "name": "lookup_refund", "args": {"id": "r1"}}],
                "tool_responses": [{"role": "tool", "tool_call_id": "call_1", "content": "approved"}],
            }

    response = await GenericAgentWrapper(ToolCallingAgent()).call(agent_input)

    assert isinstance(response, AgentResponse)
    assert response.tool_calls[0]["name"] == "lookup_refund"
    assert response.tool_responses[0]["content"] == "approved"


@pytest.mark.asyncio
async def test_framework_preset_wraps_browser_cua_agent(agent_input):
    class BrowserAgent:
        def run(self, payload):
            assert payload["modality"] == "image"
            return {
                "content": "Clicked the refund form submit button.",
                "artifacts": [
                    {
                        "type": "screenshot",
                        "uri": "file:///tmp/after-submit.png",
                        "mime_type": "image/png",
                        "role": "assistant",
                    }
                ],
                "events": [
                    {
                        "type": "browser_action",
                        "name": "click",
                        "payload": {"selector": "#submit-refund"},
                    }
                ],
            }

    response = await wrap_framework("computer_use", BrowserAgent()).call(agent_input)

    assert "computer_use" in supported_frameworks()
    assert isinstance(response, AgentResponse)
    assert response.artifacts[0].type == "screenshot"
    assert response.events[0].type == "browser_action"
    assert response.metadata["modality"] == "cua"
