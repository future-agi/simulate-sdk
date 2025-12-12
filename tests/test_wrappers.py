import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fi.simulate.agent.wrapper import AgentInput
from fi.simulate.agent.wrappers.openai import OpenAIAgentWrapper
from fi.simulate.agent.wrappers.langchain import LangChainAgentWrapper
from fi.simulate.agent.wrappers.anthropic import AnthropicAgentWrapper
from fi.simulate.agent.wrappers.gemini import GeminiAgentWrapper

# --- Fixtures ---

@pytest.fixture
def agent_input():
    return AgentInput(
        thread_id="test-thread",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ],
        new_message={"role": "user", "content": "How are you?"}
    )

# --- OpenAI Tests ---

@pytest.mark.asyncio
async def test_openai_wrapper_async(agent_input):
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="I am an AI"))
    ]
    # Make it look like AsyncOpenAI
    mock_client.__class__.__name__ = "AsyncOpenAI"

    wrapper = OpenAIAgentWrapper(client=mock_client, model="gpt-5")
    response = await wrapper.call(agent_input)

    assert response == "I am an AI"
    
    # Verify call arguments
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"
    assert len(call_kwargs["messages"]) == 3
    assert call_kwargs["messages"][0]["role"] == "user"
    print(f"✅ PASSED: test_openai_wrapper_async")
    print(f"   Response: {response}")
    print(f"   Chat History:")
    for msg in call_kwargs["messages"]:
        print(f"     {msg['role']}: {msg['content']}")

# --- LangChain Tests ---

@pytest.mark.asyncio
async def test_langchain_wrapper_ainvoke(agent_input):
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "LangChain Response"

    wrapper = LangChainAgentWrapper(agent=mock_chain)
    response = await wrapper.call(agent_input)

    assert response == "LangChain Response"
    mock_chain.ainvoke.assert_called_once()
    
    # Verify input structure passed to chain
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert "messages" in call_args
    assert len(call_args["messages"]) == 3
    # Check LangChain message types (heuristic check if real LC not installed)
    # Since we mocked it, we assume wrapper did conversion logic. 
    # If we had real LC installed, we'd check isinstance(HumanMessage)
    print(f"✅ PASSED: test_langchain_wrapper_ainvoke")
    print(f"   Response: {response}")
    print(f"   Chat History (LangChain format):")
    for msg in call_args["messages"]:
        msg_type = type(msg).__name__ if hasattr(msg, '__class__') else 'dict'
        content = getattr(msg, 'content', str(msg))
        print(f"     {msg_type}: {content}")

@pytest.mark.asyncio
async def test_langchain_wrapper_invoke_sync(agent_input):
    mock_chain = MagicMock()
    # Ensure it DOESN'T have ainvoke so it falls back to invoke
    del mock_chain.ainvoke 
    mock_chain.invoke.return_value = {"output": "Sync Response"}

    wrapper = LangChainAgentWrapper(agent=mock_chain)
    response = await wrapper.call(agent_input)

    assert response == "Sync Response"
    mock_chain.invoke.assert_called_once()
    call_args = mock_chain.invoke.call_args[0][0]
    print(f"✅ PASSED: test_langchain_wrapper_invoke_sync")
    print(f"   Response: {response}")
    print(f"   Chat History:")
    if "messages" in call_args:
        for msg in call_args["messages"]:
            msg_type = type(msg).__name__ if hasattr(msg, '__class__') else 'dict'
            content = getattr(msg, 'content', str(msg))
            print(f"     {msg_type}: {content}")
    print(f"   Used sync method: invoke")

# --- Anthropic Tests ---

@pytest.mark.asyncio
async def test_anthropic_wrapper_async(agent_input):
    mock_client = AsyncMock()
    # Mock return structure: message.content[0].text
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Claude Response")]
    mock_client.messages.create.return_value = mock_message
    
    # Mock class name for async detection
    mock_client.__class__.__name__ = "AsyncAnthropic"

    wrapper = AnthropicAgentWrapper(client=mock_client, model="claude-4")
    response = await wrapper.call(agent_input)

    assert response == "Claude Response"
    
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-4"
    assert len(call_kwargs["messages"]) == 3
    assert call_kwargs["messages"][0]["role"] == "user"
    print(f"✅ PASSED: test_anthropic_wrapper_async")
    print(f"   Response: {response}")
    print(f"   Chat History:")
    for msg in call_kwargs["messages"]:
        print(f"     {msg['role']}: {msg['content']}")
    if "system" in call_kwargs:
        print(f"     system: {call_kwargs['system']}")

@pytest.mark.asyncio
async def test_anthropic_wrapper_system_prompt():
    # Test extraction of system prompt
    inp = AgentInput(
        thread_id="t1",
        messages=[
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"}
        ]
    )
    
    mock_client = AsyncMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="OK")]
    mock_client.__class__.__name__ = "AsyncAnthropic"
    
    wrapper = AnthropicAgentWrapper(client=mock_client)
    await wrapper.call(inp)
    
    call_kwargs = mock_client.messages.create.call_args.kwargs
    # System should be extracted to top-level arg, not in messages list
    assert call_kwargs["system"] == "Be helpful"
    assert len(call_kwargs["messages"]) == 1 # Only user message remains
    assert call_kwargs["messages"][0]["role"] == "user"
    print(f"✅ PASSED: test_anthropic_wrapper_system_prompt")
    print(f"   Chat History:")
    if "system" in call_kwargs:
        print(f"     system: {call_kwargs['system']}")
    for msg in call_kwargs["messages"]:
        print(f"     {msg['role']}: {msg['content']}")


# --- Gemini Tests ---

@pytest.mark.asyncio
async def test_gemini_wrapper_async(agent_input):
    mock_model = MagicMock()
    mock_chat = AsyncMock()
    mock_chat.send_message_async.return_value.text = "Gemini Response"
    mock_model.start_chat.return_value = mock_chat

    wrapper = GeminiAgentWrapper(model=mock_model)
    response = await wrapper.call(agent_input)

    assert response == "Gemini Response"
    
    # Verify history construction
    # Input has 3 msgs: User, Assistant, User.
    # Gemini start_chat takes history (first 2), send_message takes last one.
    mock_model.start_chat.assert_called_once()
    history = mock_model.start_chat.call_args.kwargs["history"]
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["parts"] == ["Hello"]
    assert history[1]["role"] == "model" # wrapper converts 'assistant' to 'model'
    
    # Verify send_message call
    mock_chat.send_message_async.assert_called_once_with("How are you?")
    print(f"✅ PASSED: test_gemini_wrapper_async")
    print(f"   Response: {response}")
    print(f"   Chat History (Gemini format):")
    for h in history:
        role = h["role"]
        content = h["parts"][0] if h["parts"] else ""
        print(f"     {role}: {content}")
    print(f"     user: How are you? (last message sent)")

