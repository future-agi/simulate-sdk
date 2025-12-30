import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fi.simulate.agent.wrapper import AgentWrapper, AgentInput, AgentResponse
from fi.simulate.agent.wrappers.openai import OpenAIAgentWrapper
from fi.simulate.agent.wrappers.langchain import LangChainAgentWrapper
from fi.simulate.agent.wrappers.anthropic import AnthropicAgentWrapper
from fi.simulate.agent.wrappers.gemini import GeminiAgentWrapper
from fi.simulate.simulation.engines.cloud import CloudEngine

# --- Fixtures ---

@pytest.fixture
def agent_input():
    return AgentInput(
        thread_id="test-thread-123",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ],
        new_message={"role": "user", "content": "How are you?"},
        execution_id="exec-456"
    )

# --- Custom Class-Based Agent Tests ---

class EchoAgent(AgentWrapper):
    """Simple echo agent that repeats the last message."""
    
    async def call(self, input: AgentInput) -> str:
        if input.new_message:
            return f"Echo: {input.new_message['content']}"
        return "No message to echo"

class StatefulAgent(AgentWrapper):
    """Agent that maintains state across calls."""
    
    def __init__(self, prefix: str = "Agent"):
        self.prefix = prefix
        self.call_count = 0
    
    async def call(self, input: AgentInput) -> str:
        self.call_count += 1
        last_msg = input.messages[-1]["content"] if input.messages else ""
        return f"{self.prefix} (call #{self.call_count}): {last_msg}"

class CustomResponseAgent(AgentWrapper):
    """Agent that returns AgentResponse with metadata."""
    
    async def call(self, input: AgentInput) -> AgentResponse:
        content = f"Response to: {input.new_message['content'] if input.new_message else 'nothing'}"
        return AgentResponse(
            content=content,
            metadata={
                "thread_id": input.thread_id,
                "execution_id": input.execution_id,
                "message_count": len(input.messages)
            }
        )

@pytest.mark.asyncio
async def test_custom_class_based_agent_echo(agent_input):
    """Test a simple custom agent that inherits from AgentWrapper."""
    agent = EchoAgent()
    response = await agent.call(agent_input)
    
    assert response == "Echo: How are you?"
    print(f"✅ PASSED: test_custom_class_based_agent_echo")
    print(f"   Response: {response}")

@pytest.mark.asyncio
async def test_custom_class_based_agent_stateful(agent_input):
    """Test a stateful custom agent that maintains state."""
    agent = StatefulAgent(prefix="TestBot")
    
    # First call
    response1 = await agent.call(agent_input)
    assert agent.call_count == 1
    assert "TestBot (call #1)" in response1
    
    # Second call
    response2 = await agent.call(agent_input)
    assert agent.call_count == 2
    assert "TestBot (call #2)" in response2
    
    print(f"✅ PASSED: test_custom_class_based_agent_stateful")
    print(f"   Call 1: {response1}")
    print(f"   Call 2: {response2}")
    print(f"   Total calls: {agent.call_count}")

@pytest.mark.asyncio
async def test_custom_class_based_agent_with_response_object(agent_input):
    """Test custom agent returning AgentResponse object."""
    agent = CustomResponseAgent()
    response = await agent.call(agent_input)
    
    assert isinstance(response, AgentResponse)
    assert response.content.startswith("Response to:")
    assert response.metadata["thread_id"] == "test-thread-123"
    assert response.metadata["execution_id"] == "exec-456"
    assert response.metadata["message_count"] == 3
    
    print(f"✅ PASSED: test_custom_class_based_agent_with_response_object")
    print(f"   Content: {response.content}")
    print(f"   Metadata: {response.metadata}")

# --- Custom Function-Based Agent Tests ---

async def async_function_agent(input: AgentInput) -> str:
    """Simple async function agent."""
    return f"Function says: {input.new_message['content'] if input.new_message else 'nothing'}"

def sync_function_agent(input: AgentInput) -> str:
    """Simple sync function agent."""
    return f"Sync function says: {input.new_message['content'] if input.new_message else 'nothing'}"

@pytest.mark.asyncio
async def test_custom_function_based_agent_async(agent_input):
    """Test async function as agent callback."""
    response = await async_function_agent(agent_input)
    
    assert response.startswith("Function says:")
    print(f"✅ PASSED: test_custom_function_based_agent_async")
    print(f"   Response: {response}")

@pytest.mark.asyncio
async def test_custom_function_based_agent_sync(agent_input):
    """Test sync function as agent callback."""
    response = sync_function_agent(agent_input)
    
    assert response.startswith("Sync function says:")
    print(f"✅ PASSED: test_custom_function_based_agent_sync")
    print(f"   Response: {response}")

@pytest.mark.asyncio
async def test_cloud_engine_normalizes_function_callback(agent_input):
    """Test that CloudEngine._normalize_callback wraps functions correctly."""
    engine = CloudEngine(api_key="test-key", secret_key="test-secret")
    
    # Test async function
    wrapped_async = engine._normalize_callback(async_function_agent)
    assert isinstance(wrapped_async, AgentWrapper)
    response = await wrapped_async.call(agent_input)
    assert response.startswith("Function says:")
    
    # Test sync function
    wrapped_sync = engine._normalize_callback(sync_function_agent)
    assert isinstance(wrapped_sync, AgentWrapper)
    response = await wrapped_sync.call(agent_input)
    assert response.startswith("Sync function says:")
    
    print(f"✅ PASSED: test_cloud_engine_normalizes_function_callback")
    print(f"   Async wrapped: {type(wrapped_async).__name__}")
    print(f"   Sync wrapped: {type(wrapped_sync).__name__}")

# --- System Prompt Tests for Wrappers ---

@pytest.mark.asyncio
async def test_openai_wrapper_with_system_prompt(agent_input):
    """Test OpenAI wrapper includes system prompt."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="I am helpful"))
    ]
    mock_client.__class__.__name__ = "AsyncOpenAI"
    
    system_prompt = "You are a helpful assistant."
    wrapper = OpenAIAgentWrapper(client=mock_client, model="gpt-4", system_prompt=system_prompt)
    await wrapper.call(agent_input)
    
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"] == system_prompt
    assert len(call_kwargs["messages"]) == 4  # system + 3 original messages
    
    print(f"✅ PASSED: test_openai_wrapper_with_system_prompt")
    print(f"   System prompt: {system_prompt}")
    print(f"   Total messages: {len(call_kwargs['messages'])}")

@pytest.mark.asyncio
async def test_anthropic_wrapper_with_system_prompt(agent_input):
    """Test Anthropic wrapper includes system prompt."""
    mock_client = AsyncMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="OK")]
    mock_client.__class__.__name__ = "AsyncAnthropic"
    
    system_prompt = "You are a helpful assistant."
    wrapper = AnthropicAgentWrapper(client=mock_client, model="claude-3", system_prompt=system_prompt)
    await wrapper.call(agent_input)
    
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == system_prompt
    
    print(f"✅ PASSED: test_anthropic_wrapper_with_system_prompt")
    print(f"   System prompt: {system_prompt}")
    print(f"   System param set: {'system' in call_kwargs}")

@pytest.mark.asyncio
async def test_langchain_wrapper_with_system_prompt(agent_input):
    """Test LangChain wrapper includes system prompt."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "LC Response"
    
    system_prompt = "You are a helpful assistant."
    wrapper = LangChainAgentWrapper(agent=mock_chain, system_prompt=system_prompt)
    await wrapper.call(agent_input)
    
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert "messages" in call_args
    # First message should be SystemMessage
    first_msg = call_args["messages"][0]
    assert hasattr(first_msg, 'content') or isinstance(first_msg, dict)
    # Check if it's a SystemMessage (heuristic: check content or type)
    if hasattr(first_msg, 'content'):
        assert first_msg.content == system_prompt
    
    print(f"✅ PASSED: test_langchain_wrapper_with_system_prompt")
    print(f"   System prompt: {system_prompt}")
    print(f"   Total messages: {len(call_args['messages'])}")

@pytest.mark.asyncio
async def test_gemini_wrapper_with_system_prompt(agent_input):
    """Test Gemini wrapper includes system prompt."""
    mock_model = MagicMock()
    mock_chat = AsyncMock()
    mock_chat.send_message_async.return_value.text = "Gemini Response"
    mock_model.start_chat.return_value = mock_chat
    
    system_prompt = "You are a helpful assistant."
    wrapper = GeminiAgentWrapper(model=mock_model, system_prompt=system_prompt)
    await wrapper.call(agent_input)
    
    # Check that history includes system prompt
    history = mock_model.start_chat.call_args.kwargs["history"]
    assert len(history) > 0
    # System prompt should be prepended as first user message
    first_entry = history[0]
    assert first_entry["role"] == "user"
    assert system_prompt in first_entry["parts"][0]
    
    print(f"✅ PASSED: test_gemini_wrapper_with_system_prompt")
    print(f"   System prompt: {system_prompt}")
    print(f"   History length: {len(history)}")

# --- Edge Cases ---

@pytest.mark.asyncio
async def test_custom_agent_with_empty_messages():
    """Test custom agent handles empty message history."""
    agent = EchoAgent()
    input_empty = AgentInput(
        thread_id="empty-thread",
        messages=[],
        new_message=None
    )
    
    response = await agent.call(input_empty)
    assert response == "No message to echo"
    print(f"✅ PASSED: test_custom_agent_with_empty_messages")
    print(f"   Response: {response}")

@pytest.mark.asyncio
async def test_custom_agent_accesses_full_history(agent_input):
    """Test custom agent can access full conversation history."""
    class HistoryAwareAgent(AgentWrapper):
        async def call(self, input: AgentInput) -> str:
            history_summary = f"History has {len(input.messages)} messages"
            last_user_msg = None
            for msg in reversed(input.messages):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break
            return f"{history_summary}. Last user said: {last_user_msg}"
    
    agent = HistoryAwareAgent()
    response = await agent.call(agent_input)
    
    assert "History has 3 messages" in response
    assert "Last user said: How are you?" in response
    
    print(f"✅ PASSED: test_custom_agent_accesses_full_history")
    print(f"   Response: {response}")

@pytest.mark.asyncio
async def test_custom_agent_with_execution_id(agent_input):
    """Test custom agent can access execution_id for context."""
    class ContextAwareAgent(AgentWrapper):
        async def call(self, input: AgentInput) -> str:
            return f"Thread: {input.thread_id}, Execution: {input.execution_id}"
    
    agent = ContextAwareAgent()
    response = await agent.call(agent_input)
    
    assert "Thread: test-thread-123" in response
    assert "Execution: exec-456" in response
    
    print(f"✅ PASSED: test_custom_agent_with_execution_id")
    print(f"   Response: {response}")

