import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fi.simulate.simulation.runner import TestRunner
from fi.simulate.agent.definition import AgentDefinition
from fi.simulate.agent.wrapper import AgentWrapper

# Mock AgentDefinition since it's a Pydantic model required for input
class MockAgentDefinition(AgentDefinition):
    # Bypass validation for test simplicity if needed, or just provide valid data
    pass

@pytest.fixture
def valid_agent_def():
    return AgentDefinition(
        name="test-agent",
        url="ws://localhost:7880",
        room_name="test-room",
        system_prompt="You are a helper."
    )

@pytest.fixture
def mock_agent_wrapper():
    wrapper = MagicMock(spec=AgentWrapper)
    return wrapper

@pytest.mark.asyncio
async def test_runner_dispatches_to_livekit_engine(valid_agent_def):
    # Patch the engines where TestRunner imports them
    with patch("fi.simulate.simulation.runner.LiveKitEngine") as MockLiveKitEngine:
        # Setup mock engine instance
        mock_engine_instance = AsyncMock()
        MockLiveKitEngine.return_value = mock_engine_instance
        mock_engine_instance.run.return_value = "livekit_report"

        runner = TestRunner()
        
        # Act: Call with agent_definition (Trigger Local Mode)
        result = await runner.run_test(agent_definition=valid_agent_def)

        # Assert
        assert result == "livekit_report"
        MockLiveKitEngine.assert_called_once()
        mock_engine_instance.run.assert_called_once()
        print("✅ PASSED: test_runner_dispatches_to_livekit_engine")

@pytest.mark.asyncio
async def test_runner_dispatches_to_cloud_engine(mock_agent_wrapper):
    with patch("fi.simulate.simulation.runner.CloudEngine") as MockCloudEngine:
        # Setup mock engine instance
        mock_engine_instance = AsyncMock()
        MockCloudEngine.return_value = mock_engine_instance
        mock_engine_instance.run.return_value = "cloud_report"

        runner = TestRunner(api_key="sk-test", api_url="http://api.test")
        
        # Act: Call with run_id (Trigger Cloud Mode)
        result = await runner.run_test(run_id="run_123", agent_callback=mock_agent_wrapper)

        # Assert
        assert result == "cloud_report"
        # Check CloudEngine was initialized with correct config
        MockCloudEngine.assert_called_once_with("sk-test", "http://api.test")
        # Check run was called with correct args
        mock_engine_instance.run.assert_called_once()
        call_kwargs = mock_engine_instance.run.call_args.kwargs
        assert call_kwargs["run_id"] == "run_123"
        assert call_kwargs["agent_callback"] == mock_agent_wrapper
        print("✅ PASSED: test_runner_dispatches_to_cloud_engine")

@pytest.mark.asyncio
async def test_runner_raises_error_if_ambiguous():
    runner = TestRunner()
    
    # Act & Assert: Call with nothing
    with pytest.raises(ValueError, match="Must provide either"):
        await runner.run_test()
    print("✅ PASSED: test_runner_raises_error_if_ambiguous (No args)")

# Note: The logic for "cloud mode requires agent_callback" is currently inside 
# CloudEngine.run() or implicit in the type signature, but TestRunner allows passing it through.
# If CloudEngine validates it, we assume TestRunner just passes it along.

