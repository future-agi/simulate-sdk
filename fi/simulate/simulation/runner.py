from typing import Optional, Callable
import os

from ..agent.definition import AgentDefinition, SimulatorAgentDefinition
from .models import Scenario, TestReport
from .engines import LiveKitEngine, BaseEngine, CloudEngine

class TestRunner:
    """
    Main entry point for running agent simulations.
    
    Supports two execution modes:
    1. Local mode (LiveKit): Uses LiveKit to connect to deployed agents
    2. Cloud mode (Backend API): Uses Future AGI backend for orchestrated testing
    
    The mode is automatically determined based on the arguments provided.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        """
        Initialize the TestRunner.
        
        Args:
            api_key: Optional API key for cloud mode. If not provided, will check FI_API_KEY env var.
            api_url: Optional API URL for cloud mode. If not provided, will check FI_API_URL env var.
        """
        self.api_key = api_key or os.environ.get("FI_API_KEY")
        self.api_url = api_url or os.environ.get("FI_API_URL")
        self._engine: Optional[BaseEngine] = None

    async def run_test(
        self,
        # --- Local Mode Arguments (LiveKit) ---
        agent_definition: Optional[AgentDefinition] = None,
        scenario: Optional[Scenario] = None,
        simulator: Optional[SimulatorAgentDefinition] = None,
        
        # --- Cloud Mode Arguments (Backend API) ---
        run_id: Optional[str] = None,
        agent_callback: Optional[Callable] = None,
        
        # --- Shared Arguments ---
        num_scenarios: int = 1,
        topic: Optional[str] = None,
        record_audio: bool = False,
        recorder_sample_rate: int = 8000,
        recorder_join_delay: float = 0.2,
        min_turn_messages: int = 8,
        max_seconds: float = 45.0,
        **kwargs
    ) -> TestReport:
        """
        Run a test simulation.
        
        Mode is determined by arguments:
        - If `run_id` is provided → Cloud mode (Backend API)
        - If `agent_definition` is provided → Local mode (LiveKit)
        
        Args:
            agent_definition: Agent configuration for local mode
            scenario: Test scenario for local mode
            simulator: Simulator configuration for local mode
            run_id: Run ID from platform for cloud mode
            agent_callback: User's agent function to wrap for cloud mode
            num_scenarios: Number of scenarios to generate (local mode only)
            topic: Topic for scenario generation (local mode only)
            record_audio: Whether to record audio
            recorder_sample_rate: Audio sample rate
            recorder_join_delay: Delay before recorder joins
            min_turn_messages: Minimum turn messages
            max_seconds: Maximum test duration
            **kwargs: Additional arguments passed to engine
        
        Returns:
            TestReport with results from all test cases
        """
        # Dispatch to appropriate engine
        if run_id is not None:
            # Cloud mode - Use CloudEngine
            engine = CloudEngine(self.api_key, self.api_url)
            return await engine.run(
                run_id=run_id,
                agent_callback=agent_callback,
                **kwargs
            )
            
        elif agent_definition is not None:
            # Local mode - use LiveKit engine
            engine = LiveKitEngine()
            return await engine.run(
                agent_definition=agent_definition,
                scenario=scenario,
                simulator=simulator,
                num_scenarios=num_scenarios,
                topic=topic,
                record_audio=record_audio,
                recorder_sample_rate=recorder_sample_rate,
                recorder_join_delay=recorder_join_delay,
                min_turn_messages=min_turn_messages,
                max_seconds=max_seconds,
                **kwargs
            )
        else:
            raise ValueError(
                "Must provide either 'agent_definition' (Local/LiveKit mode) "
                "or 'run_id' (Cloud/Backend API mode)."
            )
