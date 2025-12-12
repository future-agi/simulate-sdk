import asyncio
import os
import contextvars
import logging
from typing import Optional, Callable, Dict, Any

from ...agent.definition import AgentDefinition, SimulatorAgentDefinition
from ...agent.wrapper import AgentWrapper, AgentInput, AgentResponse
from ..models import Scenario, TestReport, TestCaseResult, Persona
from .base import BaseEngine

# Context variable to track the current execution ID for future tool mocking
current_execution_id = contextvars.ContextVar("current_execution_id", default=None)

logger = logging.getLogger(__name__)

class CloudEngine(BaseEngine):
    """
    Execution engine that connects to the Future AGI backend to orchestrate simulations.
    It acts as a bridge between the cloud-hosted simulator and the user's local agent.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FI_API_KEY")
        self.api_url = api_url or os.environ.get("FI_API_URL") or "https://api.futureagi.com"
        
        if not self.api_key:
            # We don't raise immediately to allow instantiation, but run() will fail
            logger.warning("FI_API_KEY not provided. CloudEngine will not function correctly.")

    async def run(
        self,
        run_id: Optional[str] = None,
        agent_callback: Optional[Callable | AgentWrapper] = None,
        **kwargs
    ) -> TestReport:
        """
        Connects to the cloud run, receives user inputs, calls the agent_callback,
        and sends responses back.
        """
        if not run_id:
            raise ValueError("CloudEngine requires a 'run_id'.")
        
        if not agent_callback:
            raise ValueError("CloudEngine requires an 'agent_callback' (function or AgentWrapper).")

        # Normalize the callback to a callable that accepts AgentInput
        wrapper = self._normalize_callback(agent_callback)

        print(f"🚀 Starting Cloud Simulation for Run ID: {run_id}")
        
        # TODO: Real implementation will:
        # 1. Connect to WebSocket/Stream at self.api_url/runs/{run_id}/connect
        # 2. Listen for "turn" events
        # 3. Invoke wrapper.call() in a context-aware task
        # 4. Send results back
        
        # Mocking the loop for now to demonstrate structure
        report = await self._mock_cloud_loop(run_id, wrapper)
        
        return report

    def _normalize_callback(self, callback: Callable | AgentWrapper) -> AgentWrapper:
        """Ensures we have a AgentWrapper instance."""
        if isinstance(callback, AgentWrapper):
            return callback
        
        # If it's a function, wrap it
        class FunctionalWrapper(AgentWrapper):
            def __init__(self, func):
                self.func = func
            
            async def call(self, input: AgentInput) -> str | AgentResponse:
                # Support both async and sync functions
                if asyncio.iscoroutinefunction(self.func):
                    return await self.func(input)
                return self.func(input)
                
        return FunctionalWrapper(callback)

    async def _mock_cloud_loop(self, run_id: str, wrapper: AgentWrapper) -> TestReport:
        """
        Temporary mock loop to simulate backend interaction until API is ready.
        """
        # Simulate receiving a few turns from the cloud
        print("(Simulating connection to cloud...)")
        await asyncio.sleep(1)
        
        # Fake a test case
        execution_id = f"exec-{run_id}-1"
        thread_id = "thread-1"
        
        # Set context for this execution
        token = current_execution_id.set(execution_id)
        
        try:
            # Turn 1: Cloud sends user message
            user_msg = "Hello, I need help with my order."
            print(f"\n[Cloud -> SDK] User: {user_msg}")
            
            inp = AgentInput(
                thread_id=thread_id,
                messages=[{"role": "user", "content": user_msg}],
                new_message={"role": "user", "content": user_msg},
                execution_id=execution_id
            )
            
            # SDK calls user agent
            response = await wrapper.call(inp)
            
            # Normalize response
            content = response.content if isinstance(response, AgentResponse) else str(response)
            print(f"[SDK -> Cloud] Agent: {content}")
            
            # Turn 2
            user_msg_2 = "It's order #12345."
            print(f"\n[Cloud -> SDK] User: {user_msg_2}")
             
            inp_2 = AgentInput(
                thread_id=thread_id,
                messages=[
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": user_msg_2}
                ],
                new_message={"role": "user", "content": user_msg_2},
                execution_id=execution_id
            )
            
            response_2 = await wrapper.call(inp_2)
            content_2 = response_2.content if isinstance(response_2, AgentResponse) else str(response_2)
            print(f"[SDK -> Cloud] Agent: {content_2}")

        finally:
            current_execution_id.reset(token)

        # Return an empty report for now as the cloud calculates metrics
        return TestReport(results=[])

