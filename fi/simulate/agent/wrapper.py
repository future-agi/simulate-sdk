from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod

class AgentInput(BaseModel):
    """
    Input data passed to the user's agent wrapper during a simulation step.
    """
    thread_id: str
    messages: List[Dict[str, str]]  # Full conversation history: [{"role": "user", "content": "..."}]
    new_message: Optional[Dict[str, str]] = None # The latest message to respond to
    
    # Metadata for execution context (useful for logging/debugging)
    execution_id: Optional[str] = None

class AgentResponse(BaseModel):
    """
    Standardized response from the user's agent.
    """
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_responses: Optional[List[Dict[str, Any]]] = None  # Tool role messages with results
    metadata: Optional[Dict[str, Any]] = None

class AgentWrapper(ABC):
    """
    Base class for wrapping user agents to work with the simulation SDK.
    Users should implement the `call` method.
    """
    
    @abstractmethod
    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        """
        Process the input and return the agent's response.
        
        Args:
            input: The AgentInput object containing message history and context.
            
        Returns:
            A string (content only) or AgentResponse object.
        """
        pass

