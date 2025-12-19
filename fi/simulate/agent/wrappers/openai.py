from typing import Any, Union
from ..wrapper import AgentWrapper, AgentInput, AgentResponse

class OpenAIAgentWrapper(AgentWrapper):
    """
    Wrapper for OpenAI-based agents.
    Automatically handles message conversion to OpenAI format.
    """
    def __init__(self, client: Any, model: str = "gpt-4-turbo", system_prompt: str = None):
        """
        Args:
            client: The OpenAI client instance (AsyncOpenAI or OpenAI).
            model: The model name to use (e.g., "gpt-4-turbo").
            system_prompt: Optional system instructions for the agent.
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        # Convert internal message format to OpenAI format
        # Input messages are already in [{"role": "...", "content": "..."}] format which OpenAI accepts
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
            
        messages.extend(input.messages)
        
        # Check if client is async or sync
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
             # Handle AsyncOpenAI vs OpenAI
            if hasattr(self.client.chat.completions, "create"):
                 # Modern OpenAI SDK (v1+)
                if is_async_client(self.client):
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages
                    )
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages
                    )
                
                return completion.choices[0].message.content
        
        raise ValueError("Unsupported OpenAI client. Please provide a valid OpenAI or AsyncOpenAI client.")

def is_async_client(client: Any) -> bool:
    """Check if the client is an async client."""
    # Heuristic check for async client
    return type(client).__name__ == "AsyncOpenAI"

