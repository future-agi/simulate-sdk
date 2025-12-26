from typing import Any, Union
from fi.simulate.agent.wrapper import AgentWrapper, AgentInput, AgentResponse

class AnthropicAgentWrapper(AgentWrapper):
    """
    Wrapper for Anthropic (Claude) agents.
    Automatically handles message conversion to Anthropic format.
    """
    def __init__(self, client: Any, model: str = "claude-3-5-sonnet-20240620", system_prompt: str = None):
        """
        Args:
            client: The Anthropic client instance (AsyncAnthropic or Anthropic).
            model: The model name to use.
            system_prompt: Optional system instructions for the agent.
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        # Convert internal message format to Anthropic format
        # Anthropic messages API expects: [{"role": "user"|"assistant", "content": "..."}]
        # It does NOT support "system" role in the messages list; system prompt is a top-level param.
        
        messages = []
        # Use configured system prompt by default
        system_prompt = self.system_prompt
        
        for msg in input.messages:
            if msg["role"] == "system":
                # If history has system message (unlikely due to filtering), it overrides? 
                # Or we ignore it to respect wrapper config? 
                # Let's check if it exists and use it if self.system_prompt is None
                if system_prompt is None:
                    system_prompt = msg["content"]
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Check for AsyncAnthropic vs Sync
        # Heuristic: check for 'messages.create' and if client class name contains Async
        is_async = type(self.client).__name__.startswith("Async")
        
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": messages
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        if is_async:
            message = await self.client.messages.create(**kwargs)
        else:
            message = self.client.messages.create(**kwargs)

        return message.content[0].text

