from typing import Any, Union
from ..wrapper import AgentWrapper, AgentInput, AgentResponse

class AnthropicAgentWrapper(AgentWrapper):
    """
    Wrapper for Anthropic (Claude) agents.
    Automatically handles message conversion to Anthropic format.
    """
    def __init__(self, client: Any, model: str = "claude-4-sonnet-20250219"):
        """
        Args:
            client: The Anthropic client instance (AsyncAnthropic or Anthropic).
            model: The model name to use.
        """
        self.client = client
        self.model = model

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        # Convert internal message format to Anthropic format
        # Anthropic messages API expects: [{"role": "user"|"assistant", "content": "..."}]
        # It does NOT support "system" role in the messages list; system prompt is a top-level param.
        
        messages = []
        system_prompt = None
        
        for msg in input.messages:
            if msg["role"] == "system":
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

