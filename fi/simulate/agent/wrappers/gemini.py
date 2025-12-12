from typing import Any, Union, List, Dict
from ..wrapper import AgentWrapper, AgentInput, AgentResponse

class GeminiAgentWrapper(AgentWrapper):
    """
    Wrapper for Google Gemini (Generative AI) agents.
    Supports google-generativeai SDK.
    """
    def __init__(self, model: Any):
        """
        Args:
            model: An instance of google.generativeai.GenerativeModel
        """
        self.model = model

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        # Convert internal messages to Gemini format (Content objects)
        # Note: Gemini SDK manages chat history via ChatSession usually,
        # but for stateless call we pass full history if supported, 
        # or we might need to reconstruct a chat session.
        
        # Simple reconstruction of history for a chat session
        history = []
        last_message = None
        
        for msg in input.messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            
            # Gemini typically expects history excluding the last message which is passed to send_message
            history.append({"role": role, "parts": [content]})
            
        if not history:
            raise ValueError("No messages provided to Gemini wrapper")

        # The last user message is the prompt
        last_turn = history.pop()
        if last_turn["role"] != "user":
            # If the last message wasn't user, something is weird in the flow,
            # but we can try to send empty or handle it. 
            # Ideally simulator sends User message last.
            prompt = ""
        else:
            prompt = last_turn["parts"][0]

        # Start a chat with the history
        chat = self.model.start_chat(history=history)
        
        # Check if async generation is supported (google-generativeai >= 0.3.0 has send_message_async)
        if hasattr(chat, "send_message_async"):
            response = await chat.send_message_async(prompt)
        else:
            # Fallback to sync
            response = chat.send_message(prompt)
            
        return response.text

