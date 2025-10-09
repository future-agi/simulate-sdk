from livekit import agents
from livekit.plugins import openai

class CustomerAgent:
    def __init__(self, persona_prompt: str, request: str):
        self.persona_prompt = persona_prompt
        self.request = request
        self.llm = openai.LLM(prompt=persona_prompt)
        self.stt = openai.STT()
        self.tts = openai.TTS()

    async def _job(self, room: agents.Room):
        """
        The main job loop for the customer agent.
        """
        # Start the conversation with the initial request
        await room.say(self.request)

        async for event in room.events():
            if isinstance(event, agents.events.SpeechData) and event.is_final and event.participant != room.identity:
                response = await self.llm.complete(prompt=event.text)
                await room.say(response.text)

    def to_job_request(self):
        return agents.JobRequest(
            entrypoint=self._job,
            name="customer-agent",
        )
