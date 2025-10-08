from livekit import agents
from livekit.plugins import deepgram, openai, elevenlabs
from .agent_definition import InternalAgentDefinition

def create_agent_from_definition(agent_definition: InternalAgentDefinition):
    """
    Creates a LiveKit agent from an InternalAgentDefinition.
    """
    stt = None
    if agent_definition.stt_config.provider == "deepgram":
        stt = deepgram.STT()

    llm = None
    if agent_definition.llm_config.provider == "openai":
        llm = openai.LLM(prompt=agent_definition.instructions)

    tts = None
    if agent_definition.tts_config.provider == "elevenlabs":
        tts = elevenlabs.TTS()

    if not all([stt, llm, tts]):
        raise ValueError("Unsupported provider configuration")

    async def job(room: agents.Room):
        """
        The main job loop for the agent.
        """
        async for event in room.events():
            if isinstance(event, agents.events.SpeechData) and event.is_final:
                response = await llm.complete(prompt=event.text)
                await room.say(response.text)

    return agents.JobRequest(
        entrypoint=job,
        name=f"agent-{agent_definition.llm_config.provider}",
    )
