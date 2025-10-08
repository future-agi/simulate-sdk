import sys
import os
import asyncio
import contextlib

from ..fi.simulate import (
    AgentDefinition,
    TestRunner,
    SimulatorAgentDefinition,
)
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents import Agent, AgentSession, function_tool
from livekit.plugins import openai, silero
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions

load_dotenv()

# Configuration
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "http://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
TEST_ROOM = "test-room-001"
STOP_EVENT: asyncio.Event | None = None


class SupportAgent(Agent):
    def __init__(self, *, room: rtc.Room, **kwargs):
        super().__init__(**kwargs)
        self._room = room

    @function_tool()
    async def end_call(self) -> None:
        self.session.say("I'm glad I could help. Have a great day! Goodbye.")
        await asyncio.sleep(0.2)
        self.session.shutdown()
        if getattr(self._room, "isconnected", False):
            try:
                if callable(self._room.isconnected):
                    if self._room.isconnected():
                        await self._room.disconnect()
                elif self._room.isconnected:
                    await self._room.disconnect()
            except Exception:
                pass

    async def on_user_turn_completed(self, turn_ctx, new_message):
        text = (getattr(new_message, "text_content", None) or "").lower()
        if any(kw in text for kw in ["thanks", "thank you", "that's all", "thats all", "done", "goodbye", "bye"]):
            await self.end_call()


async def run_support_agent():
    print("Starting support agent...")

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("support-agent")
        .with_grants(VideoGrants(room_join=True, room=TEST_ROOM))
        .to_jwt()
    )

    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    print(f"✓ Support agent connected to room: {TEST_ROOM}")

    agent = SupportAgent(
        room=room,
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o", temperature=0.7),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
        instructions=(
            "You are a helpful support agent. Be friendly and proactive. "
            "Ask clarifying questions and provide step-by-step guidance. "
            "Keep the conversation going for at least 6 turns unless the issue is resolved. "
            "When the customer confirms their issue is resolved or they say they're done, "
            "call the `end_call` tool to gracefully end the call."
        ),
    )

    session = AgentSession(stt=agent.stt, llm=agent.llm, tts=agent.tts, vad=agent.vad)
    await session.start(
        agent,
        room=room,
        room_input_options=RoomInputOptions(delete_room_on_close=False),
        room_output_options=RoomOutputOptions(transcription_enabled=False),
    )

    session.say("Hello! How can I help you today?")
    print("Support agent is ready and waiting in the room...")

    closed = asyncio.Event()
    def _on_close(ev):
        closed.set()
    session.on("close", _on_close)

    try:
        global STOP_EVENT
        if STOP_EVENT is None:
            STOP_EVENT = asyncio.Event()
        done, pending = await asyncio.wait(
            [asyncio.create_task(closed.wait()), asyncio.create_task(STOP_EVENT.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if STOP_EVENT.is_set() and not closed.is_set():
            session.shutdown()
            await closed.wait()
    finally:
        try:
            if getattr(room, "isconnected", False):
                if callable(room.isconnected):
                    if room.isconnected():
                        await room.disconnect()
                elif room.isconnected:
                    await room.disconnect()
        except Exception:
            pass


async def run_test():
    print("\nStarting test with generated scenario...")

    agent_definition = AgentDefinition(
        name="Support Agent",
        url=LIVEKIT_URL,
        room_name=TEST_ROOM,
        system_prompt="Helpful support agent",
    )

    # Lightweight simulator config
    simulator = SimulatorAgentDefinition(
        name="sim-customer",
        instructions=(
            "You are a concise, realistic customer. Ask clarifying questions, keep responses short, "
            "avoid repetition, and explicitly end with 'Thanks, that's all' when satisfied."
        ),
        llm={"model": "gpt-4o-mini", "temperature": 0.6},
        tts={"model": "tts-1", "voice": "alloy"},
        stt={"language": "en"},
        vad={"provider": "silero"},
        allow_interruptions=True,
        min_endpointing_delay=0.3,
        max_endpointing_delay=4.0,
        use_tts_aligned_transcript=False,
    )

    runner = TestRunner()
    report = await runner.run_test(
        agent_definition,
        None,              # scenario=None triggers generation
        simulator,
        num_scenarios=1,   # generate 1 scenario
        topic=None,         # let runner synthesize topic from agent/simulator
    )

    print(f"\n{'='*60}")
    print("GENERATED SCENARIO")
    print(f"{'='*60}")
    for result in report.results:
        print(f"Persona: {result.persona.persona}")
        print(f"Situation: {result.persona.situation}")
        print(f"Outcome: {result.persona.outcome}")

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    for result in report.results:
        print(f"\nPersona: {result.persona.persona.get('name', 'Unknown')}")
        print(f"\nTranscript:")
        print(result.transcript)
    print(f"\n{'='*60}\n")


async def main():
    agent_task = asyncio.create_task(run_support_agent())
    await asyncio.sleep(2)
    try:
        await run_test()
    finally:
        global STOP_EVENT
        if STOP_EVENT is None:
            STOP_EVENT = asyncio.Event()
        STOP_EVENT.set()
        try:
            await asyncio.wait_for(agent_task, timeout=10)
        except asyncio.TimeoutError:
            agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await agent_task


if __name__ == "__main__":
    asyncio.run(main())


