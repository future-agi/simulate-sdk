import sys
import os
import asyncio
import contextlib

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from future_agi_sdk import AgentDefinition, Scenario, Persona, TestRunner
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents import Agent, AgentSession, function_tool
from livekit.plugins import openai, silero
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.agents.voice.room_io import RoomOutputOptions

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
        # Let the goodbye play a moment, then close and leave the room
        await asyncio.sleep(0.2)
        self.session.shutdown()
        if getattr(self._room, "isconnected", False):
            try:
                # isconnected may be property or method depending on version
                if callable(self._room.isconnected):
                    if self._room.isconnected():
                        await self._room.disconnect()
                elif self._room.isconnected:
                    await self._room.disconnect()
            except Exception:
                pass

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # If the user indicates they're done, end the call naturally
        text = (getattr(new_message, "text_content", None) or "").lower()
        if any(kw in text for kw in ["thanks", "thank you", "that's all", "thats all", "done", "goodbye", "bye"]):
            await self.end_call()

async def run_support_agent():
    """
    This simulates the 'agent-under-test' - the agent we want to test.
    In a real scenario, this would be the user's deployed agent.
    """
    print("Starting support agent...")
    
    # Generate token for the agent
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("support-agent")
        .with_grants(VideoGrants(room_join=True, room=TEST_ROOM))
        .to_jwt()
    )
    
    # Create and connect the room first
    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)

    print(f"✓ Support agent connected to room: {TEST_ROOM}")

    # Create and configure the support agent (after room exists)
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

    # Audio recording disabled for now

    # Start a Voice session for the agent and send a greeting
    session = AgentSession(
        stt=agent.stt,
        llm=agent.llm,
        tts=agent.tts,
        vad=agent.vad,
    )
    await session.start(
        agent,
        room=room,
        room_input_options=RoomInputOptions(delete_room_on_close=False),
        room_output_options=RoomOutputOptions(transcription_enabled=False),
    )

    # Local audio recording disabled for now

    session.say("Hello! How can I help you today?")
    print("Support agent is ready and waiting in the room...")
    
    # Wait for the agent session to close naturally (via end_call tool) or STOP_EVENT
    closed = asyncio.Event()
    def _on_close(ev):
        closed.set()
    session.on("close", _on_close)
    try:
        global STOP_EVENT
        if STOP_EVENT is None:
            STOP_EVENT = asyncio.Event()
        # Wait for either session close or external stop
        done, pending = await asyncio.wait(
            [asyncio.create_task(closed.wait()), asyncio.create_task(STOP_EVENT.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # If we were asked to stop explicitly, shutdown gracefully and wait for close
        if STOP_EVENT.is_set() and not closed.is_set():
            self_shutdown = True
            session.shutdown()
            await closed.wait()
    finally:
        # If still connected for any reason, disconnect the room
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
    """
    This uses our SDK to test the deployed agent.
    """
    print("\nStarting test with Future AGI Simulate SDK...")
    
    # 1. Define the agent we're testing
    agent_definition = AgentDefinition(
        name="Support Agent",
        url=LIVEKIT_URL,
        room_name=TEST_ROOM,
        system_prompt="Helpful support agent",
    )

    # 2. Define test scenarios
    scenario = Scenario(
        name="Support Test",
        dataset=[
            Persona(
                persona={"name": "Alice"},
                situation="She cannot log into her account",
                outcome="Get help with password reset",
            ),
        ]
    )

    # 3. Run the test
    runner = TestRunner()
    report = await runner.run_test(agent_definition, scenario)

    # 4. Print results
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    for result in report.results:
        print(f"\nPersona: {result.persona.persona['name']}")
        print(f"\nTranscript:")
        print(result.transcript)
    print(f"\n{'='*60}\n")

async def main():
    """
    Run both the support agent and the test in sequence.
    """
    # Start the support agent in a background task
    agent_task = asyncio.create_task(run_support_agent())
    
    # Wait a moment for the agent to connect
    await asyncio.sleep(2)
    
    # Run the test (this will block until complete)
    try:
        await run_test()
    finally:
        # Signal support agent to shutdown gracefully
        global STOP_EVENT
        if STOP_EVENT is None:
            STOP_EVENT = asyncio.Event()
        STOP_EVENT.set()
        # Give it a moment to drain and exit
        try:
            await asyncio.wait_for(agent_task, timeout=10)
        except asyncio.TimeoutError:
            agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await agent_task

if __name__ == "__main__":
    asyncio.run(main())

