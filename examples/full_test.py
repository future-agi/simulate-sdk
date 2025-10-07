import sys
import os
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from future_agi_sdk import AgentDefinition, Scenario, Persona, TestRunner
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv()

# Configuration
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "http://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
TEST_ROOM = "test-room-001"

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
    
    # Create and configure the support agent
    agent = Agent(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
        instructions="You are a helpful support agent. Be friendly and solve customer problems.",
    )
    
    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    
    print(f"✓ Support agent connected to room: {TEST_ROOM}")
    
    # Note: AgentSession initialization may vary by version
    # We'll keep the agent running by keeping the room connected
    print("Support agent is ready and waiting in the room...")
    
    # Keep the connection alive for the test to run
    await asyncio.sleep(45)  # Wait for 45 seconds

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
        # Clean up
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main())

