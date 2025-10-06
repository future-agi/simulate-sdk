import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from future_agi_sdk import AgentDefinition, Scenario, Persona, TestRunner
from dotenv import load_dotenv

load_dotenv()

def main():
    """
    Example: Testing a deployed voice AI agent with the Future AGI Simulate SDK.
    
    Prerequisites:
    - You must have your agent deployed and connected to a LiveKit room.
    - Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET in your .env file.
    - Set OPENAI_API_KEY in your .env file (for the simulated customer).
    """
    
    # 1. Define the agent you want to test (already deployed)
    agent_definition = AgentDefinition(
        name="my-support-agent",
        url=os.environ.get("LIVEKIT_URL", "wss://your-livekit-server.com"),
        room_name="agent-room", # The room where your agent is waiting
        system_prompt="You are a helpful support agent.",
    )

    # 2. Define test scenarios
    scenario = Scenario(
        name="Customer Support Test",
        description="Test the agent's ability to handle common customer queries",
        dataset=[
            Persona(
                persona={"name": "Alice", "mood": "frustrated"},
                situation="She cannot log into her account.",
                outcome="The agent should guide her through password reset.",
            ),
            Persona(
                persona={"name": "Bob", "mood": "confused"},
                situation="He has a billing question.",
                outcome="The agent should provide billing information or escalate.",
            ),
        ]
    )

    # 3. Run the test
    print("Starting test run...")
    runner = TestRunner()
    report = runner.run_test(agent_definition, scenario)

    # 4. Print results
    print("\n" + "="*60)
    print("TEST REPORT")
    print("="*60)
    for result in report.results:
        print(f"\nPersona: {result.persona.persona['name']}")
        print(f"Situation: {result.persona.situation}")
        print(f"Expected Outcome: {result.persona.outcome}")
        print(f"\nTranscript:")
        print("-" * 60)
        print(result.transcript)
        print("-" * 60)
    
    print("\n" + "="*60)
    print(f"Test completed. {len(report.results)} conversations recorded.")
    print("="*60)

if __name__ == "__main__":
    main()

