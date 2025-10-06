from future_agi_sdk import AgentDefinition, Scenario, Persona, TestRunner
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    """
    An example of how to define a deployed agent and test it against a scenario.
    """
    # 1. Describe the deployed agent that is already running and waiting in a LiveKit room.
    #    You must have your own agent running for this example to work.
    agent_definition = AgentDefinition(
        name="deployed-support-agent",
        url=os.environ.get("LIVEKIT_URL", "wss://your-livekit-server.com"),
        room_name="your-agent-room-name", # Replace with the room your agent is in
    )

    # 2. Define the test scenario with different customer personas
    scenario = Scenario(
        name="Customer Support Queries",
        dataset=[
            Persona(
                persona={"name": "Alice"},
                situation="She is having trouble with her account login.",
                outcome="The agent should guide her through the password reset process.",
            ),
            Persona(
                persona={"name": "Bob"},
                situation="He has a question about billing.",
                outcome="The agent should provide him with the billing department's contact information.",
            ),
        ]
    )

    # 3. Run the test
    # The TestRunner will create a "simulated customer" for each persona and connect
    # to your agent's room to run the conversation.
    runner = TestRunner()
    report = runner.run_test(agent_definition, scenario)

    # 4. Print the report
    print("\n--- Test Report ---")
    for result in report.results:
        print(f"\n--- Persona: {result.persona.persona['name']} ---")
        print(f"Transcript:\n{result.transcript}")
    print("\n--- End of Report ---")

if __name__ == "__main__":
    main()
