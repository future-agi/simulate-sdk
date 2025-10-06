import sys
import os

# Add the project root to the Python path to allow importing the SDK
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from future_agi_sdk import AgentDefinition, Scenario, TestRunner, ScenarioGenerator
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    """
    An example of how to automatically generate test case personas and run a simulation.
    """
    # 1. Describe the deployed agent that you want to test.
    #    This should match the agent you have running from `run_deployed_agent.py`.
    agent_definition = AgentDefinition(
        name="deployed-insurance-agent",
        url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
        room_name="my-test-room", # Must match the room the agent is in
        system_prompt="You are a friendly and helpful insurance sales agent. Your goal is to understand the customer's needs and recommend a suitable policy."
    )

    # 2. Automatically generate test case personas
    print("Generating scenarios...")
    generator = ScenarioGenerator(agent_definition=agent_definition)
    generated_personas = await generator.generate(
        topic="Customers calling to inquire about life insurance policies. Include a mix of interested, price-sensitive, and skeptical customers.",
        num_personas=5
    )
    print(f"Successfully generated {len(generated_personas)} personas.")

    # 3. Create a Scenario from the generated personas
    scenario = Scenario(
        name="Generated Life Insurance Queries",
        dataset=generated_personas
    )

    # 4. Run the test with the generated scenario
    print("Starting test runner...")
    runner = TestRunner()
    report = runner.run_test(agent_definition, scenario)

    # 5. Print the report
    print("\n--- Test Report ---")
    for result in report.results:
        print(f"\n--- Persona: {result.persona.persona['name']} ---")
        print(f"Situation: {result.persona.situation}")
        print(f"Transcript:\n{result.transcript}")
    print("\n--- End of Report ---")

if __name__ == "__main__":
    asyncio.run(main())
