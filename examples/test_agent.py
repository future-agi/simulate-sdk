import os
import asyncio
import time
import requests
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fi.simulate import (
    AgentDefinition,
    Scenario,
    Persona,
    TestRunner,
    evaluate_report,
    SimulatorAgentDefinition,
)

load_dotenv()

async def main():
    """
    Cloud-ready simulator: connects a single customer (Fubar) to your deployed agent
    in LiveKit Cloud, records audio, and runs evaluations. No local support agent.
    """
    livekit_url = os.environ["LIVEKIT_URL"]           # e.g., wss://<your>.livekit.cloud
    room_name = os.environ.get("AGENT_ROOM_NAME", "screening-room")
    cloud_agent_id = os.environ.get("LIVEKIT_AGENT_ID", "")  # e.g., CA_bUanqLv5kXSf

    agent_definition = AgentDefinition(
        name="deployed-support-agent",
        url=livekit_url,
        room_name=room_name,          # must match the room your agent joins
        system_prompt="Helpful support agent",
    )

    scenario = Scenario(
        name= "Money Management",
        dataset=[
            Persona(
                persona={"name": "Fubar"},
                situation="Wants to check account status due to changes in interest rates",
                outcome="Understands current account status",
            ),
        ],
    )

    # Make the simulator start speaking proactively so the agent hears it even if it greets early
    sim = SimulatorAgentDefinition(
        instructions=(
            "Start the call by greeting the agent and briefly describing your issue. "
            "Do not wait to hear the agent first. Keep replies concise and natural."
        ),
        llm={"model": "gpt-4o-mini", "temperature": 0.6},
        tts={"model": "tts-1", "voice": "alloy"},
        stt={"language": "en"},
        vad={"provider": "silero"},
        allow_interruptions=True,
        min_endpointing_delay=0.3,
        max_endpointing_delay=2.2,
    )

    runner = TestRunner()

    # Optionally dispatch your LiveKit Cloud agent into the room before starting the test
    # Requires LIVEKIT_API_KEY/SECRET and LIVEKIT_AGENT_ID to be set
    if cloud_agent_id:
        try:
            from livekit import api

            class Dummy:
                pass

            # Use the API key/secret from environment
            lk_api_key = os.environ.get("LIVEKIT_API_KEY")
            lk_api_secret = os.environ.get("LIVEKIT_API_SECRET")
            if not (lk_api_key and lk_api_secret):
                raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set to dispatch agent.")

            # The agent_name is the agent ID in this context
            agent_name = cloud_agent_id

            async def create_explicit_dispatch():
                lkapi = api.LiveKitAPI(api_key=lk_api_key, api_secret=lk_api_secret)
                dispatch = await lkapi.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(
                        agent_name=agent_name,
                        room=room_name,
                        metadata='{"user_id": "12345"}'
                    )
                )
                print("created dispatch", dispatch)

                dispatches = await lkapi.agent_dispatch.list_dispatch(room_name=room_name)
                print(f"there are {len(dispatches)} dispatches in {room_name}")
                await lkapi.aclose()

            await create_explicit_dispatch()
            # Give the cloud agent a brief moment to spin up and join
            import asyncio
            await asyncio.sleep(2.0)
        except Exception as e:
            print(f"Agent dispatch failed (continuing anyway): {e}")
    report = await runner.run_test(
        agent_definition,
        scenario,
        simulator=sim,
        record_audio=True,
        recorder_sample_rate=8000,
        recorder_join_delay=0.1,
        max_seconds=300.0,
    )

    # Optional: evaluate transcript + audio
    try:
        eval_specs = [
            {"template": "task_completion", "map": {"input": "persona.situation", "output": "transcript"}},
            {"template": "tone",            "map": {"output": "transcript"}},
            {"template": "audio_quality", "map": {"input_audio": "audio_combined_path"}},
        ]
        report = evaluate_report(
            report,
            eval_specs=eval_specs,
            model_name="turing_large",
            api_key=os.environ.get("FI_API_KEY"),
            secret_key=os.environ.get("FI_SECRET_KEY"),
        )
    except Exception as e:
        print(f"Eval skipped: {e}")

    print("\n--- Test Report ---")
    for result in report.results:
        print(f"\n--- Persona: {result.persona.persona['name']} ---")
        print("Transcript:")
        print(result.transcript)
        if getattr(result, "audio_combined_path", None):
            print("Combined audio:", result.audio_combined_path)
        if result.evaluation:
            print("Evaluation:")
            for k, v in result.evaluation.items():
                print(f"  - {k}: {v}")
    print("\n--- End of Report ---")

if __name__ == "__main__":
    asyncio.run(main())
