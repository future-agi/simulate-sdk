import sys
import os

# Add the project root to the Python path so that future_agi_sdk can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import uuid
import contextlib
from dotenv import load_dotenv
from future_agi_sdk import AgentDefinition, Scenario, Persona, TestRunner, evaluate_report
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents import Agent, AgentSession, function_tool
from livekit.plugins import openai, silero
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions

load_dotenv()


async def main():
    """
    Clean test runner that connects a simulated customer to your deployed agent,
    records audio via the SDK recorder, and runs evaluations.
    """
    livekit_url = os.environ.get("LIVEKIT_URL", "http://localhost:7880")
    room_name = os.environ.get("AGENT_ROOM_NAME", "test-room-001")

    agent_definition = AgentDefinition(
        name="deployed-support-agent",
        url=livekit_url,
        room_name=room_name,
        system_prompt="Helpful support agent",
    )

    scenario = Scenario(
        name="Customer Support Queries",
        dataset=[
            Persona(
                persona={"name": "Alice"},
                situation="Having trouble with account login.",
                outcome="Be guided through the password reset process.",
            ),
            Persona(
                persona={"name": "Bob"},
                situation="Internet drops during Zoom calls.",
                outcome="Stabilize his connection.",
            ),
        ],
    )

    runner = TestRunner()

    # Optionally spawn a local support agent so the simulator has someone to talk to
    spawn_local = os.environ.get("SPAWN_LOCAL_SUPPORT_AGENT", "1").lower() in ("1", "true", "yes")

    if not spawn_local:
        # Expect an existing deployed agent already in the room
        report = await runner.run_test(
            agent_definition,
            scenario,
            record_audio=True,
            recorder_sample_rate=8000,
            recorder_join_delay=0.1,
            max_seconds=300.0,
        )
    else:
        # Spawn a local support agent per persona using a fresh room each time
        class SupportAgent(Agent):
            def __init__(self, *, room: rtc.Room, **kwargs):
                super().__init__(**kwargs)
                self._room = room

            @function_tool()
            async def end_call(self) -> None:
                self.session.say("I'm glad I could help. Have a great day! Goodbye.")
                await asyncio.sleep(0.2)
                self.session.shutdown()
                # Disconnect room if still connected
                try:
                    if getattr(self._room, "isconnected", False):
                        if callable(self._room.isconnected):
                            if self._room.isconnected():
                                await self._room.disconnect()
                        elif self._room.isconnected:
                            await self._room.disconnect()
                except Exception:
                    pass

        async def run_support_agent(lk_url: str, lk_api_key: str, lk_api_secret: str, room_name: str):
            token = (
                AccessToken(lk_api_key, lk_api_secret)
                .with_identity("support-agent")
                .with_grants(VideoGrants(room_join=True, room=room_name))
                .to_jwt()
            )
            room = rtc.Room()
            await room.connect(lk_url, token)

            agent = SupportAgent(
                room=room,
                stt=openai.STT(),
                llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
                tts=openai.TTS(voice="alloy"),
                vad=silero.VAD.load(),
                allow_interruptions=True,
                min_endpointing_delay=0.4,
                max_endpointing_delay=2.2,
                instructions=(
                    "You are a helpful support agent. Be friendly and proactive. "
                    "Ask clarifying questions and provide step-by-step guidance. "
                    "Keep the conversation going for at least 6 turns unless the issue is resolved. "
                    "When the customer confirms their issue is resolved or they say they're done, "
                    "call the `end_call` tool to gracefully end the call."
                ),
            )

            session = AgentSession(
                stt=agent.stt,
                llm=agent.llm,
                tts=agent.tts,
                vad=None,
                turn_detection="stt",
                allow_interruptions=True,
                min_endpointing_delay=0.4,
                max_endpointing_delay=2.2,
                preemptive_generation=False,
                discard_audio_if_uninterruptible=False,
                min_interruption_duration=0.3,
            )
            await session.start(
                agent,
                room=room,
                room_input_options=RoomInputOptions(delete_room_on_close=False),
                room_output_options=RoomOutputOptions(transcription_enabled=False),
            )

            session.say("Hello! How can I help you today?")

            # Wait until session closes
            closed = asyncio.Event()
            session.on("close", lambda ev: closed.set())
            await closed.wait()
            # Ensure disconnect
            try:
                if getattr(room, "isconnected", False):
                    if callable(room.isconnected):
                        if room.isconnected():
                            await room.disconnect()
                    elif room.isconnected:
                        await room.disconnect()
            except Exception:
                pass

        # Aggregate report across personas
        from future_agi_sdk.simulation.models import TestReport
        full_report = TestReport()

        lk_api_key = os.environ.get("LIVEKIT_API_KEY")
        lk_api_secret = os.environ.get("LIVEKIT_API_SECRET")
        if not all([lk_api_key, lk_api_secret]):
            raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set to spawn a local support agent.")

        for p in scenario.dataset:
            room_unique = f"{room_name}-{p.persona.get('name','user').lower()}-{str(uuid.uuid4())[:8]}"

            # Start local support agent
            agent_task = asyncio.create_task(run_support_agent(livekit_url, lk_api_key, lk_api_secret, room_unique))
            await asyncio.sleep(2.0)

            # Run a single-persona test case into that room
            case_def = AgentDefinition(
                name=agent_definition.name,
                url=agent_definition.url,
                room_name=room_unique,
                system_prompt=agent_definition.system_prompt,
            )
            case_scn = Scenario(name=f"Case-{p.persona.get('name','user')}", dataset=[p])
            case_report = await runner.run_test(
                case_def,
                case_scn,
                record_audio=True,
                recorder_sample_rate=8000,
                recorder_join_delay=0.1,
                max_seconds=300.0,
            )

            full_report.results.extend(case_report.results)

            # Wait for the agent to finish
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(agent_task, timeout=10)
            if not agent_task.done():
                agent_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await agent_task

        report = full_report

    # Evaluate using transcript and combined audio (provided by SDK)
    try:
        eval_specs = [
            {"template": "task_completion", "map": {"input": "persona.situation", "output": "transcript"}},
            {"template": "tone", "map": {"output": "transcript"}},
            {"template": "audio_transcription", "map": {"audio": "audio_combined_path", "transcription": "transcript"}},
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
            print(f"Combined audio: {result.audio_combined_path}")
        if result.evaluation:
            print("Evaluation:")
            for k, v in result.evaluation.items():
                print(f"  - {k}: {v}")
    print("\n--- End of Report ---")


if __name__ == "__main__":
    asyncio.run(main())
