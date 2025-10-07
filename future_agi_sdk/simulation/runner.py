from ..agent.definition import AgentDefinition, SimulatorAgentDefinition
from .models import Scenario, Persona, TestReport, TestCaseResult
from livekit.agents import stt, tts, llm, vad, Agent, AgentSession
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import openai, silero
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents.voice import ModelSettings
from livekit.agents.voice.io import TimedString
from typing import AsyncIterable
from .generator import ScenarioGenerator
import asyncio
import os

class _TestRunnerAgent(Agent):
    """
    An agent used by the TestRunner to simulate a customer.
    """
    def __init__(self, persona: Persona, **kwargs):
        super().__init__(**kwargs)
        self._persona = persona
        self._session_future = asyncio.Future()

    async def run(self, room: rtc.Room):
        session = AgentSession(
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            vad=self.vad,
        )
        self._session_future.set_result(session)
        await session.start(
            self,
            room=room,
            room_input_options=RoomInputOptions(delete_room_on_close=False),
            room_output_options=RoomOutputOptions(transcription_enabled=False),
        )

    async def get_session(self) -> AgentSession:
        return await self._session_future

    async def transcription_node(
        self,
        text: AsyncIterable[str | TimedString],
        model_settings: ModelSettings,
    ):
        async for chunk in text:
            if isinstance(chunk, TimedString):
                print(f"ASR: '{chunk}' ({getattr(chunk, 'start_time', None)} - {getattr(chunk, 'end_time', None)})")
            else:
                print(f"LLM: {chunk}")
            yield chunk

class TestRunner:
    """
    Connects to a deployed agent and tests it against a scenario.
    """
    async def run_test(
        self,
        agent_definition: AgentDefinition,
        scenario: Scenario | None,
        simulator: SimulatorAgentDefinition | None = None,
        num_scenarios: int = 1,
        topic: str | None = None,
    ) -> TestReport:
        # If no scenario provided, generate personas using generator
        if scenario is None:
            gen = ScenarioGenerator(agent_definition)
            # Build a simple topic from provided context if none given
            if topic is None:
                agent_ctx = agent_definition.system_prompt
                sim_ctx = simulator.instructions if simulator and simulator.instructions else ""
                topic = (sim_ctx or agent_ctx or "customer support scenarios").strip()
            personas = await gen.generate(topic=topic, num_personas=num_scenarios)
            scenario = Scenario(name="Generated Scenario", dataset=personas)

        report = TestReport()
        for persona in scenario.dataset:
            print(f"Running test case for persona: {persona.persona.get('name', 'Unknown')}")
            
            transcript = await self._run_single_test_case(agent_definition, persona, simulator)
            
            report.results.append(
                TestCaseResult(
                    persona=persona,
                    transcript=transcript,
                )
            )
            
        return report

    async def _run_single_test_case(
        self,
        agent_definition: AgentDefinition,
        persona: Persona,
        simulator: SimulatorAgentDefinition | None,
    ) -> str:
        livekit_api_key = os.environ.get("LIVEKIT_API_KEY")
        livekit_api_secret = os.environ.get("LIVEKIT_API_SECRET")

        if not all([livekit_api_key, livekit_api_secret]):
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set.")

        customer_room = rtc.Room()
        
        try:
            token = (
                AccessToken(livekit_api_key, livekit_api_secret)
                .with_identity(persona.persona.get("name", "customer"))
                .with_grants(VideoGrants(room_join=True, room=agent_definition.room_name))
                .to_jwt()
            )

            await customer_room.connect(url=str(agent_definition.url), token=token)
            print(f"✓ Customer '{persona.persona.get('name')}' connected to room")
            
            customer_agent = self._create_customer_agent(persona, simulator)
            
            # Start the agent in a background task
            session_task = asyncio.create_task(
                customer_agent.run(room=customer_room)
            )

            # Wait for the session to be created
            customer_session = await customer_agent.get_session()

            # Stream transcripts and messages in real-time
            def _on_user_input_transcribed(ev):
                try:
                    suffix = "" if getattr(ev, "is_final", False) else "…"
                    print(f"ASR(user): {getattr(ev, 'transcript', '')}{suffix}")
                except Exception:
                    pass

            def _on_conversation_item_added(ev):
                try:
                    item = getattr(ev, "item", None)
                    role = getattr(item, "role", None)
                    text = getattr(item, "text_content", None)
                    if role and text:
                        print(f"MSG({role}): {text}")
                except Exception:
                    pass

            customer_session.on("user_input_transcribed", _on_user_input_transcribed)
            customer_session.on("conversation_item_added", _on_conversation_item_added)

            # Wait for the agent session to close naturally
            closed = asyncio.Event()
            def _on_close(ev):
                closed.set()
            customer_session.on("close", _on_close)
            await closed.wait()
            
            # Get transcript from history (dedupe partial repeats)
            if customer_session:
                lines: list[str] = []
                last_by_role: dict[str, str] = {}
                for item in customer_session.history.items:
                    item_type = getattr(item, "type", None)
                    role = getattr(item, "role", None)
                    text = getattr(item, "text_content", None)
                    if item_type == "message" and text is not None and role is not None:
                        prev = last_by_role.get(role)
                        # Deduplicate streaming partials by collapsing near-duplicates
                        if prev and (text.startswith(prev) or prev.startswith(text)):
                            # Replace last line for this role
                            for i in range(len(lines) - 1, -1, -1):
                                if lines[i].startswith(f"{role}:"):
                                    lines[i] = f"{role}: {text}"
                                    break
                        else:
                            lines.append(f"{role}: {text}")
                        last_by_role[role] = text
                transcript = "\n".join(lines)
            else:
                transcript = "Error: Agent session was not created."
            
        except Exception as e:
            print(f"Error during test case: {e}")
            return f"Error: {e}"
        finally:
            # Support both property and method across versions
            try:
                if getattr(customer_room, "isconnected", False):
                    if callable(customer_room.isconnected):
                        if customer_room.isconnected():
                            await customer_room.disconnect()
                    elif customer_room.isconnected:
                        await customer_room.disconnect()
                elif getattr(customer_room, "is_connected", False):
                    if customer_room.is_connected:
                        await customer_room.disconnect()
            except Exception:
                pass
                print(f"✓ Customer disconnected")
        
        return transcript

    def _create_customer_agent(self, persona: Persona, simulator: SimulatorAgentDefinition | None) -> _TestRunnerAgent:
        customer_prompt = self._create_customer_prompt(persona)

        # Build components from simulator config or use sensible defaults
        if simulator is None:
            stt_model = openai.STT(language="en")
            llm_model = openai.LLM(model="gpt-4o-mini", temperature=0.6)
            tts_model = openai.TTS(model="tts-1", voice="alloy")
            vad_model = silero.VAD.load()
            instructions = customer_prompt
            allow_interruptions = None
            min_ep = None
            max_ep = None
            use_aligned = None
        else:
            stt_model = openai.STT(language=simulator.stt.language)
            llm_model = openai.LLM(model=simulator.llm.model, temperature=simulator.llm.temperature)
            tts_model = openai.TTS(model=simulator.tts.model, voice=simulator.tts.voice)
            vad_model = silero.VAD.load()
            # Merge simulator instructions with persona-derived prompt so both are applied
            if simulator.instructions:
                instructions = f"{simulator.instructions}\n\n{customer_prompt}"
            else:
                instructions = customer_prompt
            allow_interruptions = simulator.allow_interruptions
            min_ep = simulator.min_endpointing_delay
            max_ep = simulator.max_endpointing_delay
            use_aligned = simulator.use_tts_aligned_transcript

        agent = _TestRunnerAgent(
            persona=persona,
            stt=stt_model,
            llm=llm_model,
            tts=tts_model,
            vad=vad_model,
            instructions=instructions,
            allow_interruptions=allow_interruptions,
            min_endpointing_delay=min_ep,
            max_endpointing_delay=max_ep,
            use_tts_aligned_transcript=use_aligned,
        )
        return agent

    def _create_customer_prompt(self, persona: Persona) -> str:
        return (
            "You are a realistic customer in a support call. "
            f"Profile: {persona.persona}. "
            f"Situation: {persona.situation}. "
            f"Goal: {persona.outcome}. "
            "Have a natural back-and-forth conversation, asking clarifying questions. "
            "Keep the conversation going for at least 6 turns unless the problem is fully solved. "
            "When you are satisfied and done, explicitly say: 'Thanks, that's all'. "
            "Use short, spoken-style sentences."
        )
