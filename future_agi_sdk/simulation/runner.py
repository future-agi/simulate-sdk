from ..agent.definition import AgentDefinition
from .models import Scenario, Persona, TestReport, TestCaseResult
from livekit.agents import stt, tts, llm, vad, Agent, AgentSession
from livekit.plugins import openai, silero
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.protocol import room
import asyncio
import os

class TranscriptCollector:
    def __init__(self):
        self._parts = []
        self._lock = asyncio.Lock()

    async def add_part(self, role: str, content: str):
        async with self._lock:
            self._parts.append(f"{role}: {content}")

    async def get_transcript(self) -> str:
        async with self._lock:
            return "\n".join(self._parts)

class TestRunner:
    """
    Connects to a deployed agent and tests it against a scenario.
    """
    async def run_test(
        self,
        agent_definition: AgentDefinition,
        scenario: Scenario,
    ) -> TestReport:
        """
        Executes the entire simulation against all test cases in a scenario.
        """
        report = TestReport()
        for persona in scenario.dataset:
            print(f"Running test case for persona: {persona.persona.get('name', 'Unknown')}")
            
            transcript = await self._run_single_test_case(agent_definition, persona)
            
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
    ) -> str:
        """
        Runs a single conversation between a simulated customer and the deployed agent.
        """
        livekit_api_key = os.environ.get("LIVEKIT_API_KEY")
        livekit_api_secret = os.environ.get("LIVEKIT_API_SECRET")

        if not all([livekit_api_key, livekit_api_secret]):
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set.")

        collector = TranscriptCollector()
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
            
            customer_agent = self._create_customer_agent(persona)
            customer_session = AgentSession()

            @customer_session.on("conversation_item_added")
            def on_conversation_item(item: llm.ChatItem):
                asyncio.create_task(collector.add_part(item.role, item.text))

            # Start the session in a background task
            session_task = asyncio.create_task(
                customer_session.start(customer_agent, room=customer_room)
            )

            # Let the conversation run for a set duration
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"Error during test case: {e}")
            return f"Error: {e}"
        finally:
            if customer_room.isconnected():
                await customer_room.disconnect()
                print(f"✓ Customer disconnected")
        
        return await collector.get_transcript()

    def _create_customer_agent(self, persona: Persona) -> Agent:
        customer_prompt = self._create_customer_prompt(persona)
        
        agent = Agent(
            stt=openai.STT(),
            llm=openai.LLM(model="gpt-4o"),
            tts=openai.TTS(voice="onyx"),
            vad=silero.VAD.load(),
            instructions=customer_prompt,
        )
        return agent

    def _create_customer_prompt(self, persona: Persona) -> str:
        """
        Generates the system prompt for the simulated customer.
        """
        return f"You are a customer with the following characteristics: {persona.persona}. " \
               f"Currently, {persona.situation}. " \
               f"Your goal is to achieve this outcome: {persona.outcome}."
