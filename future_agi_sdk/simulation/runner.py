from ..agent.definition import AgentDefinition
from .models import Scenario, Persona, TestReport, TestCaseResult
from livekit.agents import stt, tts, llm, vad, Agent, AgentSession
from livekit.plugins import openai, silero
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
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
        await session.start(self, room=room)

    async def get_session(self) -> AgentSession:
        return await self._session_future

class TestRunner:
    """
    Connects to a deployed agent and tests it against a scenario.
    """
    async def run_test(
        self,
        agent_definition: AgentDefinition,
        scenario: Scenario,
    ) -> TestReport:
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
            
            customer_agent = self._create_customer_agent(persona)
            
            # Start the agent in a background task
            session_task = asyncio.create_task(
                customer_agent.run(room=customer_room)
            )

            # Wait for the session to be created
            customer_session = await customer_agent.get_session()

            # Let the conversation run
            await asyncio.sleep(15)
            
            # Get transcript from history
            if customer_agent.session:
                print(f"DEBUG: customer_agent attributes: {dir(customer_agent)}")
                print(f"DEBUG: Session history has {len(customer_agent.session.history.items)} items.")
                transcript = "\n".join([f"{item.role}: {item.text}" for item in customer_agent.session.history.items])
            else:
                transcript = "Error: Agent session was not created."
            
        except Exception as e:
            print(f"Error during test case: {e}")
            return f"Error: {e}"
        finally:
            if customer_room.isconnected():
                await customer_room.disconnect()
                print(f"✓ Customer disconnected")
        
        return transcript

    def _create_customer_agent(self, persona: Persona) -> _TestRunnerAgent:
        customer_prompt = self._create_customer_prompt(persona)
        
        agent = _TestRunnerAgent(
            persona=persona,
            stt=openai.STT(),
            llm=openai.LLM(model="gpt-4o"),
            tts=openai.TTS(voice="onyx"),
            vad=silero.VAD.load(),
            instructions=customer_prompt,
        )
        return agent

    def _create_customer_prompt(self, persona: Persona) -> str:
        return f"You are a customer with the following characteristics: {persona.persona}. " \
               f"Currently, {persona.situation}. " \
               f"Your goal is to achieve this outcome: {persona.outcome}."
