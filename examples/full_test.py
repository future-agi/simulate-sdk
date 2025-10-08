import sys
import os
import asyncio
import contextlib
from glob import glob
import numpy as np
import wave

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from future_agi_sdk import AgentDefinition, Scenario, Persona, TestRunner, evaluate_report
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents import Agent, AgentSession, function_tool
from livekit.plugins import openai, silero
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.agents.voice.room_io import RoomOutputOptions
from livekit.agents.voice.recorder_io import RecorderIO
from livekit.agents.voice import recorder_io as _recorder_mod

load_dotenv()

# Configuration
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "http://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
TEST_ROOM = "test-room-001"
STOP_EVENT: asyncio.Event | None = None
RECORD_AUDIO: bool = False  # disable built-in RecorderIO; SDK recorder is used by TestRunner
LAST_RECORDING_PATH: str | None = None
RECORD_STREAM: bool = False  # disable participant polling capture; we use subscription-based capture
LAST_RECORDING_WAV: str | None = None
LAST_RECORDING_WAV_MIC: str | None = None
LAST_RECORDING_WAV_TTS: str | None = None
STT_RECORDING_WAV: str | None = None
AGENT_TTS_RECORDING_WAV: str | None = None
REMOTE_AUDIO_PATHS: dict[str, str] = {}

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

    async def stt_node(self, audio, model_settings):
        # Use default behavior; no local disk tap to reduce overhead
        return Agent.default.stt_node(self, audio, model_settings)

    async def tts_node(self, text, model_settings):
        # Use default behavior; no local disk tap to reduce overhead
        return Agent.default.tts_node(self, text, model_settings)

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

    # Start a Voice session for the agent and send a greeting
    session = AgentSession(
        stt=agent.stt,
        llm=agent.llm,
        tts=agent.tts,
        vad=None,  # disable VAD to avoid bounce EOU path
        # Use STT-based turn detection and balanced endpointing
        turn_detection="stt",
        allow_interruptions=True,
        min_endpointing_delay=0.4,
        max_endpointing_delay=2.2,
        preemptive_generation=False,
        discard_audio_if_uninterruptible=False,
        min_interruption_duration=0.2,
    )
    await session.start(
        agent,
        room=room,
        room_input_options=RoomInputOptions(delete_room_on_close=False),
        room_output_options=RoomOutputOptions(transcription_enabled=False),
    )
    # Reinforce session options on the live session
    try:
        session.update_options(min_endpointing_delay=0.4, max_endpointing_delay=2.2)
    except Exception:
        pass

    recorder = None
    recording_path = None
    if RECORD_AUDIO:
        # Disabled by default due to potential memory usage; set RECORD_AUDIO=True to enable
        os.makedirs("recordings", exist_ok=True)
        recording_path = os.path.join("recordings", f"{TEST_ROOM}.ogg")
        try:
            _recorder_mod.WRITE_INTERVAL = 1.0
        except Exception:
            pass
        recorder = RecorderIO(agent_session=session, sample_rate=8000)
        # Record only output to reduce alignment/memory issues
        session.output.audio = recorder.record_output(session.output.audio)  # type: ignore[arg-type]
        try:
            await recorder.start(output_path=recording_path)
            print(f"✓ Local recording started: {recording_path}")
            global LAST_RECORDING_PATH
            LAST_RECORDING_PATH = recording_path
        except Exception as e:
            print(f"Local recording failed: {e}")

    session.say("Hello! How can I help you today?")
    print("Support agent is ready and waiting in the room...")

    # Wait for the agent session to close naturally (via end_call tool) or STOP_EVENT
    closed = asyncio.Event()
    def _on_close(ev):
        closed.set()
    session.on("close", _on_close)

    # Recording is handled by a separate recorder process; no in-room subscribe here to avoid interference

    async def write_wav_from_remote(_room: rtc.Room, _path: str, _stop: asyncio.Event):
        # wait for any remote participant to appear
        target = None
        for _ in range(100):  # ~10s
            if _room.remote_participants:
                target = next(iter(_room.remote_participants.values()))
                break
            await asyncio.sleep(0.1)
        if not target:
            print("AudioStream: no remote participant found; skipping capture")
            return
        async def _record_source(_source: rtc.TrackSource, suffix: str) -> str | None:
            try:
                stream = rtc.AudioStream.from_participant(
                    participant=target,
                    track_source=_source,
                    sample_rate=16000,
                    num_channels=1,
                )
            except Exception as e:
                print(f"AudioStream init failed for {_source}: {e}")
                return None
            path = _path.replace(".wav", f"-{suffix}.wav")
            try:
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    start = asyncio.get_event_loop().time()
                    wrote = 0
                    async for ev in stream:
                        wf.writeframes(ev.frame.data)
                        wrote += len(ev.frame.data)
                        if _stop.is_set() and (asyncio.get_event_loop().time() - start) > 2.0:
                            break
                return path if wrote > 0 else None
            finally:
                await stream.aclose()

        # try mic and tts-like (unknown) sources in parallel
        mic_path, tts_path = await asyncio.gather(
            _record_source(rtc.TrackSource.SOURCE_MICROPHONE, "mic"),
            _record_source(rtc.TrackSource.SOURCE_UNKNOWN, "tts"),
        )
        global LAST_RECORDING_WAV_MIC, LAST_RECORDING_WAV_TTS
        if mic_path:
            LAST_RECORDING_WAV_MIC = mic_path
        if tts_path:
            LAST_RECORDING_WAV_TTS = tts_path
    try:
        record_task = None
        if RECORD_STREAM:
            os.makedirs("recordings", exist_ok=True)
            wav_path = os.path.join("recordings", f"{TEST_ROOM}.wav")
            record_task = asyncio.create_task(write_wav_from_remote(room, wav_path, closed))
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
        # Close persistent agent TTS writer if open
        try:
            if hasattr(agent, "_tts_wav_writer") and agent._tts_wav_writer is not None:
                agent._tts_wav_writer.close()
                agent._tts_wav_writer = None
        except Exception:
            pass
        # Stop recorder
        if recorder is not None:
            with contextlib.suppress(Exception):
                await recorder.aclose()
        # Wait for stream capture to finish
        if RECORD_STREAM and 'record_task' in locals() and record_task is not None:
            with contextlib.suppress(Exception):
                await record_task

def _find_room_recordings(room_name: str, identity: str | None = None, output_dir: str = "recordings") -> list[str]:
    pattern = f"{room_name}-*-track-*.wav" if identity is None else f"{room_name}-{identity}-track-*.wav"
    paths = glob(os.path.join(output_dir, pattern))
    # newest first
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)


def _pick_best_recording(paths: list[str]) -> str | None:
    if not paths:
        return None
    # prefer longest (by bytes), break ties by newest mtime
    return max(paths, key=lambda p: (os.path.getsize(p), os.path.getmtime(p)))


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
                persona={"name": "Alice", "age": 34, "communication_style": "polite, a bit anxious"},
                situation="She recently moved to a new city and her support app is showing notifications in the wrong timezone",
                outcome="Get the app to display notifications in her new local time",
            ),
            Persona(
                persona={"name": "Bob"},
                situation="He is experiencing issues with his internet connection dropping frequently while using Zoom",
                outcome="Get assistance to stabilize his internet connection",
            ),
            Persona(
                persona={"name": "Carol"},
                situation="She wants to upgrade her subscription plan for Apple Music but is unsure about the available options",
                outcome="Receive clear guidance and successfully upgrade her plan for Apple Music",
            ),
        ]
    )

    # 3. Run the test
    runner = TestRunner()
    # Enable SDK recorder participant via TestRunner
    report = await runner.run_test(
        agent_definition,
        scenario,
        record_audio=True,
        recorder_sample_rate=8000,
        recorder_join_delay=0.1,
        min_turn_messages=12,
        max_seconds=120.0,
    )

    # Attach separate paths and build combined recording BEFORE evaluation
    for r in report.results:
        try:
            persona_name = r.persona.persona.get("name")
        except Exception:
            persona_name = None

        customer_paths = _find_room_recordings(TEST_ROOM, identity=persona_name) if persona_name else []
        agent_paths = _find_room_recordings(TEST_ROOM, identity="support-agent")

        if not getattr(r, "audio_input_path", None):
            r.audio_input_path = _pick_best_recording(customer_paths)
        if not getattr(r, "audio_output_path", None):
            r.audio_output_path = _pick_best_recording(agent_paths)

    # Build a single combined WAV
    combined_path: str | None = None
    try:
        mix_inputs: list[str] = []
        for r in report.results:
            if getattr(r, "audio_input_path", None):
                mix_inputs.append(r.audio_input_path)
            if getattr(r, "audio_output_path", None):
                mix_inputs.append(r.audio_output_path)
        if not mix_inputs:
            mix_inputs = _find_room_recordings(TEST_ROOM)
        mix_inputs = [p for p in mix_inputs if p and os.path.basename(p) != f"{TEST_ROOM}-combined.wav"]

        if mix_inputs:
            os.makedirs("recordings", exist_ok=True)
            combined_path = os.path.join("recordings", f"{TEST_ROOM}-combined.wav")
            arrays = []
            max_len = 0
            for p in mix_inputs:
                with wave.open(p, "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(frames, dtype=np.int16)
                    arrays.append(arr)
                    if arr.shape[0] > max_len:
                        max_len = arr.shape[0]
            if arrays and max_len > 0:
                mix = np.zeros(max_len, dtype=np.int32)
                for arr in arrays:
                    if arr.shape[0] < max_len:
                        pad = np.zeros(max_len - arr.shape[0], dtype=arr.dtype)
                        arr = np.concatenate([arr, pad])
                    mix += arr.astype(np.int32)
                mix = np.clip(mix, -32768, 32767).astype(np.int16)
                with wave.open(combined_path, "wb") as wf_out:
                    wf_out.setnchannels(1)
                    wf_out.setsampwidth(2)
                    wf_out.setframerate(8000)
                    wf_out.writeframes(mix.tobytes())
                print(f"✓ Combined conversation saved: {combined_path}")
    except Exception as e:
        print(f"Combined mix failed: {e}")

    # Attach combined path to each result (same file)
    if combined_path:
        for r in report.results:
            r.audio_combined_path = combined_path

    # Evaluate with ai-evaluation (requires FI_API_KEY/FI_SECRET_KEY)
    try:
        eval_specs = [
            {"template": "task_completion", "map": {"input": "persona.situation", "output": "transcript"}},
            {"template": "tone", "map": {"input": "transcript"}},
            # Example audio eval (template name depends on your templates)
            {"template": "audio_transcription_accuracy", "map": {"audio_path": "audio_output_path"}},
        ]
        report = evaluate_report(
            report,
            eval_specs=eval_specs,
            model_name="turing_flash",
            api_key=os.environ.get("FI_API_KEY"),
            secret_key=os.environ.get("FI_SECRET_KEY"),
        )
    except Exception as e:
        print(f"Eval skipped: {e}")

    # 5. Print results
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    for result in report.results:
        print(f"\nPersona: {result.persona.persona['name']}")
        print(f"\nTranscript:")
        print(result.transcript)
        if result.evaluation:
            print("\nEvaluation:")
            for k, v in result.evaluation.items():
                print(f"  - {k}: {v}")
    print(f"\n{'='*60}\n")

async def main():
    """
    Run both the support agent and the test in sequence.
    """
    # Start the support agent in a background task
    agent_task = asyncio.create_task(run_support_agent())

    # No passive recorder participant; relying on track_subscribed within the agent's room
    
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
        # Combined WAV already built earlier in run_test(); nothing to do here

if __name__ == "__main__":
    asyncio.run(main())

