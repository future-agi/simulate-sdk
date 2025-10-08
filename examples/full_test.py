import sys
import os
import asyncio
import contextlib
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
RECORD_AUDIO: bool = False  # disable OGG stereo recorder (can be memory heavy)
LAST_RECORDING_PATH: str | None = None
RECORD_STREAM: bool = True  # lightweight capture via AudioStream (mono)
LAST_RECORDING_WAV: str | None = None
LAST_RECORDING_WAV_MIC: str | None = None
LAST_RECORDING_WAV_TTS: str | None = None
STT_RECORDING_WAV: str | None = None
AGENT_TTS_RECORDING_WAV: str | None = None
REMOTE_AUDIO_PATHS: dict[str, str] = {}
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
        # Tap the incoming audio and write to WAV while preserving default STT behavior
        os.makedirs("recordings", exist_ok=True)
        path = os.path.join("recordings", f"{TEST_ROOM}-stt.wav")

        async def tapped_audio():
            wf = None
            try:
                async for frame in audio:
                    if wf is None:
                        wf = wave.open(path, "wb")
                        wf.setnchannels(frame.num_channels)
                        wf.setsampwidth(2)  # int16
                        wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.data)
                    yield frame
            finally:
                if wf is not None:
                    wf.close()
                global STT_RECORDING_WAV
                STT_RECORDING_WAV = path

        # Use session-level settings; forward original model_settings
        return Agent.default.stt_node(self, tapped_audio(), model_settings)

    async def tts_node(self, text, model_settings):
        # Tap the outgoing agent speech (TTS) and write to WAV while preserving default behavior
        os.makedirs("recordings", exist_ok=True)
        path = os.path.join("recordings", f"{TEST_ROOM}-agent-tts.wav")

        stream = Agent.default.tts_node(self, text, model_settings)
        if asyncio.iscoroutine(stream):
            stream = await stream

        async def tapped_frames():
            # Persistent WAV writer across utterances to allow appending with silence gaps
            silence_gap_s = 0.4
            if not hasattr(self, "_tts_wav_writer"):
                self._tts_wav_writer = None
                self._tts_has_written = False
                self._tts_sample_rate = None
                self._tts_num_channels = None
            try:
                wrote_any = False
                async for frame in stream:
                    # Initialize writer on first frame or when not yet opened
                    if self._tts_wav_writer is None:
                        self._tts_wav_writer = wave.open(path, "wb")
                        self._tts_wav_writer.setnchannels(frame.num_channels)
                        self._tts_wav_writer.setsampwidth(2)  # int16
                        self._tts_wav_writer.setframerate(frame.sample_rate)
                        self._tts_sample_rate = frame.sample_rate
                        self._tts_num_channels = frame.num_channels
                    # If first frame of this utterance and we've written previous utterances, add a gap once
                    if not wrote_any and self._tts_has_written:
                        gap_samples = int((self._tts_sample_rate or frame.sample_rate) * silence_gap_s)
                        if gap_samples > 0:
                            self._tts_wav_writer.writeframes(b"\x00\x00" * gap_samples * (self._tts_num_channels or frame.num_channels))
                    # Write current audio frame
                    self._tts_wav_writer.writeframes(frame.data)
                    wrote_any = True
                    yield frame
            finally:
                global AGENT_TTS_RECORDING_WAV
                AGENT_TTS_RECORDING_WAV = path
                # Mark that we've written at least one utterance to enable gaps for subsequent ones
                if wrote_any:
                    self._tts_has_written = True

        return tapped_frames()

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
        llm=openai.LLM(model="gpt-4o", temperature=0.7),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
        allow_interruptions=True,
        min_endpointing_delay=0.3,
        max_endpointing_delay=4.0,
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
        # Use STT-based turn detection and ensure endpointing delays are numeric
        turn_detection="stt",
        min_endpointing_delay=0.3,
        max_endpointing_delay=4.0,
    )
    await session.start(
        agent,
        room=room,
        room_input_options=RoomInputOptions(delete_room_on_close=False),
        room_output_options=RoomOutputOptions(transcription_enabled=False),
    )
    # Reinforce numeric endpointing delays on the live session
    try:
        session.update_options(min_endpointing_delay=0.3, max_endpointing_delay=4.0)
    except Exception:
        pass

    recorder = None
    recording_path = None
    if RECORD_AUDIO:
        # Disabled by default due to potential memory usage; set RECORD_AUDIO=True to enable
        os.makedirs("recordings", exist_ok=True)
        recording_path = os.path.join("recordings", f"{TEST_ROOM}.ogg")
        try:
            _recorder_mod.WRITE_INTERVAL = 0.2
        except Exception:
            pass
        recorder = RecorderIO(agent_session=session, sample_rate=16000)
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

    # Record remote audio tracks as they are subscribed (robust capture for participants)
    @room.on("track_subscribed")
    def _on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        try:
            if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
                return
            os.makedirs("recordings", exist_ok=True)
            path = os.path.join("recordings", f"{TEST_ROOM}-{participant.identity}-track-{publication.sid}.wav")

            async def _record_track():
                try:
                    stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
                except Exception as e:
                    print(f"AudioStream(track) init failed: {e}")
                    return
                wrote = 0
                try:
                    with wave.open(path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        async for ev in stream:
                            wf.writeframes(ev.frame.data)
                            wrote += len(ev.frame.data)
                finally:
                    with contextlib.suppress(Exception):
                        await stream.aclose()
                if wrote > 0:
                    REMOTE_AUDIO_PATHS[participant.identity] = path

            asyncio.create_task(_record_track())
        except Exception:
            pass

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

    # Attach audio paths produced by the support agent (same file contains both channels)
    # In this simple example, we set both input/output paths to the same OGG file.
    # Downstream eval templates can decide which channel to use.
    # Attach separate paths: customer input (STT tap or MIC) and agent output (agent TTS tap)
    for r in report.results:
        # Prefer identity-based remote file for the persona if available
        try:
            persona_name = r.persona.persona.get("name")
        except Exception:
            persona_name = None
        if persona_name and persona_name in REMOTE_AUDIO_PATHS:
            r.audio_input_path = REMOTE_AUDIO_PATHS[persona_name]

        # input side preference: STT tap (customer audio), then MIC, then TTS from remote, then OGG
        if not r.audio_input_path and STT_RECORDING_WAV:
            r.audio_input_path = STT_RECORDING_WAV
        elif not r.audio_input_path and LAST_RECORDING_WAV_MIC:
            r.audio_input_path = LAST_RECORDING_WAV_MIC
        elif not r.audio_input_path and LAST_RECORDING_WAV_TTS:
            r.audio_input_path = LAST_RECORDING_WAV_TTS
        elif not r.audio_input_path and LAST_RECORDING_PATH:
            r.audio_input_path = LAST_RECORDING_PATH

        # output side preference: agent TTS tap, then support-agent remote track, then TTS from remote, then OGG
        if AGENT_TTS_RECORDING_WAV:
            r.audio_output_path = AGENT_TTS_RECORDING_WAV
        elif "support-agent" in REMOTE_AUDIO_PATHS:
            r.audio_output_path = REMOTE_AUDIO_PATHS["support-agent"]
        elif LAST_RECORDING_WAV_TTS:
            r.audio_output_path = LAST_RECORDING_WAV_TTS
        elif LAST_RECORDING_PATH:
            r.audio_output_path = LAST_RECORDING_PATH

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

    # 4. Print results
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
    
    # Also join as a passive recorder participant to subscribe to all remote tracks
    async def run_room_recorder():
        try:
            token = (
                AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                .with_identity("recorder")
                .with_grants(VideoGrants(room_join=True, room=TEST_ROOM))
                .to_jwt()
            )
            rec_room = rtc.Room()
            await rec_room.connect(LIVEKIT_URL, token)

            @rec_room.on("track_subscribed")
            def _on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
                try:
                    if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
                        return
                    os.makedirs("recordings", exist_ok=True)
                    path = os.path.join("recordings", f"{TEST_ROOM}-{participant.identity}-track-{publication.sid}.wav")

                    async def _record_track():
                        try:
                            stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
                        except Exception as e:
                            print(f"[recorder] AudioStream(track) init failed: {e}")
                            return
                        wrote = 0
                        try:
                            with wave.open(path, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                async for ev in stream:
                                    wf.writeframes(ev.frame.data)
                                    wrote += len(ev.frame.data)
                        finally:
                            with contextlib.suppress(Exception):
                                await stream.aclose()
                        if wrote > 0:
                            REMOTE_AUDIO_PATHS[participant.identity] = path

                    asyncio.create_task(_record_track())
                except Exception:
                    pass

            # Wait until STOP_EVENT
            global STOP_EVENT
            if STOP_EVENT is None:
                STOP_EVENT = asyncio.Event()
            await STOP_EVENT.wait()
        finally:
            with contextlib.suppress(Exception):
                if 'rec_room' in locals():
                    await rec_room.disconnect()

    recorder_task = asyncio.create_task(run_room_recorder())
    
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
            await asyncio.wait_for(asyncio.gather(agent_task, recorder_task), timeout=10)
        except asyncio.TimeoutError:
            for t in (agent_task, recorder_task):
                t.cancel()
            for t in (agent_task, recorder_task):
                with contextlib.suppress(asyncio.CancelledError):
                    await t

if __name__ == "__main__":
    asyncio.run(main())

