import os
import asyncio
import contextlib
import wave

from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants


load_dotenv()

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "http://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
ROOM_NAME = os.environ.get("ROOM_NAME", os.environ.get("TEST_ROOM", "test-room-001"))
SAMPLE_RATE = int(os.environ.get("RECORDER_SAMPLE_RATE", "8000"))


async def run_room_recorder():
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("recorder")
        .with_grants(VideoGrants(room_join=True, room=ROOM_NAME))
        .to_jwt()
    )

    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    print(f"✓ Recorder joined room: {ROOM_NAME}")

    os.makedirs("recordings", exist_ok=True)

    @room.on("track_subscribed")
    def _on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        try:
            if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
                return
            path = os.path.join("recordings", f"{ROOM_NAME}-{participant.identity}-track-{publication.sid}.wav")

            async def _record_track():
                try:
                    stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
                except Exception as e:
                    print(f"[recorder] AudioStream(track) init failed: {e}")
                    return
                wrote = 0
                try:
                    with wave.open(path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        async for ev in stream:
                            wf.writeframes(ev.frame.data)
                            wrote += len(ev.frame.data)
                finally:
                    with contextlib.suppress(Exception):
                        await stream.aclose()
                if wrote > 0:
                    print(f"✓ Recorded {participant.identity} to {path}")

            asyncio.create_task(_record_track())
        except Exception:
            pass

    try:
        # Run until interrupted
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            await room.disconnect()


if __name__ == "__main__":
    asyncio.run(run_room_recorder())


