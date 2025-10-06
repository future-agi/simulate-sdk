import asyncio
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

load_dotenv()

async def test_connection():
    """
    Simple test to verify we can connect to the LiveKit server.
    """
    url = os.environ.get("LIVEKIT_URL", "http://localhost:7880")
    
    # Try using http:// - the client library may handle the WebSocket upgrade
    api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    
    # Generate a token
    token = (
        AccessToken(api_key, api_secret)
        .with_identity("test-participant")
        .with_grants(VideoGrants(room_join=True, room="test-room"))
        .to_jwt()
    )
    
    print(f"Connecting to {url}...")
    print(f"Using token for room 'test-room'...")
    
    try:
        room = rtc.Room()
        
        @room.on("connected")
        def on_connected():
            print("✓ Successfully connected to LiveKit!")
            print(f"✓ Joined room: {room.name}")
        
        await room.connect(url, token)
        
        # Wait a bit to ensure connection is stable
        await asyncio.sleep(5)
        
        await room.disconnect()
        print("✓ Disconnected successfully")
        print("\nConnection test PASSED ✓")
        
    except Exception as e:
        print(f"✗ Connection test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())

