
import os
from openai import OpenAI

def test_kyutai_as_openai_whisper():
    """Test using OpenAI client with your Kyutai service"""
    
    # Point OpenAI client to your Kyutai service instead of OpenAI
    client = OpenAI(
        base_url="http://localhost:8080/v1",  # Your Kyutai service
        api_key="dummy_key"  # Your service doesn't validate this
    )
    
    try:
        print("🎤 Testing Kyutai STT via OpenAI Whisper API...")
        
        # This is exactly how you'd use OpenAI Whisper, but it hits your service!
        with open("audio.mp3", "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",  # Your service supports this
                file=audio_file,
                response_format="json"
            )
        
        print(f"✅ Transcription successful!")
        print(f"📝 Text: {transcription.text}")
        print(f"⏱️  Duration: {transcription.duration}s")
        print(f"🌍 Language: {transcription.language}")
        
        # Test different response formats
        print("\n🔄 Testing different formats...")
        
        # Text format
        with open("audio.mp3", "rb") as audio_file:
            text_result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        print(f"📄 Text format: {text_result}")
        
        # SRT format (for subtitles)
        with open("audio.mp3", "rb") as audio_file:
            srt_result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt"
            )
        print(f"🎬 SRT format:\n{srt_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_kyutai_as_openai_whisper()
