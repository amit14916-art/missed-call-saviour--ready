
import os
import httpx
import asyncio
import base64
from dotenv import load_dotenv

load_dotenv()

RP_KEY = os.getenv("RUNPOD_API_KEY")
RP_WHISPER = os.getenv("RUNPOD_WHISPER_ID")
RP_VLLM = os.getenv("RUNPOD_VLLM_ID")
RP_TTS = os.getenv("RUNPOD_TTS_ID")

async def test_endpoint(name, endpoint_id, payload):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {RP_KEY}",
        "Content-Type": "application/json"
    }
    print(f"Testing {name} ({endpoint_id})...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, json={"input": payload}, headers=headers)
            print(f"{name} Response: {resp.status_code}")
            print(f"{name} Result: {resp.text[:200]}...")
            return resp.status_code == 200
        except Exception as e:
            print(f"{name} Error: {e}")
            return False

async def main():
    print("--- RunPod Diagnostic ---")
    print(f"API Key: {RP_KEY[:10]}...")
    
    # 1. Test Whisper
    dummy_audio = base64.b64encode(b"fake audio data").decode()
    await test_endpoint("Whisper", RP_WHISPER, {"audio_base64": dummy_audio})
    
    # 2. Test Llama
    await test_endpoint("Llama", RP_VLLM, {"prompt": "Hello", "max_new_tokens": 10})
    
    # 3. Test TTS
    await test_endpoint("TTS", RP_TTS, {"text": "Hello", "voice": "af_sarah"})

if __name__ == "__main__":
    asyncio.run(main())
