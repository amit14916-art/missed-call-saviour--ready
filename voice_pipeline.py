import os
import io
import wave
import json
import base64
import httpx
import audioop
from google import genai

# RunPod Configuration
RP_KEY = os.getenv("RUNPOD_API_KEY")
RP_WHISPER = os.getenv("WHISPER_ENDPOINT_ID") or os.getenv("RUNPOD_WHISPER_ID")
RP_VLLM = os.getenv("LLAMA_ENDPOINT_ID") or os.getenv("RUNPOD_VLLM_ID")
RP_TTS = os.getenv("KOKORO_ENDPOINT_ID") or os.getenv("RUNPOD_TTS_ID")

# Gemini Fallback Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
genai_client = None
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except:
        pass

async def runpod_call(endpoint_id, input_data):
    if not RP_KEY or not endpoint_id:
        return {"error": "RunPod not configured"}
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {RP_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(url, json={"input": input_data}, headers=headers)
            if resp.status_code != 200:
                print(f"RunPod Error ({endpoint_id}): {resp.status_code} - {resp.text[:200]}")
                return {"error": f"HTTP {resp.status_code}"}
            return resp.json()
        except Exception as e:
            print(f"RunPod Client Exception ({endpoint_id}): {e}")
            return {"error": str(e)}

async def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "audio_base64": audio_b64,
        "model": "large-v3",
        "language": language,
        "task": "transcribe",
        "temperature": 0.0
    }
    try:
        resp = await runpod_call(RP_WHISPER, payload)
        output = resp.get("output", {})
        if isinstance(output, str):
            # Sometimes RunPod returns the output directly if it's a single string
            text = output.strip()
        else:
            text = output.get("transcript", "").strip() or output.get("text", "").strip()
        
        if not text and "error" in resp:
            print(f"Whisper RunPod Error: {resp['error']}")
            
        print(f"🎤 Whisper: '{text}'")
        return text
    except Exception as e:
        print(f"Whisper error: {e}")
        return ""

async def generate_reply(
    caller_message: str,
    business_name: str,
    owner_name: str,
    assistant_role: str,
    system_prompt: str,
    conversation_history: list
) -> str:
    base_system = f"""You are {assistant_role} for {business_name}.
Owner: {owner_name}
{system_prompt}

Rules:
- Be warm and concise — max 2 sentences per reply
- Take caller name and reason for calling
- Always offer callback from {owner_name}
- Never make up business info you don't know
- If unsure, say "I'll pass that to {owner_name}"
"""
    messages = [{"role": "system", "content": base_system}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": caller_message})

    payload = {
        "messages": messages,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["<|eot_id|>", "<|end_of_text|>"]
    }
    try:
        resp = await runpod_call(RP_VLLM, payload)
        output = resp.get("output", {})
        if "choices" in output:
            reply = output["choices"][0]["message"]["content"].strip()
        else:
            reply = output.get("response", "").strip()
            
        print(f"🤖 Llama: '{reply}'")
        return reply if reply else await _gemini_fallback(caller_message, base_system)
    except Exception as e:
        print(f"Llama error: {e}")
        return await _gemini_fallback(caller_message, base_system)

async def _gemini_fallback(prompt: str, system: str) -> str:
    try:
        if genai_client:
            r = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{system}\n\nCaller: {prompt}"
            )
            print("✅ Gemini fallback used")
            return r.text
        return "Thank you for calling. The owner will call you back shortly."
    except Exception as e:
        print(f"Gemini fallback error: {e}")
        return "Thank you for calling. The owner will call you back shortly."

async def synthesize_speech(text: str, voice: str = "af_sarah") -> bytes:
    payload = {
        "text": text,
        "voice": voice,
        "speed": 1.0,
        "format": "wav"
    }
    try:
        resp = await runpod_call(RP_TTS, payload)
        audio_b64 = resp.get("output", {}).get("audio", "") or resp.get("output", {}).get("audio_base64", "")
        if audio_b64:
            audio = base64.b64decode(audio_b64)
            print(f"🔊 Kokoro TTS done for: '{text[:40]}...'")
            return audio
        return b""
    except Exception as e:
        print(f"Kokoro error: {e}")
        return b""

class MissedCallPipeline:
    def __init__(self, business_name: str, owner_name: str, assistant_role: str = "Senior AI Representative",
                 system_prompt: str = "", greeting: str = "Hello!", voice: str = "af_sarah"):
        self.business_name = business_name
        self.owner_name = owner_name
        self.assistant_role = assistant_role
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.voice = voice
        self.conversation_history = []
        self.caller_name = None

    async def get_greeting_audio(self) -> bytes:
        print(f"🚀 Pipeline started for: {self.business_name}")
        return await synthesize_speech(self.greeting, self.voice)

    async def process_turn(self, audio_bytes: bytes):
        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            fallback_audio = await synthesize_speech(
                "I didn't catch that, could you say that again?", self.voice
            )
            return "", fallback_audio

        reply = await generate_reply(
            caller_message=transcript,
            business_name=self.business_name,
            owner_name=self.owner_name,
            assistant_role=self.assistant_role,
            system_prompt=self.system_prompt,
            conversation_history=self.conversation_history
        )

        self.conversation_history.append({"role": "user", "content": transcript})
        self.conversation_history.append({"role": "assistant", "content": reply})

        tl = transcript.lower()
        if "my name is" in tl:
            try:
                self.caller_name = tl.split("my name is")[1].strip().split()[0].title()
            except:
                pass

        reply_audio = await synthesize_speech(reply, self.voice)
        return transcript, reply_audio

    def get_summary(self) -> dict:
        return {
            "caller_name": self.caller_name,
            "conversation_history": self.conversation_history,
            "turn_count": len(self.conversation_history) // 2
        }
