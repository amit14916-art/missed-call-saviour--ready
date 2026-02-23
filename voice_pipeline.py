"""
Missed Call Saviour — Voice AI Pipeline
Whisper (STT) + vLLM OpenAI (LLM) + Piper (TTS) via RunPod Serverless
"""
import os
import io
import wave
import json
import base64
import httpx
import audioop
from google import genai

# RunPod Configuration — matches .env variable names exactly
RP_KEY     = os.getenv("RUNPOD_API_KEY")
RP_WHISPER = os.getenv("RUNPOD_WHISPER_ID")   or os.getenv("WHISPER_ENDPOINT_ID")
RP_VLLM    = os.getenv("RUNPOD_VLLM_ID")      or os.getenv("LLAMA_ENDPOINT_ID")
RP_TTS     = os.getenv("RUNPOD_TTS_ID")       or os.getenv("KOKORO_ENDPOINT_ID") or os.getenv("PIPER_ENDPOINT_ID")

print(f"[Pipeline] Whisper: {RP_WHISPER}, vLLM: {RP_VLLM}, TTS: {RP_TTS}")

# Gemini Fallback Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
genai_client = None
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[Pipeline] Gemini fallback configured.")
    except Exception as e:
        print(f"[Pipeline] Gemini init failed: {e}")

# ── Core RunPod Caller ────────────────────────────────────────────
async def runpod_call(endpoint_id: str, input_data: dict, timeout: float = 90.0):
    """Call a RunPod /runsync endpoint and return parsed JSON."""
    if not RP_KEY or not endpoint_id:
        return {"error": "RunPod not configured"}

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {RP_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json={"input": input_data}, headers=headers)
            if resp.status_code != 200:
                print(f"[RunPod] Error ({endpoint_id}): {resp.status_code} — {resp.text[:300]}")
                return {"error": f"HTTP {resp.status_code}"}
            return resp.json()
        except Exception as e:
            print(f"[RunPod] Exception ({endpoint_id}): {e}")
            return {"error": str(e)}

# ── Whisper STT ───────────────────────────────────────────────────
async def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """Convert audio bytes to text via Whisper on RunPod."""
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
            text = output.strip()
        else:
            text = (output.get("transcript", "") or output.get("text", "")).strip()

        if not text and "error" in resp:
            print(f"[Whisper] Error: {resp['error']}")

        print(f"[Whisper] '{text}'")
        return text
    except Exception as e:
        print(f"[Whisper] Exception: {e}")
        return ""

# ── vLLM (OpenAI-compatible) LLM ─────────────────────────────────
async def generate_reply(
    caller_message: str,
    business_name: str,
    owner_name: str,
    assistant_role: str,
    system_prompt: str,
    conversation_history: list
) -> str:
    """Generate AI reply using vLLM OpenAI-compat endpoint on RunPod."""
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

    # vLLM OpenAI-compatible format (chat/completions style)
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["<|eot_id|>", "<|end_of_text|>"]
    }
    try:
        resp = await runpod_call(RP_VLLM, payload)
        output = resp.get("output", {})

        # OpenAI compat response format: output.choices[0].message.content
        if isinstance(output, dict) and "choices" in output:
            reply = output["choices"][0]["message"]["content"].strip()
        elif isinstance(output, dict):
            reply = output.get("response", output.get("text", "")).strip()
        elif isinstance(output, str):
            reply = output.strip()
        else:
            reply = ""

        print(f"[vLLM] '{reply[:80]}'")
        return reply if reply else await _gemini_fallback(caller_message, base_system)
    except Exception as e:
        print(f"[vLLM] Exception: {e}")
        return await _gemini_fallback(caller_message, base_system)

# ── Gemini Fallback ───────────────────────────────────────────────
async def _gemini_fallback(prompt: str, system: str) -> str:
    try:
        if genai_client:
            r = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{system}\n\nCaller: {prompt}"
            )
            print("[Gemini] Fallback used")
            return r.text
        return "Thank you for calling. The owner will call you back shortly."
    except Exception as e:
        print(f"[Gemini] Fallback error: {e}")
        return "Thank you for calling. The owner will call you back shortly."

# ── Piper TTS ─────────────────────────────────────────────────────
async def synthesize_speech(text: str, voice: str = "en_US-lessac-medium") -> bytes:
    """
    Convert text to audio using Piper TTS on RunPod.
    Docker image: runpod/ai-api-piper:latest
    Payload: { "text": "...", "voice": "en_US-lessac-medium" }
    Response: output.audio (base64 WAV)
    """
    # Map friendly names to Piper voice model names
    voice_map = {
        "af_sarah": "en_US-lessac-medium",
        "friendly": "en_US-lessac-medium",
        "professional": "en_US-ryan-medium",
        "male": "en_US-ryan-medium",
        "female": "en_US-lessac-medium",
    }
    piper_voice = voice_map.get(voice, voice if "_" in voice else "en_US-lessac-medium")

    payload = {
        "text": text,
        "voice": piper_voice
    }
    try:
        resp = await runpod_call(RP_TTS, payload)
        output = resp.get("output", {})

        # Piper returns base64 audio in output.audio or output.audio_base64 or output directly
        if isinstance(output, str):
            audio_b64 = output
        elif isinstance(output, dict):
            audio_b64 = output.get("audio", "") or output.get("audio_base64", "")
        else:
            audio_b64 = ""

        if audio_b64:
            audio = base64.b64decode(audio_b64)
            print(f"[Piper] TTS done: '{text[:50]}...'")
            return audio

        print(f"[Piper] No audio in response: {str(resp)[:200]}")
        return b""
    except Exception as e:
        print(f"[Piper] Exception: {e}")
        return b""

# ── Pipeline Class ────────────────────────────────────────────────
class MissedCallPipeline:
    def __init__(
        self,
        business_name: str,
        owner_name: str,
        assistant_role: str = "Senior AI Representative",
        system_prompt: str = "",
        greeting: str = "Hello!",
        voice: str = "en_US-lessac-medium"
    ):
        self.business_name = business_name
        self.owner_name = owner_name
        self.assistant_role = assistant_role
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.voice = voice
        self.conversation_history = []
        self.caller_name = None

    async def get_greeting_audio(self) -> bytes:
        print(f"[Pipeline] Starting for: {self.business_name}")
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

        # Extract caller name if mentioned
        tl = transcript.lower()
        if "my name is" in tl:
            try:
                self.caller_name = tl.split("my name is")[1].strip().split()[0].title()
            except Exception:
                pass

        reply_audio = await synthesize_speech(reply, self.voice)
        return transcript, reply_audio

    def get_summary(self) -> dict:
        return {
            "caller_name": self.caller_name,
            "conversation_history": self.conversation_history,
            "turn_count": len(self.conversation_history) // 2
        }
