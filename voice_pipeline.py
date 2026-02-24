"""
Missed Call Saviour — Voice AI Pipeline
Whisper (STT via RunPod, fallback: Gemini) + Gemini (LLM primary) + gTTS (TTS, free)
Cost target: ~₹0.95/min — Fully functional even without RunPod!
"""
import os
import io
import wave
import json
import base64
import asyncio
import httpx
import audioop
from gtts import gTTS
from pydub import AudioSegment
from google import genai

# ── RunPod Configuration ─────────────────────────────────────────
RP_KEY     = os.getenv("RUNPOD_API_KEY")
RP_WHISPER = os.getenv("RUNPOD_WHISPER_ID") or os.getenv("WHISPER_ENDPOINT_ID")
RP_VLLM    = os.getenv("RUNPOD_VLLM_ID")    or os.getenv("LLAMA_ENDPOINT_ID")
# Piper TTS removed — using free gTTS instead

print(f"[Pipeline] Whisper: {RP_WHISPER}, vLLM: {RP_VLLM}, TTS: gTTS (free)")

# ── Gemini Fallback ───────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
genai_client = None
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[Pipeline] Gemini fallback ready.")
    except Exception as e:
        print(f"[Pipeline] Gemini init failed: {e}")

# ── RunPod API Caller ─────────────────────────────────────────────
async def runpod_call(endpoint_id: str, input_data: dict, timeout: float = 90.0):
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

# ── Whisper STT (RunPod) with Gemini Fallback ────────────────────
async def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    # Try RunPod Whisper first (if configured)
    if RP_KEY and RP_WHISPER:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "audio_base64": audio_b64,
            "model": "base",
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
            if text:
                print(f"[Whisper] '{text}'")
                return text
            print(f"[Whisper] Empty/Error — falling back to Gemini STT")
        except Exception as e:
            print(f"[Whisper] Exception: {e} — falling back to Gemini STT")

    # Gemini STT Fallback (FREE)
    return await _gemini_stt_fallback(audio_bytes)

async def _gemini_stt_fallback(audio_bytes: bytes) -> str:
    """Use Gemini to transcribe audio when Whisper is unavailable."""
    try:
        if not genai_client:
            print("[Gemini STT] No client available")
            return ""
        # Convert audio bytes to base64 for Gemini
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        # Use Gemini with inline audio data
        import google.genai.types as genai_types
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"
                ),
                "Transcribe this audio exactly. Return ONLY the spoken words, nothing else. If silent or unclear, return empty string."
            ]
        )
        text = response.text.strip() if response.text else ""
        # Filter out common Gemini non-transcription responses
        if any(phrase in text.lower() for phrase in ["cannot", "no audio", "unable", "i don't", "silent"]):
            text = ""
        print(f"[Gemini STT] '{text}'")
        return text
    except Exception as e:
        print(f"[Gemini STT] Error: {e}")
        return ""

# ── LLM: Gemini Primary + vLLM (RunPod) as Optional ─────────────
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
- Speak in Hinglish (a mix of Hindi and English) as it's more natural for callers.
- Be extremely concise — max 1-2 SHORT sentences per reply.
- Capture the caller's name and why they are calling.
- Always offer a callback from {owner_name}.
- Keep the tone helpful, warm, and professional.
"""
    # Use Gemini as PRIMARY (free, fast, always available)
    gemini_reply = await _gemini_fallback(caller_message, base_system, conversation_history)
    if gemini_reply:
        return gemini_reply

    # Fallback to RunPod vLLM only if Gemini fails
    if RP_KEY and RP_VLLM:
        messages = [{"role": "system", "content": base_system}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": caller_message})
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": messages,
            "max_tokens": 120,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "<|end_of_text|>"]
        }
        try:
            resp = await runpod_call(RP_VLLM, payload)
            output = resp.get("output", {})
            if isinstance(output, dict) and "choices" in output:
                reply = output["choices"][0]["message"]["content"].strip()
            elif isinstance(output, dict):
                reply = output.get("response", output.get("text", "")).strip()
            elif isinstance(output, str):
                reply = output.strip()
            else:
                reply = ""
            if reply:
                print(f"[vLLM] '{reply[:80]}'")
                return reply
        except Exception as e:
            print(f"[vLLM] Exception: {e}")

    return "Thank you for calling. The owner will call you back shortly."

# ── Gemini LLM (Primary) ─────────────────────────────────────────
async def _gemini_fallback(prompt: str, system: str, history: list = None) -> str:
    try:
        if not genai_client:
            return ""
        # Build conversation context
        ctx = ""
        if history:
            for msg in history[-6:]:  # last 3 exchanges
                role = "Caller" if msg["role"] == "user" else "AI"
                ctx += f"{role}: {msg['content']}\n"
        full_prompt = f"{system}\n\nConversation so far:\n{ctx}\nCaller: {prompt}\nAI:"
        r = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        reply = r.text.strip() if r.text else ""
        print(f"[Gemini] '{reply[:80]}'")
        return reply
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return ""

# ── gTTS (Free Text-to-Speech) ────────────────────────────────────
def _gtts_to_wav(text: str, lang: str = "en") -> bytes:
    """Synchronous gTTS → MP3 → WAV conversion (runs in thread pool)."""
    try:
        mp3_buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        # Convert MP3 → WAV via pydub (ffmpeg already in Dockerfile)
        audio = AudioSegment.from_mp3(mp3_buf)
        # Normalize to 8kHz mono 16-bit for telephony compatibility
        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        wav_buf = io.BytesIO()
        audio.export(wav_buf, format="wav")
        wav_buf.seek(0)
        return wav_buf.read()
    except Exception as e:
        print(f"[gTTS] Error: {e}")
        return b""

async def synthesize_speech(text: str, voice: str = "en") -> bytes:
    """Async wrapper — gTTS runs in a thread so it doesn't block the event loop."""
    if not text:
        return b""
    # Detect Hindi/Hinglish — use 'hi' lang code
    lang = "hi" if any(ord(c) > 127 for c in text) else "en"
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, _gtts_to_wav, text, lang)
    print(f"[gTTS] Done: '{text[:50]}'")
    return audio

# ── Pipeline Class ────────────────────────────────────────────────
class MissedCallPipeline:
    def __init__(
        self,
        business_name: str,
        owner_name: str,
        assistant_role: str = "Senior AI Representative",
        system_prompt: str = "",
        greeting: str = "Hello! How can I help you today?",
        voice: str = "en"
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
        return await synthesize_speech(self.greeting)

    async def process_turn(self, audio_bytes: bytes):
        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            fallback_audio = await synthesize_speech(
                "I didn't catch that, could you say that again?"
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
        if "my name is" in tl or "mera naam" in tl:
            try:
                self.caller_name = tl.split("my name is")[-1].strip().split()[0].title()
            except Exception:
                pass

        reply_audio = await synthesize_speech(reply)
        return transcript, reply_audio

    def get_summary(self) -> dict:
        return {
            "caller_name": self.caller_name,
            "conversation_history": self.conversation_history,
            "turn_count": len(self.conversation_history) // 2
        }
