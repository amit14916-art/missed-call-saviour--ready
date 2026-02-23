import runpod
import base64
import io
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Loaded once at cold start
pipeline = KPipeline(lang_code="a")   # American English

def handler(job):
    inp   = job["input"]
    text  = inp.get("text", "")
    voice = inp.get("voice", "af_sarah")
    speed = inp.get("speed", 1.0)
    fmt   = inp.get("format", "wav")

    if not text:
        return {"error": "No text provided"}

    chunks = []
    for audio, _, _ in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio.numpy())

    audio_array = np.concatenate(chunks)
    buf = io.BytesIO()
    # Kokoro outputs at 24kHz
    sf.write(buf, audio_array, 24000, format=fmt.upper())
    buf.seek(0)

    return {
        "audio": base64.b64encode(buf.read()).decode("utf-8"),
        "sample_rate": 24000,
        "format": fmt
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
