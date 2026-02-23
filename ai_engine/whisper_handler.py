import runpod
import torch
from faster_whisper import WhisperModel
import base64
import os
import uuid

# Load model into memory on startup
# Default to 'medium' for better Hindi/Hinglish accuracy if GPU allows
model_size = os.getenv("WHISPER_MODEL", "base")
# GPU is required for RunPod, using float16 for speed
try:
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print(f"Whisper Model '{model_size}' loaded successfully on GPU.")
except Exception as e:
    print(f"Failed to load model on GPU, falling back to CPU: {e}")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

def handler(event):
    """
    Expects input event:
    {
        "input": {
            "audio_base64": "...",
            "language": "hi", (optional)
            "initial_prompt": "..." (optional for context)
        }
    }
    """
    input_data = event.get("input", {})
    audio_b64 = input_data.get("audio_base64")
    language = input_data.get("language")
    initial_prompt = input_data.get("initial_prompt")
    
    if not audio_b64:
        return {"error": "Missing audio_base64"}
    
    # Save temp audio file with unique name to avoid collisions
    temp_filename = f"audio_{uuid.uuid4()}.wav"
    try:
        with open(temp_filename, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        
        # Transcribe
        # beam_size=5 is good for accuracy
        segments, info = model.transcribe(
            temp_filename, 
            beam_size=5, 
            language=language, 
            initial_prompt=initial_prompt
        )
        
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
        
        return {
            "transcript": transcript.strip(),
            "language": info.language,
            "language_probability": info.language_probability
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

runpod.serverless.start({"handler": handler})
