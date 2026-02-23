# ═══════════════════════════════════════════════════════════
# MISSED CALL SAVIOUR — RunPod Serverless Deployment Guide
# ═══════════════════════════════════════════════════════════

## 1. Llama 3.1 8B Instruct (LLM)
- **Container Image**: `runpod/worker-vllm:stable-cuda12.1.0`
- **GPU**: RTX 4090
- **Env Vars**:
  - `MODEL_NAME`: `meta-llama/Llama-3.1-8B-Instruct`
  - `MAX_MODEL_LEN`: `4096`
  - `HF_TOKEN`: (Your HuggingFace Token)
  - `TENSOR_PARALLEL_SIZE`: `1`

## 2. Whisper Large v3 (STT)
- **Container Image**: `onerahmet/openai-whisper-asr-webservice:latest-gpu`
- **GPU**: RTX 3090
- **Env Vars**:
  - `ASR_MODEL`: `large-v3`
  - `ASR_ENGINE`: `faster_whisper`

## 3. Kokoro TTS (TTS)
- **Files**: Use `/ai_engine/tts_handler.py` and `/ai_engine/Dockerfile.tts`
- **Build**: `docker build -t your-user/kokoro-tts:latest -f Dockerfile.tts .`
- **Push**: `docker push your-user/kokoro-tts:latest`
- **RunPod**: Deploy as Serverless with the pushed image.

---
**Min Workers**: 0 (Scale to zero for cost saving)
**Idle Timeout**: 5 seconds
