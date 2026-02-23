"""
Missed Call Saviour — Exotel Voicebot Handler
Integrates Exotel Media Streams with RunPod AI Pipeline
"""

import os
import io
import json
import base64
import asyncio
import logging
import audioop
import wave
from fastapi import WebSocket, WebSocketDisconnect, Request, Response
from sqlalchemy.orm import Session
from voice_pipeline import MissedCallPipeline

logger = logging.getLogger(__name__)

# -- Exotel Webhook (Optional/Passthru) --
async def handle_exotel_webhook(request: Request, db: Session):
    """
    Handles Exotel Passthru requests if needed for call tracking.
    """
    try:
        data = await request.form()
        call_sid = data.get("CallSid")
        caller = data.get("From")
        logger.info(f"📞 Exotel Webhook: {call_sid} from {caller}")
        return Response(content="OK", media_type="text/plain")
    except Exception as e:
        logger.error(f"Exotel Webhook Error: {e}")
        return Response(content="Error", status_code=500)

# -- WebSocket Audio Handler --
async def handle_exotel_websocket(websocket: WebSocket, db: Session):
    await websocket.accept()
    logger.info("🔌 Exotel Voicebot Stream connected")

    from main import AIConfig, CallLog, sse_manager
    from datetime import datetime

    stream_sid = None
    call_sid = None
    pipeline = None

    try:
        config = db.query(AIConfig).first()
        if not config:
            logger.error("No AIConfig found in DB")
            await websocket.close()
            return

        pipeline = MissedCallPipeline(
            business_name = config.business_name or "the business",
            owner_name    = getattr(config, "owner_name", "the owner"),
            assistant_role = getattr(config, "assistant_role", "Senior AI Representative"),
            system_prompt = config.system_prompt or "",
            greeting      = config.greeting or f"Hello!",
            voice         = config.persona if config.persona else "af_sarah"
        )

        audio_buffer = io.BytesIO()

        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
            except:
                continue
                
            event = data.get("event")

            if event == "connected":
                stream_sid = data.get("stream_sid")
                call_sid = data.get("call_sid")
                logger.info(f"🚀 Exotel Stream started SID: {stream_sid}, Call: {call_sid}")
                
                # Create/Update call log
                log = db.query(CallLog).filter_by(vapi_call_id=call_sid).first()
                if not log:
                    log = CallLog(
                        phone_number="Exotel-Inbound",
                        vapi_call_id=call_sid,
                        call_type="inbound-exotel",
                        status="in-progress",
                        timestamp=datetime.utcnow()
                    )
                    db.add(log)
                    db.commit()
                else:
                    log.status = "in-progress"
                    db.commit()

                # Send greeting
                greeting_audio = await pipeline.get_greeting_audio()
                if greeting_audio:
                    # Exotel expects PCM 16-bit 8kHz
                    await _send_audio(websocket, greeting_audio)

            elif event == "media":
                payload = data["media"]["payload"]
                chunk = base64.b64decode(payload)
                audio_buffer.write(chunk)

                # Process every ~0.8s (12800 bytes for 16-bit 8kHz = 0.8s)
                if audio_buffer.tell() > 12800:
                    audio_buffer.seek(0)
                    raw_pcm = audio_buffer.read()
                    audio_buffer = io.BytesIO()

                    # Convert to WAV for Whisper
                    wav_mem = io.BytesIO()
                    with wave.open(wav_mem, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(8000)
                        wav_file.writeframes(raw_pcm)
                    
                    try:
                        transcript, reply_audio = await pipeline.process_turn(wav_mem.getvalue())
                        
                        if transcript:
                            logger.info(f"🎤 {call_sid} Transcript: {transcript}")

                        if reply_audio:
                            await _send_audio(websocket, reply_audio)
                    except Exception as turn_e:
                        logger.error(f"Turn processing error: {turn_e}")

            elif event == "stop" or event == "clear":
                logger.info(f"🛑 Exotel Stream stopped: {stream_sid}")
                break

        # Cleanup & Summary
        if pipeline:
            summary = pipeline.get_summary()
            log = db.query(CallLog).filter_by(vapi_call_id=call_sid).first()
            if log:
                from notification import send_owner_notification
                log.status = "completed"
                log.transcript = json.dumps(summary.get("conversation_history", []))
                log.caller_name = summary.get("caller_name")
                db.commit()
                
                try:
                    await send_owner_notification(
                        caller_number=log.phone_number,
                        transcript=log.transcript,
                        caller_name=log.caller_name
                    )
                except Exception as e:
                    logger.error(f"Notification error: {e}")
                
                await sse_manager.broadcast("update_dashboard")

    except WebSocketDisconnect:
        logger.info("Exotel Stream disconnected")
    except Exception as e:
        logger.error(f"Exotel Handler Error: {e}")

async def _send_audio(websocket: WebSocket, audio_bytes: bytes):
    try:
        # Pipeline synthesis returns WAV bytes
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
            params = wav.getparams()
            raw_audio = wav.readframes(params.nframes)
            
            # Resample to 8000Hz (Exotel default)
            if params.framerate != 8000:
                raw_audio, _ = audioop.ratecv(raw_audio, 2, 1, params.framerate, 8000, None)
            
            # Send to Exotel (PCM 16-bit)
            message = {
                "event": "media",
                "media": {
                    "payload": base64.b64encode(raw_audio).decode("utf-8")
                }
            }
            await websocket.send_text(json.dumps(message))
    except Exception as e:
        logger.error(f"Exotel audio send error: {e}")
