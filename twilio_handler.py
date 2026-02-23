"""
Missed Call Saviour — Twilio Call Handler
Integrates Twilio Media Streams with RunPod AI Pipeline
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

# ── Twilio Webhook ────────────────────────────────────────────
async def handle_twilio_webhook(request: Request, db: Session):
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    caller_number = form_data.get("From", "Unknown")
    
    # Store initial call log
    from main import CallLog
    from datetime import datetime
    
    log = CallLog(
        phone_number=caller_number,
        vapi_call_id=call_sid, # Map CallSid to this generic ID field
        call_type="inbound-twilio",
        status="initiated",
        timestamp=datetime.utcnow()
    )
    db.add(log)
    db.commit()
    logger.info(f"📞 Twilio Inbound: {call_sid} from {caller_number}")

    domain = os.environ.get("DOMAIN", "").replace("https://", "").replace("http://", "")
    if not domain or "127.0.0.1" in domain or "localhost" in domain:
        domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    
    if not domain:
        # Fallback for local testing with ngrok (manual)
        logger.warning("No public domain found. Twilio Stream might fail.")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="wss://{domain}/ws/twilio" />
        </Connect>
    </Response>"""
    return Response(content=twiml, media_type="text/xml")

# ── WebSocket Audio Handler ───────────────────────────────────
async def handle_twilio_websocket(websocket: WebSocket, db: Session):
    await websocket.accept()
    logger.info("🔌 Twilio Stream connected")

    from main import AIConfig, CallLog
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
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                logger.info(f"🚀 Stream started SID: {stream_sid}")
                
                # Update status
                log = db.query(CallLog).filter_by(vapi_call_id=call_sid).first()
                if log:
                    log.status = "in-progress"
                    db.commit()

                # Send greeting
                greeting_audio = await pipeline.get_greeting_audio()
                if greeting_audio:
                    await _send_audio(websocket, stream_sid, greeting_audio)

            elif event == "media":
                payload = data["media"]["payload"]
                chunk = base64.b64decode(payload)
                audio_buffer.write(chunk)

                # Wait for enough audio (~1s)
                if audio_buffer.tell() > 8000:
                    audio_buffer.seek(0)
                    raw_mulaw = audio_buffer.read()
                    audio_buffer = io.BytesIO()

                    # Transcribe and Reply
                    pcm_raw = audioop.ulaw2lin(raw_mulaw, 2)
                    wav_mem = io.BytesIO()
                    with wave.open(wav_mem, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(8000)
                        wav_file.writeframes(pcm_raw)
                    
                    transcript, reply_audio = await pipeline.process_turn(wav_mem.getvalue())
                    
                    if reply_audio:
                        await _send_audio(websocket, stream_sid, reply_audio)

            elif event == "stop":
                logger.info(f"🛑 Stream stopped: {stream_sid}")
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
                
                # Update Dashboard
                try:
                    from main import sse_manager
                    await sse_manager.broadcast("update_dashboard")
                except:
                    pass

    except WebSocketDisconnect:
        logger.info("Twilio Stream disconnected")
    except Exception as e:
        logger.error(f"Twilio Handler Error: {e}")

async def _send_audio(websocket: WebSocket, stream_sid: str, audio_bytes: bytes):
    try:
        # Pipeline synthesis typically returns a WAV file bytes (from synthesize_speech)
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
            params = wav.getparams()
            raw_audio = wav.readframes(params.nframes)
            
            # 1. Resample to 8000Hz (Twilio requirement)
            if params.framerate != 8000:
                raw_audio, _ = audioop.ratecv(raw_audio, 2, 1, params.framerate, 8000, None)
            
            # 2. Convert to linear PCM to 8-bit mulaw
            mulaw_audio = audioop.lin2ulaw(raw_audio, 2)
            
            # 3. Send to Twilio
            message = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": base64.b64encode(mulaw_audio).decode("utf-8")
                }
            }
            await websocket.send_text(json.dumps(message))
    except Exception as e:
        logger.error(f"Twilio audio send error: {e}")
