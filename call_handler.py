"""
Missed Call Saviour — Telnyx Call Handler
Replaces Vapi with open source pipeline
"""

import os
import io
import json
import base64
import asyncio
import logging
import httpx
import wave
import audioop
from fastapi import WebSocket, WebSocketDisconnect, Request
from sqlalchemy.orm import Session
from voice_pipeline import MissedCallPipeline

logger = logging.getLogger(__name__)

def get_telnyx_headers():
    api_key = os.environ.get("TELNYX_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

# ── Telnyx Webhook ────────────────────────────────────────────
async def handle_telnyx_webhook(request: Request, db: Session):
    try:
        payload    = await request.json()
        event_type = payload.get("data", {}).get("event_type", "")
        call_data  = payload.get("data", {}).get("payload", {})

        logger.info(f"📞 Telnyx event: {event_type}")

        if event_type == "call.initiated":
            caller_number    = call_data.get("from")
            call_control_id  = call_data.get("call_control_id")
            await _answer_call(call_control_id)
            await _log_call(caller_number, call_control_id, db)

        elif event_type == "call.answered":
            call_control_id = call_data.get("call_control_id")
            await _start_stream(call_control_id)

        elif event_type == "call.hangup":
            call_control_id = call_data.get("call_control_id")
            caller_number   = call_data.get("from")
            await _on_hangup(call_control_id, caller_number, db)

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Telnyx Webhook Error: {e}")
        return {"status": "error", "detail": str(e)}

async def _answer_call(call_control_id: str):
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/answer"
    async with httpx.AsyncClient() as c:
        resp = await c.post(url, headers=get_telnyx_headers(), json={})
        print(f"Telnyx Answer Status: {resp.status_code} - {resp.text}")
    logger.info(f"✅ Call answered attempt: {call_control_id}")

async def _start_stream(call_control_id: str):
    domain = os.environ.get("DOMAIN", "").replace("https://", "").replace("http://", "")
    if not domain:
        domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    
    ws_url = f"wss://{domain}/ws/call/{call_control_id}"
    url    = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/streaming_start"
    print(f"🔗 Starting Telnyx Stream to: {ws_url}")
    async with httpx.AsyncClient() as c:
        resp = await c.post(url, headers=get_telnyx_headers(), json={
            "stream_url": ws_url,
            "stream_track": "both_tracks"
        })
        print(f"Telnyx Stream Status: {resp.status_code} - {resp.text}")
    logger.info(f"🔌 Stream started → {ws_url}")

async def _log_call(caller_number: str, call_control_id: str, db: Session):
    try:
        from main import CallLog
        from datetime import datetime
        log = CallLog(
            phone_number=caller_number,
            telnyx_call_control_id=call_control_id,
            call_type="inbound-telnyx",
            status="initiated",
            timestamp=datetime.utcnow()
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error(f"DB log error: {e}")

async def _on_hangup(call_control_id: str, caller_number: str, db: Session):
    try:
        from main import CallLog
        from notification import send_owner_notification
        log = db.query(CallLog).filter_by(
            telnyx_call_control_id=call_control_id
        ).first()
        if log:
            log.status = "completed"
            db.commit()
            # Broadcast update via SSE (assuming global sse_manager access or notification)
            # In a real app we'd pass a broadcast hook
            try:
                from main import sse_manager
                await sse_manager.broadcast("update_dashboard")
            except: pass

            await send_owner_notification(
                caller_number=caller_number,
                transcript=log.transcript or "",
                caller_name=log.caller_name
            )
    except Exception as e:
        logger.error(f"Hangup handler error: {e}")
    logger.info(f"📵 Call ended: {call_control_id}")

# ── WebSocket Audio Handler ───────────────────────────────────
async def handle_call_websocket(websocket: WebSocket, call_control_id: str, db: Session):
    await websocket.accept()
    logger.info(f"🔌 WS connected: {call_control_id}")

    from main import AIConfig, CallLog

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
            greeting      = config.greeting or f"Thank you for calling {config.business_name}. How can I help?",
            voice         = config.persona if config.persona else "af_sarah"
        )

        # Send greeting
        greeting_audio = await pipeline.get_greeting_audio()
        if greeting_audio:
            await _play_audio(call_control_id, greeting_audio)

        audio_buffer = io.BytesIO()

        while True:
            try:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")

                if event == "media":
                    payload = msg['media']['payload']
                    chunk = base64.b64decode(payload)
                    audio_buffer.write(chunk)
                    
                    if audio_buffer.tell() > 16000:
                        audio_buffer.seek(0)
                        raw_audio = audio_buffer.read()
                        audio_buffer = io.BytesIO()
                        
                        try:
                            pcm_raw = audioop.ulaw2lin(raw_audio, 2)
                            wav_mem = io.BytesIO()
                            with wave.open(wav_mem, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(8000)
                                wav_file.writeframes(pcm_raw)
                            
                            transcript, reply_audio = await pipeline.process_turn(wav_mem.getvalue())
                            
                            if transcript:
                                _append_transcript(call_control_id, transcript, db)

                            if reply_audio:
                                await _play_audio(call_control_id, reply_audio)
                        except Exception as pipe_e:
                            logger.error(f"Pipeline processing error: {pipe_e}")

                elif event == "stop":
                    logger.info("🛑 Stream stopped")
                    break

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WS Loop error: {e}")
                break

        # Save final summary
        summary = pipeline.get_summary()
        log = db.query(CallLog).filter_by(telnyx_call_control_id=call_control_id).first()
        if log:
            log.caller_name = summary.get("caller_name")
            log.transcript  = json.dumps(summary.get("conversation_history", []))
            log.summary = f"Telnyx Call. Turns: {summary.get('turn_count')}"
            db.commit()

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {call_control_id}")
    except Exception as e:
        logger.error(f"WS error: {e}")
    finally:
        pass # Session is managed by FastAPI dependency injection in main.py

async def _play_audio(call_control_id: str, audio_bytes: bytes):
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/playback_start"
    async with httpx.AsyncClient() as c:
        await c.post(url, headers=get_telnyx_headers(), json={
            "audio_url": f"data:audio/wav;base64,{audio_b64}",
            "loop": "once"
        })

def _append_transcript(call_control_id: str, text: str, db: Session):
    # This is a helper to update transcript in real-time if needed
    pass
