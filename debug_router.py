from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from main import get_db, CallLog

router = APIRouter()

@router.get("/api/debug/latest-call")
def get_latest_call(db: Session = Depends(get_db)):
    latest_call = db.query(CallLog).order_by(CallLog.id.desc()).first()
    if not latest_call:
        return {"error": "No calls found"}
    
    return {
        "id": latest_call.id,
        "phone": latest_call.phone_number,
        "status": latest_call.status,
        "summary": latest_call.summary,
        "transcript": latest_call.transcript, # This checks if the column exists and has data
        "recording_url": latest_call.recording_url,
        "duration": latest_call.duration,
        "raw_timestamp": latest_call.timestamp
    }
