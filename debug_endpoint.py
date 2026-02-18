
# ------------------------------
# DEBUG ENDPOINT
# ------------------------------
@app.get("/api/debug-latest-call")
def debug_latest_call(db: Session = Depends(get_db)):
    try:
        call = db.query(CallLog).order_by(CallLog.id.desc()).first()
        if not call:
            return {"message": "No calls found"}
        
        return {
            "id": call.id,
            "status": call.status,
            "summary": call.summary,
            "transcript_exists_in_orm": hasattr(call, 'transcript'),
            "transcript_value": getattr(call, 'transcript', "ATTRIBUTE_MISSING"),
            "recording_url": call.recording_url,
            "duration": call.duration,
            "timestamp": str(call.timestamp)
        }
    except Exception as e:
        return {"error": str(e)}
