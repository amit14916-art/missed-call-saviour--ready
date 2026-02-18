from fastapi import FastAPI, Request, Form, BackgroundTasks, Depends, HTTPException, Body, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import uvicorn
import os
import stripe
import httpx
import razorpay
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
DOMAIN = os.getenv("DOMAIN", "http://127.0.0.1:8000")

# Razorpay
# Security
SECRET_KEY = "super-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Database Setup ---

try:
    from config_secrets import DATABASE_URL, VAPI_PRIVATE_KEY, VAPI_ASSISTANT_ID, VAPI_PHONE_NUMBER_ID, RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET
    print("Using hardcoded secrets from config_secrets.py")
except ImportError:
    print("Using environment variables (config_secrets.py not found)")
    DATABASE_URL = os.getenv("DATABASE_URL")
    VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY")
    VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID")
    VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID")
    RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
    RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

# Initialize Razorpay Client
try:
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    else:
        razorpay_client = None
        print("WARNING: Razorpay Keys Missing")
except NameError:
    razorpay_client = None
    print("WARNING: Razorpay config failed")

# Ensure DATABASE_URL is valid
if not DATABASE_URL:
    print("WARNING: DATABASE_URL not found!")
    # Allow SQLite fallback for temporary use
    DATABASE_URL = "sqlite:///./missed_calls.db"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Initialize App ---
app = FastAPI(title="Missed Call Saviour Backend")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    plan = Column(String, default="Free")
    is_active = Column(Boolean, default=False)
    registration_date = Column(DateTime, default=datetime.utcnow) 
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    stripe_customer_id = Column(String, nullable=True)
    
class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    amount = Column(Float)
    plan_name = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    stripe_session_id = Column(String, nullable=True)

class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True, nullable=True) 
    phone_number = Column(String)
    call_type = Column(String) 
    status = Column(String) 
    summary = Column(String, nullable=True)
    transcript = Column(String, nullable=True) # Added transcript column
    recording_url = Column(String, nullable=True)
    duration = Column(Integer, default=0) 
    timestamp = Column(DateTime, default=datetime.utcnow)

# Table creation moved to startup_event

# ... (rest of code)

@app.post("/api/vapi/webhook")
async def vapi_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        message_type = payload.get("message", {}).get("type") or payload.get("type")
        print(f"Received Vapi Event: {message_type}")

        if message_type == "function-call":
            # ... (function call logic same as before) ...
            function_name = payload.get("functionCall", {}).get("name")
            parameters = payload.get("functionCall", {}).get("parameters", {})
            if function_name == "book_appointment":
                 return JSONResponse(content={"result": "Appointment booked successfully for " + parameters.get("time", "tomorrow")})
            elif function_name == "send_sms":
                 print(f"Vapi requested SMS to {parameters.get('phone')}: {parameters.get('message')}")
                 return JSONResponse(content={"result": "SMS logged"})

        elif message_type == "end-of-call-report":
             # Extract Data
             analysis = payload.get("analysis", {})
             summary = analysis.get("summary", "No summary provided.")
             recording_url = payload.get("recordingUrl")
             
             # Extract Duration (Vapi sends 'durationSeconds' or similar in top level or analysis)
             duration = payload.get("durationSeconds", 0)
             if not duration:
                 duration = payload.get("endedReason", {}).get("durationSeconds", 0) # sometimes here

             # Robust extraction of phone number
             customer_data = payload.get("customer", {})
             customer_number = customer_data.get("number")
             
             if not customer_number:
                 customer_number = payload.get("call", {}).get("customer", {}).get("number")
             
             if not customer_number:
                  customer_number = "Unknown"

             # Save to DB
             db = SessionLocal()
             try:
                 new_call = CallLog(
                     phone_number=customer_number,
                     call_type="inbound/outbound", 
                     status="completed",
                     summary=summary,
                     recording_url=recording_url,
                     duration=int(duration) if duration else 0
                 )
                 db.add(new_call)
                 db.commit()
             except Exception as db_e:
                 print(f"Failed to save call log: {db_e}")
             finally:
                 db.close()

             admin_email = os.getenv("MAIL_USERNAME")
             if admin_email:
                 email_body = f"<h1>New Call</h1><p>Duration: {duration}s</p><p>{summary}</p><p><a href='{recording_url}'>Recording</a></p>"
                 background_tasks.add_task(send_email_background, "New Call Summary", admin_email, email_body)

        return JSONResponse(content={"status": "ok"})
        
    except Exception as e:
        print(f"Error processing Vapi webhook: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Startup Event for Migrations
@app.on_event("startup")
async def startup_event():
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    # Auto-Migration: Add 'transcript' column if missing
    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('call_logs')]
        if 'transcript' not in columns:
            print("⚠️ 'transcript' column missing in 'call_logs'. Attempting auto-migration...", flush=True)
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE call_logs ADD COLUMN transcript TEXT"))
                conn.commit()
            print("✅ Auto-migration successful: 'transcript' column added!", flush=True)
        else:
            print("✅ DB Check: 'transcript' column exists.", flush=True)
    except Exception as e:
        print(f"❌ Auto-migration failed: {e}", flush=True)

# Define Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Auth Helpers ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# --- Email Config ---
conf = None
if os.getenv("MAIL_USERNAME") and os.getenv("MAIL_PASSWORD"):
    conf = ConnectionConfig(
        MAIL_USERNAME = os.getenv("MAIL_USERNAME"),
        MAIL_PASSWORD = os.getenv("MAIL_PASSWORD"),
        MAIL_FROM = os.getenv("MAIL_FROM", "noreply@missedcallsaviour.com"),
        MAIL_PORT = int(os.getenv("MAIL_PORT", 587)),
        MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com"),
        MAIL_FROM_NAME = os.getenv("MAIL_FROM_NAME", "Missed Call Saviour"),
        MAIL_STARTTLS = True,
        MAIL_SSL_TLS = False,
        USE_CREDENTIALS = True,
        VALIDATE_CERTS = True
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

async def send_email_background(subject: str, email_to: str, body: str):
    if not conf:
        print(f"Mock Sended Email: {subject} to {email_to}")
        return

    message = MessageSchema(
        subject=subject,
        recipients=[email_to],
        body=body,
        subtype=MessageType.html
    )
    
    fm = FastMail(conf)
    try:
        await fm.send_message(message)
        print(f"Email sent: {subject} to {email_to}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# --- Vapi Helper ---
async def trigger_vapi_outbound_call(phone: str, message: str = None):
    """
    Triggers an outbound call using Vapi.ai API
    """
    vapi_url = "https://api.vapi.ai/call"
    # Vapi Call
    vapi_private_key = VAPI_PRIVATE_KEY
    vapi_assistant_id = VAPI_ASSISTANT_ID
    vapi_phone_number_id = VAPI_PHONE_NUMBER_ID
    
    if not vapi_private_key or not vapi_assistant_id:
        print("Skipping Vapi call (not configured): Missing VAPI_PRIVATE_KEY or VAPI_ASSISTANT_ID")
        return

    webhook_url = f"{DOMAIN}/api/vapi/webhook"
    if "127.0.0.1" in webhook_url or "localhost" in webhook_url:
        webhook_url = "https://missed-call-saviour-ready-production.up.railway.app/api/vapi/webhook"

    payload = {
      "assistantId": vapi_assistant_id,
      "customer": {
        "number": phone
      },
      "phoneNumberId": vapi_phone_number_id,
    }
    
    # Only override serverUrl (webhook) to ensure we capture logs
    # We DO NOT override 'model' or 'messages' anymore, so Vapi Dashboard settings are used.
    payload["assistant"] = {
         "serverUrl": webhook_url
    }
    
    if message:
         payload["assistant"]["firstMessage"] = message

    headers = {
        "Authorization": f"Bearer {vapi_private_key}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(vapi_url, json=payload, headers=headers)
            if response.status_code != 201:
                print(f"Vapi Error {response.status_code}: {response.text}")
            response.raise_for_status()
            print(f"Successfully triggered Vapi call to {phone}")
    except httpx.HTTPStatusError as e:
        print(f"Vapi HTTP Error: {e.response.text}")
    except Exception as e:
        print(f"Failed to trigger Vapi call: {e}")

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/chatbot.js")
async def read_chatbot_js():
    with open(os.path.join(BASE_DIR, "chatbot.js"), "r", encoding="utf-8") as f:
        return JSONResponse(content=f.read(), media_type="application/javascript")

@app.get("/login", response_class=HTMLResponse)
async def read_login():
    with open(os.path.join(BASE_DIR, "login.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/signup", response_class=HTMLResponse)
async def read_signup():
    with open(os.path.join(BASE_DIR, "signup.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/privacy", response_class=HTMLResponse)
async def read_privacy():
    with open(os.path.join(BASE_DIR, "privacy.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/terms", response_class=HTMLResponse)
async def read_terms():
    with open(os.path.join(BASE_DIR, "terms.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/roi", response_class=HTMLResponse)
async def read_roi():
    with open(os.path.join(BASE_DIR, "roi.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/pricing", response_class=HTMLResponse)
async def read_pricing():
    with open(os.path.join(BASE_DIR, "pricing.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard():
    with open(os.path.join(BASE_DIR, "dashboard.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        print(f"Login Failed: User {form_data.username} not found in DB.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_password(form_data.password, user.hashed_password):
        print(f"Login Failed: Password mismatch for {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user.last_login = datetime.utcnow()
    db.commit()
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/signup")
async def signup(background_tasks: BackgroundTasks, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == email).first()
        if user:
             raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = get_password_hash(password)
        new_user = User(
            email=email, 
            hashed_password=hashed_password,
            registration_date=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_user)
        db.commit()
        
        welcome_body = """
        <h1>Welcome to Missed Call Saviour!</h1>
        <p>Hi there,</p>
        <p>Thanks for creating an account.</p>
        """
        background_tasks.add_task(send_email_background, "Welcome to Missed Call Saviour", email, welcome_body)
        return {"message": "User created successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/send-demo-call")
async def send_demo_call(
    background_tasks: BackgroundTasks, 
    phone: str = Form(...), 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    print(f"Received demo Call request for: {phone} from {current_user.email}")
    
    # 1. Log the call immediately
    try:
        new_call = CallLog(
            phone_number=phone,
            call_type="outbound-demo", 
            status="initiated",
            summary=f"Demo call initiated (Vapi Settings)",
            recording_url=None,
            duration=0,
            user_email=current_user.email 
        )
        db.add(new_call)
        db.commit()
    except Exception as e:
        print(f"Failed to log initial call: {e}")

    # Direct Call - strict usage of Vapi Dashboard Settings
    try:
        # We pass a generic opening message just to start the call, 
        # but the Persona/System Prompt comes from Vapi now.
        await trigger_vapi_outbound_call(phone, "Namaste! This is a demo call from Missed Call Saviour.")
        return {"success": True, "message": "Demo call initiated successfully."}
    except Exception as e:
        print(f"Error in demo call endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to initiate call", "details": str(e)})



@app.post("/api/process-payment")
async def process_payment(background_tasks: BackgroundTasks, email: str = Form(...), plan: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email)
        db.add(user)
    
    # Pricing Logic (Matches Razorpay Order)
    amount = 0
    if "Starter" in plan: amount = 10
    elif "Growth" in plan: amount = 50
    elif "Pro" in plan: amount = 100
    
    total_amount = amount 
    
    new_payment = Payment(user_email=email, amount=total_amount, plan_name=plan)
    db.add(new_payment)
    user.plan = plan
    user.is_active = True
    db.commit()

    body = f"<h1>Payment Successful!</h1><p>Amount: ${total_amount}</p>"
    background_tasks.add_task(send_email_background, "Welcome to Missed Call Saviour!", email, body)
    return {"success": True, "message": "Payment successful!", "redirect_url": "/dashboard?status=active"}

@app.post("/api/vapi/webhook")
async def vapi_webhook(request: Request, background_tasks: BackgroundTasks):
    import sys
    import json
    try:
        # RAW PAYLOAD LOGGING (CRITICAL)
        body_bytes = await request.body()
        raw_body_str = body_bytes.decode('utf-8')
        print(f"\n🔥 WEBHOOK RECEIVED! Raw Body: {raw_body_str}", flush=True)
        sys.stdout.flush()

        try:
           payload = json.loads(raw_body_str)
        except:
           print("Failed to parse JSON")
           return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        # Normalize message type
        message_type = payload.get("message", {}).get("type") or payload.get("type")
        print(f"Parsed Message Type: {message_type}", flush=True)
        sys.stdout.flush()

        # Extract Deeply Nested Logic
        # Vapi payloads can vary wildly. We assume 'message' or root.
        data = payload.get("message", payload)
        
        if message_type == "function-call":
             # ... (function call logic preserved) ...
            function_name = data.get("functionCall", {}).get("name")
            parameters = data.get("functionCall", {}).get("parameters", {})
            
            if function_name == "book_appointment":
                return JSONResponse(content={"result": "Appointment booked successfully for " + parameters.get("time", "tomorrow")})
            elif function_name == "send_sms":
                 print(f"SMS Request: {parameters}")
                 return JSONResponse(content={"result": "SMS logged"})

        elif message_type == "end-of-call-report" or message_type == "End Of Call Report":
             
             # Fields Extraction
             analysis = data.get("analysis", {})
             if not analysis:
                 analysis = data.get("call", {}).get("analysis", {})
             
             summary = analysis.get("summary", "No summary provided.")
             print(f"📝 SUMMARY EXTRACTED: {summary}", flush=True)
             
             recording_url = data.get("recordingUrl")
             if not recording_url:
                 recording_url = data.get("call", {}).get("recordingUrl")

             duration = data.get("durationSeconds")
             if not duration:
                 duration = data.get("call", {}).get("durationSeconds", 0)

             # Transcript Extraction
             transcript = data.get("transcript")
             if not transcript:
                 transcript = data.get("artifact", {}).get("transcript")
             if not transcript:
                 transcript = data.get("call", {}).get("transcript")
             
             if not transcript:
                 transcript = "Transcript not provided."

             # Phone Extraction
             customer_number = None
             customer_data = data.get("customer", {})
             if customer_data: 
                customer_number = customer_data.get("number")
             
             if not customer_number:
                 customer_number = data.get("call", {}).get("customer", {}).get("number")
             
             if not customer_number:
                 customer_number = "Unknown"

             print(f"Saving Call Log -> Phone: {customer_number}, Duration: {duration}s", flush=True)
             sys.stdout.flush()

             # Database Logic (Simplified Merge)
             db = SessionLocal()
             try:
                 existing_call = None
                 
                 # Try matching by Phone + Initiated
                 if customer_number != "Unknown":
                     existing_call = db.query(CallLog).filter(
                         CallLog.phone_number == customer_number,
                         CallLog.status == "initiated"
                     ).order_by(CallLog.id.desc()).first()
                 
                 # Fallback: Match ANY recent initiated call
                 if not existing_call:
                     existing_call = db.query(CallLog).filter(
                         CallLog.status == "initiated"
                     ).order_by(CallLog.id.desc()).first()

                 if existing_call:
                     print(f"🔥 MATCH FOUND! Merging Check: ID {existing_call.id}")
                     existing_call.status = "completed"
                     
                     # Update fields only if they have values
                     if summary and summary != "No summary provided.":
                         existing_call.summary = summary
                     
                     if transcript and transcript != "Transcript not provided.":
                        existing_call.transcript = transcript
                     
                     if recording_url:
                         existing_call.recording_url = recording_url
                     
                     if duration:
                         existing_call.duration = int(duration)
                         
                     db.commit()
                     print("Call Log MERGED Successfully!", flush=True)
                 else:
                     print("No Initiated call found. Creating new row.", flush=True)
                     new_call = CallLog(
                         phone_number=customer_number,
                         call_type="inbound", 
                         status="completed",
                         summary=summary,
                         transcript=transcript, 
                         recording_url=recording_url,
                         duration=int(duration) if duration else 0
                     )
                     db.add(new_call)
                     db.commit()
                     print("New Call Log SAVED Successfully!", flush=True)
                 
             except Exception as db_e:
                 print(f"Failed to save call log: {db_e}", flush=True)
             finally:
                 db.close()

             # Email Logic (Admin)
             admin_email = os.getenv("MAIL_USERNAME")
             if admin_email:
                 email_body = f"<h1>New Call</h1><p>Duration: {duration}s</p><p>{summary}</p><p><a href='{recording_url}'>Recording</a></p>"
                 background_tasks.add_task(send_email_background, "New Call Summary", admin_email, email_body)

        elif message_type == "status-update" and data.get("status") == "ended":
             # Fallback logic for when End-of-Call report is missing/delayed
             print("⚠️ Received status-update: ended. Attempting to mark as completed.", flush=True)
             
             # Extract sparse details
             customer_number = "Unknown"
             customer_data = data.get("customer", {})
             if customer_data: 
                customer_number = customer_data.get("number")
             
             if not customer_number:
                 customer_number = data.get("call", {}).get("customer", {}).get("number", "Unknown")

             duration = data.get("durationSeconds", 0)
             recording_url = data.get("recordingUrl")
             summary = "Call ended (Summary pending...)"

             # Database Logic (Simplified Merge)
             db = SessionLocal()
             try:
                 existing_call = None
                 if customer_number != "Unknown":
                     existing_call = db.query(CallLog).filter(
                         CallLog.phone_number == customer_number,
                         CallLog.status == "initiated"
                     ).order_by(CallLog.id.desc()).first()
                 
                 if not existing_call:
                     existing_call = db.query(CallLog).filter(
                         CallLog.status == "initiated"
                     ).order_by(CallLog.id.desc()).first()

                 if existing_call:
                     existing_call.status = "completed"
                     if duration: existing_call.duration = int(duration)
                     if recording_url: existing_call.recording_url = recording_url
                     # Only overwrite summary if it's currently default/empty
                     if not existing_call.summary or "Initiated" in existing_call.summary:
                         existing_call.summary = summary
                     
                     db.commit()
                     print("Call Marked Completed via status-update!", flush=True)
                 else:
                     print("No initiated call to complete via status-update.", flush=True)
             except Exception as e:
                 print(f"Error in status-update handler: {e}", flush=True)
             finally:
                 db.close()

        return {"status": "ok"}
    except Exception as e:
        print(f"Error processing Vapi webhook: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "plan": current_user.plan,
        "is_active": current_user.is_active,
        "stripe_customer_id": current_user.stripe_customer_id,
        "registration_date": current_user.registration_date
    }

@app.get("/api/calls")
async def get_call_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # In a real SaaS, filter by current_user.id or phone_number
    # For now in MVP (single tenant feeling), return last 20 calls
    logs = db.query(CallLog).order_by(CallLog.timestamp.desc()).limit(20).all()
    return logs

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 1. Total Calls
    total_calls = db.query(CallLog).count()
    
    # 2. Estimate Revenue (Mock logic: Each call "saves" $200 potentially)
    est_revenue = total_calls * 200
    
    # 3. Recent Activity (Last 5 calls)
    recent_calls = db.query(CallLog).order_by(CallLog.timestamp.desc()).limit(5).all()
    
    recent_activity = []
    for call in recent_calls:
        outcome = "Failed"
        if call.status == "completed":
            outcome = "Recovered"
        elif call.status == "initiated":
            outcome = "Calling..."
            
        recent_activity.append({
            "phone": call.phone_number,
            "time": call.timestamp, # specific formatting can be done in frontend
            "action": "Call Completed" if call.status == "completed" else "Outbound Call", 
            "status": outcome,
            "recording_url": call.recording_url
        })

    return {
        "missed_calls_saved": total_calls,
        "est_revenue": est_revenue,
        "engagement_rate": "100%", # Placeholder
        "recent_activity": recent_activity
    }

# --- AI Configuration Endpoints ---

class AIConfig(Base):
    __tablename__ = "ai_configs"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True) # Linking by email for MVP simplicity
    business_name = Column(String, default="My Business")
    greeting = Column(String, default="Namaste! Main kaise help kar sakta hoon?")
    persona = Column(String, default="friendly") # New Persona Field

# Table creation moved to startup_event

# ... (rest of code)

@app.get("/api/ai-config")
async def get_ai_config(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
    if not config:
        return {
            "business_name": "My Business",
            "greeting": "Namaste! Main My Business se bol raha hoon. Kya main aapki help kar sakta hoon?",
            "persona": "friendly"
        }
    return {
        "business_name": config.business_name,
        "greeting": config.greeting,
        "persona": config.persona if hasattr(config, "persona") else "friendly"
    }

@app.post("/api/ai-config")
async def update_ai_config(
    request: Request,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # 1. Save to DB
    try:
        data = await request.json()
        business_name = data.get("business_name")
        greeting = data.get("greeting")
        persona = data.get("persona", "friendly")
        
        config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
        if not config:
            config = AIConfig(user_email=current_user.email, business_name=business_name, greeting=greeting, persona=persona)
            db.add(config)
        else:
            config.business_name = business_name
            config.greeting = greeting
            config.persona = persona
        db.commit()
    except Exception as e:
        print(f"DB Update Error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Database Error: {str(e)}"})

    # 2. Update Vapi Assistant via API
    # 2. Update Vapi Assistant via API
    vapi_private_key = VAPI_PRIVATE_KEY
    vapi_assistant_id = VAPI_ASSISTANT_ID

    if not vapi_private_key or not vapi_assistant_id:
        print("Vapi environment variables missing.")
        # We perform a soft failure here - DB updated, but Vapi not connected
        return JSONResponse(content={"success": True, "message": "Settings saved to DB, but Vapi Agent not updated (Missing API Keys)."})

    try:
        vapi_url = f"https://api.vapi.ai/assistant/{vapi_assistant_id}"
        headers = {
            "Authorization": f"Bearer {vapi_private_key}",
            "Content-Type": "application/json"
        }
        
        # Determine Prompt based on Persona
        tone_instruction = ""
        if persona == "professional":
            tone_instruction = "Tone: Be strictly professional, polite, and concise. Use formal language (Aap, Sir/Ma'am). reliable and trustworthy."
        elif persona == "urgent":
            tone_instruction = "Tone: Be high-energy (sales mode). Create urgency. Focus on booking the appointment NOW. Use persuasive language."
        else: # friendly
            tone_instruction = "Tone: Be warm, engaging, and patient like a friend. Use natural Hinglish with common words like 'Ji', 'Haan', 'Thik hai'."

        # New "Conversational Hinglish" Prompt
        updated_prompt = f"""
        Role: You are a {persona} AI receptionist for {business_name}.
        Context: You are handling calls for an Indian business.
        Language: Speak in natural Hinglish (mix of Hindi and English).
        {tone_instruction}
        Task: 
        1. Start by welcoming them with: "{greeting}"
        2. Understand their query (Appointment, Price, or General Info).
        3. Explain details clearly in Hinglish.
        4. Always ask a follow-up question to keep the chat alive.
        
        Guardrails:
        - If they speak English, reply in English.
        - If they speak Hindi, reply in Hindi/Hinglish.
        - Never end the call abruptly.
        """
        
        payload = {
            "model": {
                "provider": "openai",
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": updated_prompt}
                ]
            },
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.patch(vapi_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                print(f"Vapi Update Error: {response.text}")
                return JSONResponse(status_code=500, content={"error": f"Vapi Error: {response.text}"})
                
        return {"success": True, "message": "AI Settings Updated!"}

    except Exception as e:
        print(f"Vapi Config Update Exception: {e}")
        return JSONResponse(status_code=500, content={"error": f"Vapi Client Error: {str(e)}"})

# Razorpay Routes
@app.post("/api/razorpay/create-order")
async def create_razorpay_order(email: str = Form(...), plan: str = Form(...)):
    if not razorpay_client:
        return JSONResponse(status_code=500, content={"error": "Razorpay not configured"})
    
    # Pricing Logic (subunits)
    # USD 10 -> 1000 cents
    amount = 0
    if "Starter" in plan: amount = 1000
    elif "Growth" in plan: amount = 5000
    elif "Pro" in plan: amount = 10000
    
    # Razorpay requires currency to charge in USD. Ensure your Razorpay account supports international payments.
    # If not, you might need to convert to INR (e.g. 10 USD ~ 850 INR -> 85000 paise)
    # For now, we proceed with USD as requested by user.
    data = { "amount": amount, "currency": "USD", "receipt": email, "notes": {"plan": plan} }
    try:
        order = razorpay_client.order.create(data=data)
        return {"id": order['id'], "amount": order['amount'], "currency": order['currency'], "key_id": RAZORPAY_KEY_ID}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/razorpay/verify")
async def verify_razorpay_payment(background_tasks: BackgroundTasks, razorpay_payment_id: str = Form(...), razorpay_order_id: str = Form(...), razorpay_signature: str = Form(...), email: str = Form(...), plan: str = Form(...), db: Session = Depends(get_db)):
    if not razorpay_client:
        return JSONResponse(status_code=500, content={"error": "Razorpay not configured"})
    try:
        params_dict = {'razorpay_order_id': razorpay_order_id, 'razorpay_payment_id': razorpay_payment_id, 'razorpay_signature': razorpay_signature}
        razorpay_client.utility.verify_payment_signature(params_dict)
        return await process_payment(background_tasks, email, plan, db)
    except razorpay.errors.SignatureVerificationError:
         return JSONResponse(status_code=400, content={"error": "Payment verification failed"})
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})

@app.on_event("startup")
async def startup_event():
    """
    Ensure the DB schema is up to date on startup.
    Compatible with both SQLite and PostgreSQL.
    """
    try:
        from sqlalchemy import text, inspect
        # ensure tables exist
        Base.metadata.create_all(bind=engine)

        # Use a fresh connection for migration check
        with engine.connect() as connection:
            # Check if ai_configs has persona column
            try:
                inspector = inspect(engine)
                if inspector.has_table("ai_configs"):
                    columns = [col['name'] for col in inspector.get_columns("ai_configs")]
                    
                    if "persona" not in columns:
                        print("Migrating DB: Adding persona column to ai_configs...")
                        # Use generic SQL standard syntax
                        connection.execute(text("ALTER TABLE ai_configs ADD COLUMN persona VARCHAR(50) DEFAULT 'friendly'"))
                        connection.commit()
                        print("Migration successful.")
            except Exception as e:
                print(f"Migration step failed: {e}")
                
    except Exception as e:
        print(f"Startup check failed: {e}")


@app.get("/api/debug-env")
async def debug_env():
    import os
    keys = list(os.environ.keys())
    vapi_key = os.getenv("VAPI_PRIVATE_KEY")
    vapi_id = os.getenv("VAPI_ASSISTANT_ID")
    return {
        "all_keys": [k for k in keys if "VAPI" in k or "DATABASE" in k],
        "vapi_private_key_exists": bool(vapi_key),
        "vapi_private_key_length": len(vapi_key) if vapi_key else 0,
        "vapi_private_key_first_5": vapi_key[:5] if vapi_key else "None",
        "vapi_assistant_id": vapi_id
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
