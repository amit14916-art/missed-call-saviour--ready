from fastapi import FastAPI, Request, Form, BackgroundTasks, Depends, HTTPException, Body, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr, BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, inspect, text, or_
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
import google.generativeai as genai
from config_secrets import DATABASE_URL, VAPI_PRIVATE_KEY, VAPI_ASSISTANT_ID, VAPI_PHONE_NUMBER_ID, RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, GEMINI_API_KEY
from fastapi import UploadFile, File
import shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# ... (rest of imports)

# Configure Gemini (Prefer Environment Variable)
try:
    FINAL_GEMINI_KEY = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    genai.configure(api_key=FINAL_GEMINI_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
    print("Gemini AI Configured Successfully.")
except Exception as e:
    print(f"Failed to configure Gemini: {e}")
    gemini_model = None

# Load environment variables
load_dotenv()

import asyncio
from fastapi.responses import StreamingResponse

# --- SSE Manager ---
class SSEManager:
    def __init__(self):
        self.connections = set()

    async def connect(self, request: Request):
        queue = asyncio.Queue()
        self.connections.add(queue)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Wait for message with timeout to send keep-alive
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            self.connections.remove(queue)

    async def broadcast(self, message: str):
        for connection in list(self.connections):
            await connection.put(message)

sse_manager = SSEManager()


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

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/debug-status")
async def get_debug_status():
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    return {
        "gemini_key_present": bool(api_key),
        "gemini_key_suffix": f"***{api_key[-4:]}" if api_key else "NONE",
        "database_connected": "True",
        "version": "1.3.0"
    }

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    plan = Column(String, default="Free")
    is_active = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False) # New Admin Flag
    phone_number = Column(String, nullable=True) # Added for user contact
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

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True) # Unique ID for each website visitor
    role = Column(String) # 'user' or 'model'
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String)
    answer = Column(String)
    embedding = Column(String) # Storing as stringified list for simplicity if pgvector isn't installed
    timestamp = Column(DateTime, default=datetime.utcnow)

# Table creation moved to startup_event

# ... (rest of code)

# Duplicate webhook handler removed. Using the robust one defined later.

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
            print("INFO: 'transcript' column missing in 'call_logs'. Attempting auto-migration...", flush=True)
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE call_logs ADD COLUMN transcript TEXT"))
                conn.commit()
            print("SUCCESS: Auto-migration successful: 'transcript' column added!", flush=True)
        else:
            print("SUCCESS: DB Check: 'transcript' column exists.", flush=True)
            
        # Auto-Migration: Add 'is_admin' to users if missing
        columns_users = [col['name'] for col in inspector.get_columns('users')]
        if 'is_admin' not in columns_users:
            print("INFO: 'is_admin' column missing in 'users'. Attempting auto-migration...", flush=True)
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE"))
                conn.execute(text("UPDATE users SET is_admin = TRUE WHERE id = 1")) # Make first user Admin
                conn.execute(text("UPDATE users SET is_admin = TRUE WHERE email = 'amit14916@gmail.com'")) # Ensure Amit is Admin
            print("SUCCESS: Auto-migration successful: 'is_admin' column added & User 1 promoted!", flush=True)

        # Force First User as Admin (Safety Net)
        try:
            with engine.connect() as conn:
                conn.execute(text("UPDATE users SET is_admin = TRUE WHERE id = 1"))
                conn.execute(text("UPDATE users SET is_admin = TRUE WHERE email = 'amit14916@gmail.com'"))
                conn.commit()
        except: pass

        # --- DEBUG: Verify API Key Source ---
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        usage_key = env_key if env_key else GEMINI_API_KEY.strip()
        
        source = "Environment Variable (Railway)" if env_key else "Config File (Hardcoded)"
        masked_key = f"********{usage_key[-4:]}" if len(usage_key) > 4 else "Not Set"
        
        print(f"INFO: Gemini Key Loaded From: {source}", flush=True)
        print(f"INFO: Active Key Suffix: {masked_key}", flush=True)
        
        if not usage_key:
            print("ERROR: CRITICAL: No Gemini API Key found!", flush=True)
        # ------------------------------------

        if 'phone_number' not in columns_users:
            print("INFO: 'phone_number' column missing in 'users'. Attempting auto-migration...", flush=True)
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)"))
                conn.commit()
            print("SUCCESS: Auto-migration successful: 'phone_number' column added!", flush=True)

        # AI Config Migrations (Merged from secondary startup event)
        if inspector.has_table("ai_configs"):
             columns_ai = [col['name'] for col in inspector.get_columns("ai_configs")]
             with engine.connect() as conn:
                 if "persona" not in columns_ai:
                     print("Migrating: Adding persona to ai_configs...")
                     conn.execute(text("ALTER TABLE ai_configs ADD COLUMN persona VARCHAR(50) DEFAULT 'friendly'"))
                     conn.commit()
                 if "owner_phone" not in columns_ai:
                     print("Migrating: Adding owner_phone to ai_configs...")
                     conn.execute(text("ALTER TABLE ai_configs ADD COLUMN owner_phone VARCHAR(50)"))
                     conn.commit()
                 if "vapi_assistant_id" not in columns_ai:
                     print("Migrating: Adding vapi_assistant_id to ai_configs...")
                     conn.execute(text("ALTER TABLE ai_configs ADD COLUMN vapi_assistant_id VARCHAR(100)"))
                     conn.commit()
                 if "eleven_labs_voice_id" not in columns_ai:
                     print("Migrating: Adding eleven_labs_voice_id to ai_configs...")
                     conn.execute(text("ALTER TABLE ai_configs ADD COLUMN eleven_labs_voice_id VARCHAR(100)"))
                     conn.commit()
                 if "eleven_labs_api_key" not in columns_ai:
                     print("Migrating: Adding eleven_labs_api_key to ai_configs...")
                     conn.execute(text("ALTER TABLE ai_configs ADD COLUMN eleven_labs_api_key VARCHAR(255)"))
                     conn.commit()

    except Exception as e:
        print(f"ERROR: Auto-migration failed: {e}", flush=True)

    # Force Specific Admin Update (Run every time to ensure access)
    try:
        with engine.connect() as conn:
            conn.execute(text("UPDATE users SET is_admin = TRUE WHERE email = 'amit14916@gmail.com'"))
            conn.commit()
            print("SUCCESS: Admin privileges explicitly granted to amit14916@gmail.com")
    except Exception as e:
        print(f"INFO: Failed to force admin update: {e}")

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
async def trigger_vapi_outbound_call(phone: str, message: str = None, user_email: str = None, assistant_id: str = None):
    """
    Triggers an outbound call using Vapi.ai API
    """
    vapi_url = "https://api.vapi.ai/call"
    
    # Sanitization is done upstream in send_demo_call
    print(f"Triggering Vapi call to {phone} (user={user_email})")
    
    # Vapi Call
    vapi_private_key = VAPI_PRIVATE_KEY
    vapi_assistant_id = assistant_id if assistant_id else VAPI_ASSISTANT_ID
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
      # "metadata": { "user_email": ... }  <-- Removed to reduce complexity/failure risk.
      # User assignment now relies on send_demo_call DB logging + phone matching.
    }
    
    # Only override serverUrl (webhook) to ensure we capture logs
    # We DO NOT override 'model' or 'messages' anymore, so Vapi Dashboard settings are used.
    # Inject Smart System Prompt for the Demo
    system_prompt = """
    Role: You are the Senior AI Sales Representative for 'Missed Call Saviour'.
    
    Product: 'Missed Call Saviour' is an AI tool that answers calls when business owners are busy. 
    
    CORE VALUE PROPOSITION (How it makes money):
    1. "Every missed call is a lost customer." If a customer calls and you don't pick up, they call your competitor.
    2. We answer INSTANTLY, 24/7. We take the booking or query.
    3. Even one saved customer per month pays for the entire subscription.
    
    INTEGRATION PROCESS (How it works):
    1. Sign up and get a dedicated AI Number.
    2. On your personal phone, enable "Conditional Call Forwarding" (e.g., *67* on most carriers).
    3. Now, whenever you miss a call or decline it, it automatically forwards to our AI Agent.
    
    Pricing Plans:
    1. Starter: $10/month (Solo founders, 100 calls).
    2. Growth: $50/month (Small teams, 500 calls + CRM sync).
    3. Pro: $100/month (Agencies, Unlimited calls, Custom Voice Cloning).
    
    Goal:
    - Convince the user that missing calls = losing money.
    - Explain that setup takes less than 2 minutes (just call forwarding).
    - Be confident, professional, and persuasive.
    - Mention: "I am an AI, but I can handle your entire front desk."
    
    Language: Speak in clear, professional English with a distinct Indian accent. Do not use Hinglish unless the user switches to Hindi. Focus on business value and ROI.
    """

    # VAPI_WEBHOOK_OVERRIDE_DISABLED: Let Vapi Dashboard settings take precedence.
    # To fix 'Call Not Coming', we revert the override as it might be malformed or conflicting.
    
    if message:
         # Still inject the first message if provided
         if "assistant" not in payload:
             payload["assistant"] = {}
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

@app.get("/api/sse")
async def sse_endpoint(request: Request):
    return StreamingResponse(sse_manager.connect(request), media_type="text/event-stream")


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

@app.get("/chat_only", response_class=HTMLResponse)
async def read_chat_only():
    with open(os.path.join(BASE_DIR, "chat_only.html"), "r", encoding="utf-8") as f:
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
async def signup(background_tasks: BackgroundTasks, email: str = Form(...), password: str = Form(...), phone: str = Form(None), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == email).first()
        if user:
             raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = get_password_hash(password)
        new_user = User(
            email=email, 
            hashed_password=hashed_password,
            phone_number=phone,
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
    
    # 0. Sanitize Phone Number (to ensure match with CallLog)
    if not phone.startswith("+"):
         clean_phone = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
         # Assume India (+91) if 10 digits
         if len(clean_phone) == 10:
             phone = f"+91{clean_phone}"
         else:
             phone = f"+{clean_phone}"
    
    print(f"Normalized Phone: {phone}")
    
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
        # Check if user has a custom assistant configured
        user_config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
        assistant_id = user_config.vapi_assistant_id if user_config else None
        
        # We pass a generic opening message just to start the call, 
        # but the Persona/System Prompt comes from Vapi now.
        await trigger_vapi_outbound_call(phone, "Hello! This is a demo call from Missed Call Saviour.", user_email=current_user.email, assistant_id=assistant_id)
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
        
        # MULTI-TENANT LOGIC: Identify User
        call_obj = data.get("call", {})
        metadata = payload.get("metadata") or call_obj.get("metadata") or {}
        
        # 1. Check Metadata (Primary Source for Outbound/Demo Calls)
        user_email = metadata.get("user_email")
        
        # 2. Check Assistant ID (Secondary Source for Inbound Calls from specific assistants)
        assistant_id = data.get("assistantId") or call_obj.get("assistantId")
        
        if not user_email and assistant_id:
            db_session = SessionLocal()
            try:
                # Find which user owns this specific AI Assistant
                config = db_session.query(AIConfig).filter(AIConfig.vapi_assistant_id == assistant_id).first()
                if config:
                    user_email = config.user_email
                    print(f"✅ Call Routed to User via Assistant ID: {user_email}")
                else:
                    print(f"⚠️ Unknown Assistant ID: {assistant_id}. Logging as Global/Admin.")
            except Exception as e:
                print(f"Error checking assistant_id: {e}")
            finally:
                db_session.close()
        
        if user_email:
             print(f"✅ Call Identified for User: {user_email}")

        
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
             
             summary = analysis.get("summary") or "No summary provided."

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
             
             transcript = transcript or "Transcript not provided."

             # Intelligent Fallback Summary (Runs if Vapi failed but Transcript exists)
             if (not summary or summary == "No summary provided.") and transcript and transcript != "Transcript not provided.":
                 keywords = []
                 text_lower = transcript.lower()
                 if "price" in text_lower or "cost" in text_lower or "rate" in text_lower: keywords.append("Pricing 💰")
                 if "appointment" in text_lower or "book" in text_lower or "schedule" in text_lower: keywords.append("Appointment 📅")
                 if "urgent" in text_lower or "emergency" in text_lower: keywords.append("Urgent 🚨")
                 if "refund" in text_lower: keywords.append("Refund 💸")
                 
                 validation_prefix = "📊 [Auto-Generated Insights]\n"
                 data_points = []
                 
                 if keywords:
                     data_points.append(f"• Key Topics: {', '.join(keywords)}")
                 else:
                     data_points.append(f"• Topic: General Inquiry")

                 # Simple Sentiment Analysis
                 if "angry" in text_lower or "upset" in text_lower or "bad" in text_lower: 
                     data_points.append("• Sentiment: Negative 🔴")
                 elif "thank" in text_lower or "great" in text_lower or "good" in text_lower: 
                     data_points.append("• Sentiment: Positive 🟢")
                 else: 
                     data_points.append("• Sentiment: Neutral ⚪")
                 
                 # Preview
                 data_points.append(f"• Context: \"{transcript[:60]}...\"")

                 summary = validation_prefix + "\n".join(data_points)

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
                         CallLog.status.in_(["initiated", "completed"]) # Allow updating calls marked via status-update
                     ).order_by(CallLog.id.desc()).first()
                 
                 # Fallback: Match ANY recent initiated call
                 if not existing_call:
                     existing_call = db.query(CallLog).filter(
                         CallLog.status.in_(["initiated", "completed"]) 
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
                     await sse_manager.broadcast("update_dashboard")
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
                         duration=int(duration) if duration else 0,
                         user_email=user_email # Link to specific user
                     )
                     db.add(new_call)
                     db.commit()
                     await sse_manager.broadcast("update_dashboard")
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
                     await sse_manager.broadcast("update_dashboard")
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
        "is_admin": current_user.is_admin,
        "stripe_customer_id": current_user.stripe_customer_id,
        "registration_date": current_user.registration_date
    }

@app.get("/api/calls")
async def get_call_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # MULTI-TENANT SECURITY: 
    # Show logs for logged-in user ONLY
    logs = db.query(CallLog).filter(
        CallLog.user_email == current_user.email
    ).order_by(CallLog.timestamp.desc()).limit(50).all()
    return logs


@app.get("/api/dashboard/stats")
async def get_dashboard_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 1. Total Calls (Filtered)
    total_calls = db.query(CallLog).filter(
        CallLog.user_email == current_user.email
    ).count()
    
    # 2. Estimate Revenue (Based on Industry Average $150 per lead saved)
    # Each call handled is considered a lead saved.
    LEAD_VALUE = 150 
    est_revenue = total_calls * LEAD_VALUE
    
    # 3. Recent Activity (Filtered)
    recent_calls = db.query(CallLog).filter(
        CallLog.user_email == current_user.email
    ).order_by(CallLog.timestamp.desc()).limit(5).all()

    
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
        
    # 4. Weekly Call Volume (Last 7 Days)
    from datetime import timedelta
    today = datetime.utcnow().date()
    weekly_volume = []
    
    # Generate labels (e.g., "Mon", "Tue") and data count for the last 7 days
    days_labels = []
    
    for i in range(6, -1, -1):
        day_start = datetime.combine(today - timedelta(days=i), datetime.min.time())
        day_end = datetime.combine(today - timedelta(days=i), datetime.max.time())
        
        count = db.query(CallLog).filter(
            or_(CallLog.user_email == current_user.email, CallLog.user_email == None),
            CallLog.timestamp >= day_start,
            CallLog.timestamp <= day_end
        ).count()
        
        weekly_volume.append(count)
        days_labels.append(day_start.strftime("%a"))

    return {
        "missed_calls_saved": total_calls,
        "est_revenue": est_revenue,
        "engagement_rate": "100%", # Placeholder
        "recent_activity": recent_activity,
        "weekly_volume": weekly_volume,
        "weekly_labels": days_labels
    }

# --- AI Configuration Endpoints ---

class AIConfig(Base):
    __tablename__ = "ai_configs"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True) # Linking by email for MVP simplicity
    business_name = Column(String, default="My Business")
    greeting = Column(String, default="Hello! How can I help you today?")
    persona = Column(String, default="friendly")
    owner_phone = Column(String, nullable=True)
    vapi_assistant_id = Column(String, nullable=True) # CRITICAL: Links User to their unique Vapi Agent # New Persona Field
    eleven_labs_voice_id = Column(String, nullable=True) # Voice Cloning
    eleven_labs_api_key = Column(String, nullable=True) # User's ElevenLabs Key

# Table creation moved to startup_event

# ... (rest of code)

@app.get("/api/ai-config")
async def get_ai_config(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
    if not config:
        return {
            "business_name": "My Business",
            "greeting": "Hello! I'm calling from My Business. How can I assist you?",
            "persona": "friendly"
        }
    return {
        "business_name": config.business_name,
        "greeting": config.greeting,
        "persona": config.persona if hasattr(config, "persona") else "friendly",
        "owner_phone": config.owner_phone if hasattr(config, "owner_phone") else "",
        "eleven_labs_voice_id": config.eleven_labs_voice_id if hasattr(config, "eleven_labs_voice_id") else "",
        "eleven_labs_api_key": config.eleven_labs_api_key if hasattr(config, "eleven_labs_api_key") else ""
    }

@app.post("/api/ai-config/clone-voice")
async def clone_voice(
    file: UploadFile = File(...),
    eleven_labs_api_key: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Clones a user's voice using ElevenLabs Instant Voice Cloning API.
    """
    import httpx
    
    # 1. Get API Key (Priority: Form > User Config > Env)
    api_key = eleven_labs_api_key
    if not api_key:
        config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
        if config and config.eleven_labs_api_key:
            api_key = config.eleven_labs_api_key
    
    if not api_key:
        api_key = os.getenv("ELEVEN_LABS_API_KEY") # System-wide fallback
        
    if not api_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API Key is required for voice cloning.")

    # 2. Call ElevenLabs API
    # API: POST https://api.elevenlabs.io/v1/voices/add
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {
        "xi-api-key": api_key
    }
    
    # Read file content
    content = await file.read()
    
    # Prepare payload for multipart/form-data
    files = {
        'files': (file.filename, content, file.content_type)
    }
    data = {
        'name': f"Voice_{current_user.email[:10]}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
        'description': f"Cloned via Missed Call Saviour Dashboard for {current_user.email}"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", {}).get("message", response.text)
                print(f"ElevenLabs Error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"ElevenLabs Error: {error_detail}")
            
            result = response.json()
            voice_id = result.get("voice_id")
            
            if not voice_id:
                raise HTTPException(status_code=500, detail="No voice_id returned from ElevenLabs.")
            
            # 3. Update DB (Optional, but good for persistence)
            config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
            if config:
                config.eleven_labs_voice_id = voice_id
                if eleven_labs_api_key:
                    config.eleven_labs_api_key = eleven_labs_api_key
                db.commit()
            
            return {"success": True, "voice_id": voice_id}
            
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

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
        owner_phone = data.get("owner_phone")
        
        config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
        if not config:
            config = AIConfig(
                user_email=current_user.email, 
                business_name=business_name, 
                greeting=greeting, 
                persona=persona, 
                owner_phone=owner_phone,
                eleven_labs_voice_id=data.get("eleven_labs_voice_id"),
                eleven_labs_api_key=data.get("eleven_labs_api_key")
            )
            db.add(config)
        else:
            config.business_name = business_name
            config.greeting = greeting
            config.persona = persona
            config.owner_phone = owner_phone
            config.eleven_labs_voice_id = data.get("eleven_labs_voice_id")
            config.eleven_labs_api_key = data.get("eleven_labs_api_key")
        db.commit()
    except Exception as e:
        print(f"DB Update Error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Database Error: {str(e)}"})

    # 2. Update Vapi Assistant via API
    try:
        vapi_private_key = VAPI_PRIVATE_KEY
        if not vapi_private_key:
             print("Vapi environment variables missing.", flush=True)
             return {"success": True, "message": "Settings saved to DB (Vapi Not Configured)."}

        # Define Webhook URL (Production)
        webhook_url = "https://missed-call-saviour-ready-production.up.railway.app/api/vapi/webhook"

        vapi_url = "https://api.vapi.ai/assistant"
        method = "POST"
        
        # If User already has an Assistant, UPDATE it (PATCH). If not, CREATE one (POST).
        # Check against "Not Created" string just in case legacy data exists
        if config.vapi_assistant_id and config.vapi_assistant_id != "Not Created":
            vapi_url = f"https://api.vapi.ai/assistant/{config.vapi_assistant_id}"
            method = "PATCH"
            print(f"Updating Existing Assistant: {config.vapi_assistant_id}")
        else:
            print("Creating NEW Vapi Assistant for User...")

        headers = {
            "Authorization": f"Bearer {vapi_private_key}",
            "Content-Type": "application/json"
        }
        
        # Determine Prompt based on Persona
        tone_instruction = ""
        if persona == "professional":
            tone_instruction = "Tone: Be strictly professional, polite, and concise. Use formal Indian English. Reliable and trustworthy."
        elif persona == "urgent":
            tone_instruction = "Tone: Be high-energy (sales mode). Create urgency. Focus on booking the appointment NOW. Use persuasive Indian English."
        else: # friendly
            tone_instruction = "Tone: Be warm, engaging, and patient like a friend. Use clear Indian English with a welcoming vibe."

        # New "Indian English" Prompt
        updated_prompt = f"""
        Role: You are a {persona} AI receptionist for {business_name}.
        Context: You are handling calls for an Indian business.
        Language: Speak in clear, professional English with an Indian accent. Do not use Hinglish unless requested.
        {tone_instruction}
        Task: 
        1. Start by welcoming them with: "{greeting}"
        2. Always be polite.
        """
        
        payload = {
            "name": f"{business_name} - Assistant",
            "model": {
                "provider": "openai",
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": updated_prompt}
                ]
            },
            "serverUrl": webhook_url
        }

        # Inject ElevenLabs Voice if provided
        if config.eleven_labs_voice_id:
            payload["voice"] = {
                "provider": "eleven-labs",
                "voiceId": config.eleven_labs_voice_id
            }
            if config.eleven_labs_api_key:
                payload["voice"]["apiKey"] = config.eleven_labs_api_key
        
        async with httpx.AsyncClient() as client:
            if method == "POST":
                response = await client.post(vapi_url, json=payload, headers=headers)
            else:
                response = await client.patch(vapi_url, json=payload, headers=headers)
            
            if response.status_code not in [200, 201]:
                print(f"Vapi Error: {response.text}")
                # Return 'detail' so frontend can display the actual error message
                return JSONResponse(status_code=400, content={"detail": f"Vapi Error: {response.text}"})
            
            # If we created a new assistant, save the ID
            if method == "POST":
                new_assistant = response.json()
                config.vapi_assistant_id = new_assistant.get("id")
                db.commit()
                print(f"✅ New Assistant Created & Linked: {config.vapi_assistant_id}")
                
        return {"success": True, "message": "AI Settings Updated & Unique Agent Deployed!"}

    except Exception as e:
        print(f"Vapi Config Update Exception: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Vapi Client Error: {str(e)}"})

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

# --- Android / Automation Hook ---
@app.get("/api/hooks/android-trigger")
async def android_trigger_callback(
    phone: str, 
    email: str, 
    db: Session = Depends(get_db)
):
    """
    Called by MacroDroid/IFTTT when a missed call occurs on the phone.
    Triggers the AI Agent to call the 'phone' number immediately.
    """
    print(f"📳 Mobile Hook Triggered! Owner: {email}, Target: {phone}")
    
    # 1. Clean Phone Number
    target_phone = phone.replace(" ", "").replace("-", "")
    if not target_phone.startswith("+"):
        # Assume India +91 if missing, or generic +
        target_phone = "+" + target_phone
        
    # 2. Get User Config
    config = db.query(AIConfig).filter(AIConfig.user_email == email).first()
    if not config or not config.vapi_assistant_id:
        return JSONResponse(status_code=404, content={"error": "User AI Agent not configured yet."})
        
    # 3. Call Vapi (Outbound)
    try:
        from config_secrets import VAPI_PHONE_NUMBER_ID, VAPI_PRIVATE_KEY
        
        url = "https://api.vapi.ai/call"
        headers = {
            "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "assistantId": config.vapi_assistant_id,
            "phoneNumberId": VAPI_PHONE_NUMBER_ID, # Uses the System Twilio Number
            "customer": {
                "number": target_phone
            }
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers)
            print(f"Vapi Call Request: {resp.status_code} - {resp.text}")
            
            if resp.status_code == 201:
                return {"success": True, "message": "AI is calling back now!", "target": target_phone}
            else:
                return JSONResponse(status_code=500, content={"error": "Vapi Failed", "details": resp.json()})
                
    except Exception as e:
        print(e)
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


# ------------------------------
# DEBUG ENDPOINT (TEMPORARY)
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

# --- ADMIN ENDPOINTS ---
@app.get("/api/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Security Check: Must be Admin (or User ID 1 for recovery)
    if not current_user.is_admin and current_user.id != 1:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    # Fetch all users with their configurations
    users_data = []
    all_users = db.query(User).all()
    
    for u in all_users:
        config = db.query(AIConfig).filter(AIConfig.user_email == u.email).first()
        call_count = db.query(CallLog).filter(CallLog.user_email == u.email).count()
        
        users_data.append({
            "id": u.id,
            "email": u.email,
            "plan": u.plan,
            "status": "Active" if u.is_active else "Inactive",
            "is_admin": u.is_admin,
            "business_name": config.business_name if config else "N/A",
            "vapi_assistant_id": config.vapi_assistant_id if config else "Not Created",
            "total_calls": call_count,
            "joined_at": u.registration_date.strftime("%Y-%m-%d") if u.registration_date else "N/A"
        })
        
    return users_data

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
    
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# --- Gemini AI Intelligence Routes ---

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/api/vapi-knowledge")
async def vapi_knowledge_retrieval(payload: dict = Body(...), db: Session = Depends(get_db)):
    """
    Experimental: Allows Vapi to query Alex's web chat memory via tool call or assistant context.
    Identifies user by phone number.
    """
    customer_phone = payload.get("message", {}).get("call", {}).get("customer", {}).get("number")
    
    if not customer_phone:
        return {"context": "No previous history found for this number."}
        
    # Find messages linked to this phone (requires session <-> phone mapping)
    # For now, let's look for any user who registered with this phone
    user = db.query(User).filter(User.phone_number == customer_phone).first()
    if user:
        # In a real app, you'd map session_id to user.email
        # We can search for the last few interactions for this user's email if stored
        return {"context": f"This user is a {user.plan} plan member. They joined on {user.registration_date.date()}."}

    return {"context": "New lead detected. Be helpful and charismatic."}

import json

async def get_embedding(text: str, api_key: str):
    """Generates embedding for RAG using Gemini's embedding model."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]}
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()["embedding"]["values"]
        except Exception as e:
            print(f"Embedding Error: {e}")
    return None

def cosine_similarity(v1, v2):
    if not v1 or not v2: return 0
    dot = sum(a*b for a,b in zip(v1, v2))
    norm1 = sum(a*a for a in v1)**0.5
    norm2 = sum(b*b for b in v2)**0.5
    return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0

@app.post("/api/analyze-chat")
async def analyze_chat_message(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Alex's intelligence with RAG and strict role alternation for Gemini.
    """
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "GEMINI_API_KEY is not set in environment or config."})
    
    # 1. RAG context fetching
    rag_context = ""
    try:
        current_emb = await get_embedding(request.message, api_key)
        if current_emb:
            kb_items = db.query(KnowledgeBase).all()
            scored_items = []
            for item in kb_items:
                try:
                    item_emb = json.loads(item.embedding)
                    if cosine_similarity(current_emb, item_emb) > 0.85:
                        scored_items.append(f"Q: {item.question}\nA: {item.answer}")
                except: continue
            if scored_items:
                rag_context = "\nRelevant Knowledge:\n" + "\n---\n".join(scored_items[:3])
    except Exception as e:
        print(f"RAG Error: {e}")

    # 2. Chat History with Role Alternation
    try:
        history = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id)\
                    .order_by(ChatMessage.timestamp.desc()).limit(6).all()
        history.reverse()
    except:
        history = []

    system_persona = f"You are Alex, the senior AI at Missed Call Saviour. Be charismatic and helpful. {rag_context}"
    
    # Construct alternating contents
    contents = [
        {"role": "user", "parts": [{"text": system_persona}]},
        {"role": "model", "parts": [{"text": "Understood. I am Alex, ready to assist."}]}
    ]

    last_role = "model"
    for m in history:
        current_role = "user" if m.role == "user" else "model"
        if current_role != last_role:
            contents.append({"role": current_role, "parts": [{"text": m.content}]})
            last_role = current_role
    
    # Ensure next is user
    if last_role == "user":
        contents.append({"role": "model", "parts": [{"text": "I see. How else can I help?"}]})
    
    contents.append({"role": "user", "parts": [{"text": request.message}]})

    # 3. Call Gemini
    models = ["gemini-1.5-flash", "gemini-pro"]
    last_error = "Connection Failed"

    async with httpx.AsyncClient() as client:
        # Save User Message
        try:
            db.add(ChatMessage(session_id=request.session_id, role="user", content=request.message))
            db.commit()
        except: db.rollback()

        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                resp = await client.post(url, json={"contents": contents}, timeout=25.0)
                if resp.status_code == 200:
                    data = resp.json()
                    reply = data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Save AI Reply
                    try:
                        db.add(ChatMessage(session_id=request.session_id, role="model", content=reply))
                        db.commit()
                    except: db.rollback()
                    
                    return {"reply": reply}
                else:
                    error_text = resp.text or f"HTTP {resp.status_code}"
                    last_error = f"Gemini API Error: {error_text}"
            except Exception as e:
                last_error = f"Connection error: {str(e)}"
                continue

    return JSONResponse(status_code=500, content={"error": f"Alex is offline. Reason: {last_error}"})

@app.post("/api/upload-call-recording")
async def upload_call_recording(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_email = current_user.email
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{datetime.now().timestamp()}_{file.filename}"
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Analyze with Gemini (Audio Native)
        summary = "Audio recording received."
        transcript = "Transcript not provided."
        
        try:
            if gemini_model:
                # Upload to Gemini for native audio analysis
                audio_file = genai.upload_file(path=str(file_path))
                prompt = "Please provide a detailed transcript and a concise summary of this audio recording. Format: Transcript: [text] Summary: [text]"
                response = gemini_model.generate_content([prompt, audio_file])
                
                res_text = response.text
                if "Transcript:" in res_text and "Summary:" in res_text:
                    transcript = res_text.split("Summary:")[0].replace("Transcript:", "").strip()
                    summary = res_text.split("Summary:")[1].strip()
                else:
                    summary = res_text.strip()
                    transcript = "Transcript integrated in summary."
        except Exception as ai_e:
            print(f"Gemini Audio Analysis failed: {ai_e}")
            summary = "Audio recording received. (AI Analysis failed)"
        
        # 3. Save to DB
        new_call = CallLog(
            phone_number="App-Recording",
            call_type="app-recording",
            status="completed",
            summary=summary,
            transcript=transcript,
            recording_url=str(file_path),
            duration=0,
            user_email=user_email,
            timestamp=datetime.utcnow() # Explicitly set for instant visibility
        )
        db.add(new_call)
        db.commit()
        
        await sse_manager.broadcast("update_dashboard")
        return {"status": "success", "message": "Recording uploaded", "summary": summary}
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Admin Routes ---
@app.get("/api/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = db.query(User).all()
    user_list = []
    for u in users:
        # Get extra details
        stats = db.query(CallLog).filter(or_(CallLog.user_email == u.email, CallLog.user_email == None)).count()
        config = db.query(AIConfig).filter(AIConfig.user_email == u.email).first()
        
        user_list.append({
            "id": u.id,
            "email": u.email,
            "business_name": config.business_name if config else "N/A",
            "plan": u.plan,
            "status": "Active" if u.is_active else "Inactive",
            "is_admin": u.is_admin,
            "total_calls": stats,
            "vapi_assistant_id": config.vapi_assistant_id if config else "Not Created",
            "joined_at": u.registration_date.strftime("%Y-%m-%d")
        })
    return user_list

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

# --- Demo Call Route ---
@app.post("/api/send-demo-call")
async def send_demo_call(
    phone: str = Form(...), 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """
    Triggers an outbound call from Vapi to the user's phone number as a demo.
    """
    # 1. Get User Config or Default
    config = db.query(AIConfig).filter(AIConfig.user_email == current_user.email).first()
    assistant_id = config.vapi_assistant_id if config and config.vapi_assistant_id else VAPI_ASSISTANT_ID
    
    if not assistant_id or assistant_id == "Not Created":
         return JSONResponse(status_code=400, content={"details": "AI Assistant not configured yet. Save settings first."})

    vapi_url = "https://api.vapi.ai/call"
    headers = {
        "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "assistantId": assistant_id,
        "customer": {
            "number": phone
        },
        "phoneNumberId": VAPI_PHONE_NUMBER_ID  # Must be a purchased Vapi/Twilio number
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(vapi_url, json=payload, headers=headers)
            if response.status_code == 201:
                # Log the initiated call
                new_log = CallLog(
                    user_email=current_user.email,
                    phone_number=phone,
                    call_type="outbound-demo",
                    status="initiated",
                    summary="Demo call triggered from dashboard."
                )
                db.add(new_log)
                db.commit()
                await sse_manager.broadcast("update_dashboard")
                return {"success": True, "message": "Demo call initiated successfully."}
            else:
                return JSONResponse(status_code=response.status_code, content={"details": response.text})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/calls/{call_id}/re-summarize")
async def re_summarize_call(call_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    call = db.query(CallLog).filter(CallLog.id == call_id, CallLog.user_email == current_user.email).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if not call.transcript or call.transcript == "Transcript not provided.":
        # If no transcript, we can't summarize
        raise HTTPException(status_code=400, detail="Cannot summarize without a transcript.")

    try:
        # Use Gemini to summarize the transcript
        if gemini_model:
            prompt = f"Please provide a concise summary of the following call transcript:\n\n{call.transcript}"
            response = gemini_model.generate_content(prompt)
            call.summary = response.text
            db.commit()
            await sse_manager.broadcast("update_dashboard")
            return {"success": True, "summary": response.text}
        else:
            raise HTTPException(status_code=500, detail="Gemini AI not configured.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
