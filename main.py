from fastapi import FastAPI, Request, Form, BackgroundTasks, Depends, HTTPException, Body, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
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
DOMAIN = "http://127.0.0.1:8000"
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL")

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) if RAZORPAY_KEY_ID else None


@app.post("/api/razorpay/create-order")
async def create_razorpay_order(email: str = Form(...), plan: str = Form(...)):
    if not razorpay_client:
        return JSONResponse(status_code=500, content={"error": "Razorpay not configured"})
    
    amount = 4900 if "Starter" in plan else 9900 # In paise (4900 = 49 INR)
    data = { "amount": amount, "currency": "INR", "receipt": email, "notes": {"plan": plan} }
    
    try:
        order = razorpay_client.order.create(data=data)
        return {"id": order['id'], "amount": order['amount'], "currency": order['currency'], "key_id": RAZORPAY_KEY_ID}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/razorpay/verify")
async def verify_razorpay_payment(
    background_tasks: BackgroundTasks,
    razorpay_payment_id: str = Form(...),
    razorpay_order_id: str = Form(...),
    razorpay_signature: str = Form(...),
    email: str = Form(...),
    plan: str = Form(...),
    db: Session = Depends(get_db)
):
    if not razorpay_client:
        return JSONResponse(status_code=500, content={"error": "Razorpay not configured"})
        
    try:
        # Verify Signature
        params_dict = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        razorpay_client.utility.verify_payment_signature(params_dict)
        
        # If successful, activate user
        return await process_payment(background_tasks, email, plan, "RAZORPAY", db)
        
    except razorpay.errors.SignatureVerificationError:
         return JSONResponse(status_code=400, content={"error": "Payment verification failed"})
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})


# --- Security Config ---
# --- Security Config ---
SECRET_KEY = "super-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

# Use sha256_crypt to avoid binary dependency issues on Render
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Fix Render's postgres:// to postgresql:// if needed
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    engine = create_engine(DATABASE_URL)
else:
    # Local fallback
    print("WARNING: Using Local SQLite Database. Data will be lost on re-deploy!")
    SQLALCHEMY_DATABASE_URL = "sqlite:////tmp/missed_calls.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    plan = Column(String, default="Free")
    is_active = Column(Boolean, default=False)
    registration_date = Column(DateTime, default=datetime.utcnow)
    stripe_customer_id = Column(String, nullable=True)
    
class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    amount = Column(Float)
    plan_name = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    stripe_session_id = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Missed Call Saviour Backend")

# --- Auth Helpers ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
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

# Email Config
# Email Config
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

async def trigger_make_webhook(event_type: str, data: dict):
    """
    Sends data to Make.com webhook
    """
    if not MAKE_WEBHOOK_URL or "your-webhook-id" in MAKE_WEBHOOK_URL:
        print(f"Skipping Make.com webhook (not configured): {event_type}")
        return

    payload = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }

    try:
        async with httpx.AsyncClient() as client:
            await client.post(MAKE_WEBHOOK_URL, json=payload)
            print(f"Successfully triggered Make.com webhook for {event_type}")
    except Exception as e:
        print(f"Failed to trigger Make.com webhook: {e}")

# --- Helper Functions ---
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

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/signup")
async def signup(background_tasks: BackgroundTasks, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == email).first()
        if user:
             raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = get_password_hash(password)
        new_user = User(email=email, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        
        # Send Welcome Email
        welcome_body = f"""
        <h1>Welcome to Missed Call Saviour!</h1>
        <p>Hi there,</p>
        <p>Thanks for creating an account. We are excited to help you recover lost revenue from missed calls.</p>
        <p>Get started by setting up your integrations in the dashboard.</p>
        <br>
        <p>Best,</p>
        <p>The Missed Call Saviour Team</p>
        """
        background_tasks.add_task(send_email_background, "Welcome to Missed Call Saviour", email, welcome_body)
        
        return {"message": "User created successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/api/send-demo-sms")
async def send_demo_sms(background_tasks: BackgroundTasks, phone: str = Form(...)):
    """
    Simulates sending a demo SMS.
    Triggers Make.com to actually send it if configured there.
    """
    print(f"Received demo SMS request for: {phone}")
    
    webhook_data = {
        "phone": phone,
        "type": "demo_request",
        "message": "Hey! This is your Missed Call Saviour demo. Imagine this text just saved you a customer."
    }
    
    # Send to Make.com to handle the actual SMS (via Twilio/etc connected to Make)
    background_tasks.add_task(trigger_make_webhook, "demo_sms_request", webhook_data)
    
    return {"success": True, "message": "Demo SMS queued."}

@app.post("/api/create-checkout-session")
async def create_checkout_session(email: str = Form(...), plan: str = Form(...)):
    """
    Creates a Stripe Checkout Session for real payments.
    """
    if not stripe.api_key:
         return JSONResponse(status_code=400, content={"error": "Stripe API Key is missing. Check .env"})

    price_id = "price_1Ot..." # You need to replace this with your actual Stripe Price ID for the plan
    
    # Dynamic Price Lookup (Simplified)
    if "Starter" in plan:
        amount = 4900 # $49.00
    else:
        amount = 9900 # $99.00
        
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f"Missed Call Saviour - {plan} Plan",
                        },
                        'unit_amount': amount,
                    },
                    'quantity': 1,
                },
            ],
            mode='payment', # Or 'subscription' if you have recurring
            success_url=DOMAIN + '/dashboard?status=active&session_id={CHECKOUT_SESSION_ID}',
            cancel_url=DOMAIN + '/pricing',
            customer_email=email,
        )
        return {"checkout_url": checkout_session.url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/process-payment")
async def process_payment(
    background_tasks: BackgroundTasks,
    email: str = Form(...),
    plan: str = Form(...),
    card_number: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Legacy/Fallback Mock Payment Processor
    """
    # 1. Start or Update User
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email)
        db.add(user)
    
    # 2. Logic for Registration Fee
    REGISTRATION_FEE = 10
    plan_price = 49 if "Starter" in plan else 99
    total_amount = plan_price + REGISTRATION_FEE

    # 3. Record 'Payment'
    new_payment = Payment(
        user_email=email,
        amount=total_amount,
        plan_name=plan
    )
    db.add(new_payment)
    
    # 4. Activate User Plan
    user.plan = plan
    user.is_active = True
    db.commit()

    # 5. Send Email (if configured)
    body = f"""
    <h1>Payment Successful!</h1>
    <p>Hi there,</p>
    <p>Your payment of <strong>${total_amount}</strong> was successful.</p>
    <p>Plan: {plan} (${plan_price}/mo)</p>
    <p>Registration Fee: ${REGISTRATION_FEE} (Paid)</p>
    <br>
    <p>Welcome to the family!</p>
    """
    background_tasks.add_task(send_email_background, "Welcome to Missed Call Saviour!", email, body)

    # 6. Trigger Make.com Webhook
    webhook_data = {
        "email": email,
        "plan": plan,
        "amount": total_amount,
        "status": "paid"
    }
    background_tasks.add_task(trigger_make_webhook, "new_subscription", webhook_data)

    return {
        "success": True,
        "message": "Payment and registration successful!",
        "redirect_url": "/dashboard?status=active"
    }

@app.post("/api/vapi/webhook")
async def vapi_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming webhooks from Vapi.ai (e.g., call logs, function calls).
    """
    try:
        payload = await request.json()
        message_type = payload.get("message", {}).get("type") or payload.get("type")
        
        print(f"Received Vapi Event: {message_type}")

        if message_type == "function-call":
            # Example: Assistant wants to "book_appointment"
            function_name = payload.get("functionCall", {}).get("name")
            parameters = payload.get("functionCall", {}).get("parameters", {})
            
            if function_name == "book_appointment":
                # Logic to book appointment
                return JSONResponse(content={"result": "Appointment booked successfully for " + parameters.get("time", "tomorrow")})
            
            elif function_name == "send_sms":
                 # Trigger Make.com to send SMS
                 phone = parameters.get("phone")
                 message = parameters.get("message")
                 background_tasks.add_task(trigger_make_webhook, "vapi_sms_request", {"phone": phone, "message": message})
                 return JSONResponse(content={"result": "SMS sent successfully"})

        elif message_type == "end-of-call-report":
             # Log call summary to database/Make.com
             summary = payload.get("analysis", {}).get("summary", "No summary provided.")
             recording_url = payload.get("recordingUrl")
             
             # Notify Make.com
             background_tasks.add_task(trigger_make_webhook, "call_completed", {"summary": summary, "recording": recording_url})
             
             # Notify Admin/User via Email (simulated to Admin for now)
             admin_email = os.getenv("MAIL_USERNAME")
             if admin_email:
                 email_body = f"""
                 <h1>New Missed Call Handled</h1>
                 <p><strong>Summary:</strong> {summary}</p>
                 <p><strong>Recording:</strong> <a href="{recording_url}">Listen Here</a></p>
                 """
                 background_tasks.add_task(send_email_background, "New Call Summary", admin_email, email_body)

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
        "stripe_customer_id": current_user.stripe_customer_id
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
