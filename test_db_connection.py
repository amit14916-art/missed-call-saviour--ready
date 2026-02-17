import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL is not set in .env")
    print("Please follow the steps in SUPABASE_GUIDE.md to set it.")
    sys.exit(1)

print(f"Testing connection to: {DATABASE_URL}")

try:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # Add pool_pre_ping for better robustness
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version()"))
        version = result.scalar()
        print(f"✅ Success! Connected to database version: {version}")
        
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    # Suggest specific fixes
    if "Connection refused" in str(e):
        print(" -> Check if the database host and port are correct.")
        print(" -> If local, ensure Postgres service is running.")
    elif "password authentication failed" in str(e):
        print(" -> Check your PASSWORD in the connection string.")
    elif "does not exist" in str(e):
        print(" -> Check if the DATABASE NAME is correct (default is often 'postgres').")
