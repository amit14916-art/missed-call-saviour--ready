from sqlalchemy import create_engine, text
from config_secrets import DATABASE_URL

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        print("Connected to DB")
        
        # Check users
        result = conn.execute(text("SELECT id, email, is_admin FROM users WHERE email LIKE '%amit%@gmail.com'"))
        rows = result.fetchall()
        print(f"found {len(rows)} users:")
        for r in rows:
            print(r)
            
except Exception as e:
    print(f"Error: {e}")
