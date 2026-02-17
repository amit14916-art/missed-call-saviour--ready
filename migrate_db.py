import os
import sys
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from main import Base, User, Payment, CallLog, AIConfig  # Import models from main.py
from dotenv import load_dotenv

load_dotenv()

# Configuration
SQLITE_URL = "sqlite:///./missed_calls.db"
POSTGRES_URL = os.getenv("DATABASE_URL")

if not POSTGRES_URL:
    print("Error: DATABASE_URL not set in .env")
    sys.exit(1)

if POSTGRES_URL.startswith("postgres://"):
    POSTGRES_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)

def migrate():
    print("Starting migration from SQLite to PostgreSQL...")
    
    # 1. Connect to SQLite
    sqlite_engine = create_engine(SQLITE_URL)
    sqlite_session = sessionmaker(bind=sqlite_engine)()
    print("Connected to SQLite.")

    # 2. Connect to PostgreSQL
    try:
        pg_engine = create_engine(POSTGRES_URL)
        pg_connection = pg_engine.connect()
        pg_session = sessionmaker(bind=pg_engine)()
        print("Connected to PostgreSQL.")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        print("Please check your DATABASE_URL in .env and ensure the database exists.")
        return

    # 3. Create Tables in PostgreSQL
    print("Creating tables in PostgreSQL...")
    Base.metadata.drop_all(pg_engine) # Optional: Drop existing tables to start fresh
    Base.metadata.create_all(pg_engine)
    print("Tables created.")

    # 4. Migrate Data
    tables = [User, Payment, CallLog, AIConfig]
    
    for table_model in tables:
        table_name = table_model.__tablename__
        print(f"Migrating table: {table_name}...")
        
        # Fetch data from SQLite
        records = sqlite_session.query(table_model).all()
        count = len(records)
        print(f"Found {count} records in SQLite for {table_name}.")
        
        if count > 0:
            # Insert into PostgreSQL
            # We must detach instances from the SQLite session before adding to Postgres session
            # or simply create new instances/dicts. 
            # Check if we can just merge.
            
            for record in records:
                # Expunge from source session so it can be added to another
                sqlite_session.expunge(record)
                pg_session.merge(record) 
                
            pg_session.commit()
            print(f"Migrated {count} records for {table_name}.")
        else:
            print(f"No records to migrate for {table_name}.")

    print("Migration completed successfully!")
    sqlite_session.close()
    pg_session.close()
    pg_connection.close()

if __name__ == "__main__":
    confirm = input("This will overwrite data in the PostgreSQL database defined in .env. Continue? (y/n): ")
    if confirm.lower() == 'y':
        migrate()
    else:
        print("Migration cancelled.")
