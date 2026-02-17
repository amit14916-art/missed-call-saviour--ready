# ⚡ Supabase Setup Guide

Use Supabase (a free PostgreSQL host) for your database.

## 1. Get Your Connection String
1.  Go to [Supabase Dashboard](https://supabase.com/dashboard/projects) -> Select your project.
2.  Click **Project Settings** (gear icon at the bottom left).
3.  Click **Database**.
4.  Scroll down to **Connection parameters** or **Connection string**.
5.  Click on **URI** tab. It should look like this:
    `postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres`
    
    > **Note:** Replace `[YOUR-PASSWORD]` with the password you created when setting up the project. If you forgot it, click "Reset Database Password" in the connection settings.

## 2. Update Local Config
1.  Open `.env` file in your project folder.
2.  Update `DATABASE_URL` with your Supabase URI:
    ```env
    DATABASE_URL=postgresql://postgres:mysecretpassword@db.abcdefgh.supabase.co:5432/postgres
    ```
    
    *(Make sure there are no spaces around the `=` sign)*

## 3. Migrate Data
Once your `.env` is updated, run the migration script to move your local data to Supabase:
```bash
python migrate_db.py
```

## 4. Verify
Run the test script to ensure connection:
```bash
python test_db_connection.py
```
