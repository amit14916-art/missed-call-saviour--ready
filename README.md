# Missed Call Saviour

This is the backend and frontend for the Missed Call Saviour application.

## Project Structure

- `main.py`: The main FastAPI application entry point.
- `index.html`: The landing page served at `/`.
- `dashboard.html`: The main dashboard view.
- `login.html`, `signup.html`: Authentication pages.
- `missed_calls.db`: SQLite database for storing user data.

## Running the Application

To run the server, use:

```bash
python -m uvicorn main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## Key Features

- **FastAPI Backend**: Handles API requests and serves HTML templates.
- **SQLite Database**: Stores users and payments.
- **Stripe Integration**: Mock payment processing.
- **Vapi Integration**: Webhook endpoint for AI calls.
- **Make.com Integration**: Triggers external workflows.

## Important Notes

Please ensure you are editing the files within this directory (`missed-call-saviour`), NOT the ones in the parent `Downloads` directory.
