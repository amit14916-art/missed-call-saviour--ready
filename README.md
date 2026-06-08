# Missed Call Saviour
# 📞 Missed Call Saviour — Distributed Event Automation SaaS

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/framework-FastAPI-green.svg)
![Architecture](https://img.shields.io/badge/system-Distributed-orange.svg)

A production-ready, highly available backend and frontend architecture designed to neutralize business revenue leakage. The platform catches inbound missed call webhooks, runs asynchronous cognitive evaluation on intent, and fires immediate multi-channel recovery workflows.

---

## 🏗️ Core Architecture & Topology

- **Event Ingestion Layer:** Exposes robust, stateless FastAPI endpoints to ingest webhook payloads from third-party communication networks (Twilio, cloud telephony, etc.) under sub-second latency.
- **Cognitive Intent Resolver:** Passes contextual caller logs into localized or sovereign LLM chains to evaluate customer intent and priority levels dynamically.
- **Workflow State Machine:** Dispatches automated messaging hooks (WhatsApp Business API, SMS Gateways), dynamically renders payment gateways (Razorpay Integration), or schedules calendar slots based on structural responses.
- **Frontend Dashboard:** A clean reactive control panel allowing business administrators to audit active recovery pipelines, log histories, and system health graphs.

---

## 🗂️ High-Level Project Structure

```text
missed_call_saviour/
├── app/
│   ├── main.py              # Application entrypoint & CORS config
│   ├── api/                 # API router endpoints (V1)
│   │   ├── webhooks.py      # Telephony webhook handlers
│   │   └── dashboard.py     # Analytics and metrics metrics
│   ├── core/                # Core security protocols & configurations
│   ├── services/            # Deep-level LLM and external API bindings
│   └── models/              # System state database tables
├── frontend/                # Interactive management portal assets
├── Dockerfile               # High-efficiency container definition
└── requirements.txt         # Runtime dependencies
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
