# 🚀 Deployment Guide: Missed Call Saviour

This guide helps you deploy your app to **Render** using Docker. This ensures your app runs exactly like it does on your machine.

---

## 🔒 Step 0: Security First
Before pushing code, **make sure your `.env` file is ignored**.
1.  Open `.gitignore`.
2.  Ensure it has a line that says `.env`.
3.  If not, add it! **Never push your API keys to GitHub.**

---

## 📦 Step 1: Push to GitHub
1.  Open your terminal.
2.  Run these commands to save your changes (including the new Dockerfile):
    ```bash
    git add .
    git commit -m "Ready for deploy"
    git push origin main
    ```
    *(If you don't have a repo yet, go to github.com -> New Repository -> Copy the push commands)*

---

## 🗄️ Step 2: Create a Database (PostgreSQL)
*Since Render deletes files on restart, we need a real database to save call logs.*

1.  Log in to [Render.com](https://render.com).
2.  Click **New +** > **PostgreSQL**.
3.  Name: `missed-call-db`.
4.  Region: **Singapore** (Best for India).
5.  Plan: **Free** (Sandbox).
6.  Click **Create Database**.
7.  **Copy** the `Internal Database URL` when it's ready.

---

## ☁️ Step 3: Deploy the App
1.  Click **New +** > **Web Service**.
2.  Select **"Build and deploy from a Git repository"**.
3.  Connect your repo.
4.  **Runtime**: Select **Docker** (Important!).
5.  **Region**: Singapore.
6.  **Instance Type**: Free.
7.  **Environment Variables** (Advanced Section):
    Click "Add Environment Variable" for each:

    | Key | Value |
    | :--- | :--- |
    | `DATABASE_URL` | *(Paste your Postgres URL from Step 2)* |
    | `VAPI_PRIVATE_KEY` | *(Your Vapi Private Key)* |
    | `VAPI_ASSISTANT_ID` | *(Your Vapi Assistant ID)* |
    | `VAPI_PHONE_NUMBER_ID` | *(Your Vapi Phone Number ID)* |
    | `MAIL_USERNAME` | *(Your Gmail)* |
    | `MAIL_PASSWORD` | *(Your App Password)* |
    | `SECRET_KEY` | `any-random-string-you-like` |
    | `DOMAIN` | `https://your-app-name.onrender.com` (Update this after deploy) |

    *(Razorpay keys are optional if not using currently)*

8.  Click **Create Web Service**.

---

## ✅ Step 4: Verify
1.  Wait for the build to finish (5-10 mins).
2.  Visit your URL/dashboard!
3.  **Final Touch**: Go to Vapi.ai and update your "Server URL" to:
    `https://your-new-app-url.onrender.com/api/vapi/webhook`

**Enjoy your live SaaS!** 🚀
