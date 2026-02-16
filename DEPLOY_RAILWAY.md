# 🚂 Deploying Missed Call Saviour to Railway

Railway is arguably the easiest way to deploy this app because it handles the database connection automatically!

## 📦 Step 1: Push to GitHub
(Ensure your code is on GitHub).
```bash
git add .
git commit -m "Ready for Railway"
git push origin main
```

## 🚀 Step 2: Create Project on Railway
1.  Go to [railway.app](https://railway.app).
2.  Login with **GitHub**.
3.  Click **New Project** -> **Deploy from GitHub repo**.
4.  Select your `missed-call-saviour` repo.
5.  Click **Deploy Now**.

## 🗄️ Step 3: Add Database (Crucial!)
1.  In your project view, click **New** (or right-click) -> **Database** -> **Add PostgreSQL**.
2.  Railway will spin up a database.
3.  Wait for it to be ready.
4.  **The Magic Part**: Railway automatically injects the `DATABASE_URL` environment variable into your app! **You DO NOT need to copy-paste it manually.**

## 🔑 Step 4: Add Environment Variables
1.  Click on your **App Service** block (the box with your repo name).
2.  Go to the **Variables** tab.
3.  Add your secrets (Copy from your local `.env` file):
    *   `VAPI_PRIVATE_KEY`
    *   `VAPI_ASSISTANT_ID`
    *   `VAPI_PHONE_NUMBER_ID`
    *   `MAIL_USERNAME`
    *   `MAIL_PASSWORD`
    *   `SECRET_KEY` (make up a random string)
    *   `PORT` = `8000` (Important: Set this to 8000 to match our Dockerfile)

    *(Razorpay keys are optional)*

## 🌐 Step 5: Public URL
1.  Go to the **Settings** tab of your App Service.
2.  Scroll down to "Networking" / "Public Domain".
3.  Click **Generate Domain**.
4.  Copy this URL (e.g., `missed-call-saviour-production.up.railway.app`).
5.  **Update Variables**: Go back to Variables and add/update `DOMAIN` with this full URL.
6.  **Redeploy**: Use the "Redeploy" button if it doesn't happen automatically.

## ✅ Step 6: Final Connection
1.  Go to **Vapi.ai Dashboard**.
2.  Update your **Server URL** to:
    `https://your-railway-url.up.railway.app/api/vapi/webhook`

**You are LIVE!** 🚂
