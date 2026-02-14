# Deploying to Render

It's super easy and **Free**. Follow these steps:

1.  **Sign Up / Login**
    *   Go to [render.com](https://render.com/).
    *   Login with **GitHub** (this is important, it connects automatically).

2.  **Create New Web Service**
    *   Click **"New +"** (blue button).
    *   Select **"Web Service"**.
    *   Select **"Build and deploy from a Git repository"**
    *   Connect your GitHub repo: `missed-call-saviour`.

3.  **Use These Settings (Important!)**
    Render will auto-detect some, but ensure these are correct:
    *   **Name**: `missed-call-saviour` (or whatever you like)
    *   **Region**: Singapore (closest to India, usually fastest)
    *   **Runtime**: **Python 3**
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4.  **Add Environment Variables**
    Click **"Advanced"** -> **"Add Environment Variable"**. Add these from your `.env` file (OPEN THEM IN VS CODE AND COPY-PASTE):
    *   `SECRET_KEY`: (your secret key)
    *   `STRIPE_SECRET_KEY`: (if you have one, or just put 'dummy')
    *   `MAKE_WEBHOOK_URL`: (your make.com url)
    *   `MAIL_USERNAME`: (your email)
    *   `MAIL_PASSWORD`: (your app password)
    *   `RAZORPAY_KEY_ID`: (your key)
    *   `RAZORPAY_KEY_SECRET`: (your secret)

5.  **Click "Create Web Service"**
    It will deploy! Wait 2-3 minutes.

6.  **Done!**
    You will get a URL like `https://missed-call-saviour.onrender.com`.
    **Copy this URL and give it to Razorpay!** 🚀
