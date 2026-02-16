# How to Get Your Razorpay API Keys

To connect your "Missed Call Saviour" app to Razorpay for payments, you need to get your **Key ID** and **Key Secret**.

## Step 1: Log in to Razorpay
1.  Go to [https://dashboard.razorpay.com](https://dashboard.razorpay.com) and log in.
2.  **Top Menu**: Ensure you are in **Test Mode** for now (toggle is usually at the top right).
    *   *Note: When you are ready to take real money, switch to Live Mode and generate new keys.*

## Step 2: Navigate to Settings
1.  In the left sidebar menu, scroll down to the bottom.
2.  Click on **Settings** (gear icon).

## Step 3: Generate API Keys
1.  In the Settings menu tabs, click on **API Keys**.
2.  Click the button **Generate Test Key** (or "Regenerate Test Key" if you already had one).
3.  **IMPORTANT:** A popup will appear showing your `Key ID` and `Key Secret`.
    *   **Download** them or **Copy** them immediately.
    *   You will **NOT** be able to see the `Key Secret` again after you close this popup.

## Step 4: Add to Your Project
1.  Open your `.env` file in the project folder.
2.  Add or update these lines:

```env
RAZORPAY_KEY_ID=rzp_test_... (paste your Key ID)
RAZORPAY_KEY_SECRET=... (paste your Key Secret)
```

3.  **Restart your backend**:
    *   If your server is running, stop it (Ctrl+C).
    *   Run `python main.py` again.

## Step 5: Test It
1.  Go to your app's **Pricing Page** (`/pricing`).
2.  Click "Get Started" on the Starter Plan.
3.  The Razorpay checkout popup should now appear!
