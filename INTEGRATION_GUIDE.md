# Integration Guide: Make.com & Vapi.ai

Here is how to connect your deployed "Missed Call Saviour" app to the AI services.

## Part 1: Connect Make.com (For Sending SMS)

Your app sends a "signal" to Make.com whenever you need to send an SMS (e.g., "Send Demo SMS").

1.  **Go to Make.com**: Create a new Scenario.
2.  **Add Trigger**: Search for **"Webhooks"** -> **"Custom Webhook"**.
3.  **Create Webhook**: Click "Add", name it `MissedCallSaviour`, and click "Save".
4.  **Copy URL**: Copy the URL it gives you (e.g., `https://hook.us1.make.com/...`).
5.  **Add to Render**:
    *   Go to your [Render Dashboard](https://dashboard.render.com).
    *   Select your `missed-call-saviour` service.
    *   Go to **"Environment"**.
    *   Add a new variable:
        *   **Key**: `MAKE_WEBHOOK_URL`
        *   **Value**: (The URL you copied from Make.com)
    *   Click "Save Changes".

## Part 2: Connect Vapi.ai (For AI Calls)

Vapi needs to know where to send call reports and function calls (like "book appointment").

1.  **Go to Vapi.ai**: Open your Dashboard and select your Assistant.
2.  **Find "Server URL"**: Look for a field named **Server URL** or **Webhook URL** in the Assistant's settings.
3.  **Enter Your App URL**:
    *   Paste this exact URL:
        `https://missed-call-saviours.onrender.com/api/vapi/webhook`
4.  **Save**: Click Save in Vapi.

**Now they are connected!**
*   **App -> Make.com**: When you click "Send Demo SMS", your App talks to Make.com.
*   **Vapi -> App**: Even a call finishes, Vapi talks to your App (and your App sends you an email summary!).
