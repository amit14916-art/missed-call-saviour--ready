# Integration Guide: Make.com & Vapi.ai

Here is how to connect your deployed "Missed Call Saviour" app to the AI services.

## Part 1: Connect Vapi.ai (For AI Calls & SMS)

Your app now communicates directly with Vapi.ai for all voice and messaging needs.

1.  **Get Your Vapi Credentials**:
    *   Log in to [Vapi.ai](https://vapi.ai).
    *   Go to **Account** or **Settings**.
    *   Copy your **Private (Server) API Key**.
    *   Copy your **Assistant ID** (from the Assistants page).
    *   Copy your **Phone Number ID** (from the Phone Numbers page).

2.  **Add to Render**:
    *   Go to your [Render Dashboard](https://dashboard.render.com).
    *   Select your `missed-call-saviour` service.
    *   Go to **"Environment"**.
    *   Add the following variables:
        *   `VAPI_PRIVATE_KEY`: (Paste your key)
        *   `VAPI_ASSISTANT_ID`: (Paste your Assistant ID)
        *   `VAPI_PHONE_NUMBER_ID`: (Paste your Phone Number ID)
    *   Click "Save Changes".

## Part 2: Connect Vapi.ai (Configuring the Server URL)

Vapi needs to know where to send call reports and function calls (like "book appointment").

1.  **Go to Vapi.ai**: Open your Dashboard and select your Assistant.
2.  **Find "Server URL"**: Look for a field named **Server URL** or **Webhook URL** in the Assistant's settings.
3.  **Enter Your App URL**:
    *   Paste this exact URL from your deployed app:
        `https://<your-app-name>.onrender.com/api/vapi/webhook`
4.  **Save**: Click Save in Vapi.

**Now they are connected!**
*   **App -> Vapi**: When you click "Send Demo Call", your App talks directly to Vapi API to start a call.
*   **Vapi -> App**: When a call finishes or an action is needed, Vapi talks to your App (and your App logs it or sends email).
