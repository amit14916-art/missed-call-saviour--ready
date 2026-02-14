# How to Configure Make.com for SMS Demo

Your application sends a "signal" to Make.com whenever someone clicks "Send Test" on the demo section. However, **Make.com needs to be told what to do with that signal** (i.e., actually send the SMS).

## Step 1: Open Your Scenario
1. Go to your [Make.com Dashboard](https://www.make.com/en/dashboard).
2. Open the scenario where you created the "Custom Webhook".
3. You should see a module called **Custom Webhook** (likely the starting point).

## Step 2: Add the SMS Module
You need to connect an SMS service to send the text. You can use **Twilio**, **Vonage**, or even the **Android** app integration if you have an Android phone.

### Option A: Using Twilio (Recommended for Business)
1. Click the **plus (+)** button next to the Webhook module to add a new module.
2. Search for **Twilio**.
3. Select **"Create a Message"**.
4. **Connection**: Click "Add" and enter your Twilio Account SID and Auth Token (from your Twilio Console).
5. **From**: Select your Twilio phone number.
6. **To**: Click the field, then drag the **`data: phone`** pill from the Webhook values panel.
   - *Note: You might need to run the "Send Test" button on your website once first so Make.com "learns" the data structure.*
7. **Body**: Click the field and drag the **`data: message`** pill (or type your own message).
8. Click **OK**.

### Option B: Using Android Phone (Free/Personal)
1. Install the **Make** app on your Android phone and sign in.
2. In your Make.com scenario, add a module and search for **Android**.
3. Select **"Send SMS"**.
4. **Phone Number**: Drag the **`data: phone`** pill.
5. **Message**: Drag the **`data: message`** pill.
6. Click **OK**.

## Step 3: Activate the Scenario
1. Click the **Save** icon (floppy disk) at the bottom.
2. Toggle the **Scheduling** switch to **ON** (bottom left).
3. Now, go back to your website and try the "Send Test" button again!

---

## Troubleshooting
- **No Data in Make?** 
  - Go to your website.
  - Click "Send Test".
  - Go to Make.com specific scenario run history.
  - If nothing appears, check if your `MAKE_WEBHOOK_URL` in `.env` matches the one in the Webhook module.
- **Error in Make?**
  - Click on the error bubble details. It usually means the phone number format was wrong (e.g., missing `+1`) or credentials are invalid.
