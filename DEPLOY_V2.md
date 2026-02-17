# Deploying Version 2.0 (AI Persona + Audio Player)

Great news! The local version of your app now has:
1.  **AI Persona Selector** (Friendly, Professional, Urgent)
2.  **Audio Player** for call recordings
3.  **Transcript Viewer**

However, the screenshot you shared is from the **Live Website**, which is still running the old version. The "Error updating settings" you see there is likely because the old version is missing some configuration on Render.

## Step 1: Push Changes to GitHub

To update the live site, run these commands in your terminal:

```bash
git add .
git commit -m "feat: add AI persona settings and audio player"
git push origin main
```

## Step 2: Update Render Configuration

For the new features to work perfectly on the live site, ensure these **Environment Variables** are set in your Render Dashboard:

1.  Go to [Render Dashboard](https://dashboard.render.com) -> Select your service.
2.  Click **"Environment"**.
3.  Ensure these keys are present and correct:
    *   `VAPI_PRIVATE_KEY` (starts with `sk-`)
    *   `VAPI_ASSISTANT_ID`
    *   `VAPI_PHONE_NUMBER_ID`
    *   `DATABASE_URL` (if using PostgreSQL, otherwise it uses a temporary SQLite file that resets on deploy)

## Step 3: Verify

Once Render finishes building (usually 2-3 minutes), refresh your dashboard.
- You should see the **"Agent Persona"** dropdown.
- Listening to recordings will now work directly in the browser!
