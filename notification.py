import os
import httpx
import logging

logger = logging.getLogger(__name__)

async def send_owner_notification(caller_number: str, transcript: str, caller_name: str = None):
    """
    Sends a WhatsApp notification to the owner using Twilio.
    """
    owner_phone = os.environ.get("OWNER_WHATSAPP_NUMBER")
    if not owner_phone:
        logger.warning("OWNER_WHATSAPP_NUMBER not set. Skipping notification.")
        return

    name_str = f" from *{caller_name}*" if caller_name else ""
    
    # Pre-process transcript if it's a JSON string list
    formatted_transcript = transcript
    try:
        if transcript.startswith("[") and transcript.endswith("]"):
            import json
            turns = json.loads(transcript)
            formatted_transcript = ""
            for t in turns:
                if isinstance(t, dict):
                    role = "Bot" if t.get("role") == "assistant" else "User"
                    formatted_transcript += f"*{role}:* {t.get('content')}\n"
                else:
                    formatted_transcript += f"- {t}\n"
    except:
        pass

    message = f"""📞 *Missed Call Summary*

Caller{name_str}: {caller_number}

*Conversation:*
{formatted_transcript or 'No transcript recorded'}

_— Missed Call Saviour AI_"""

    provider = os.environ.get("NOTIFICATION_PROVIDER", "twilio")
    if provider == "twilio":
        await _whatsapp(owner_phone, message)
    else:
        logger.info(f"Notification (log only): {message}")

async def _whatsapp(to: str, message: str):
    sid   = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    frm   = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

    if not sid or not token:
        logger.error("Twilio SID/Token missing for WhatsApp notification")
        return

    # Ensure phone number is formatted correctly for Twilio WhatsApp
    if not to.startswith("whatsapp:"):
        if not to.startswith("+"):
            to = f"+{to}"
        to = f"whatsapp:{to}"

    try:
        async with httpx.AsyncClient() as c:
            response = await c.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                auth=(sid, token),
                data={"From": frm, "To": to, "Body": message}
            )
            if response.status_code != 201:
                logger.error(f"Twilio WhatsApp Error {response.status_code}: {response.text}")
            else:
                logger.info(f"✅ WhatsApp sent to {to}")
    except Exception as e:
        logger.error(f"WhatsApp notification exception: {e}")
