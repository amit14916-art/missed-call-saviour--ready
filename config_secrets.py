
# ⚠️ HARDCODED SECRETS FOR DEPLOYMENT FIX ⚠️
# Ideally these should be environment variables, but Railway is failing to inject them.
# We are placing them here to force the application to work.

DATABASE_URL = "postgresql://postgres.anfrcjrpuhfhiuadzytl:StrongAndSafePassword2026forSupabaseOnly@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres"

VAPI_PRIVATE_KEY = "f73a207c-c3f7-47c3-afab-e679cd5400b8"
VAPI_ASSISTANT_ID = "47f672ff-38e1-4590-9d86-a84c5db45bbc"
VAPI_PHONE_NUMBER_ID = "c7394be1-2357-4e8b-a891-30029f071f55"

# Razorpay Keys (Live)
RAZORPAY_KEY_ID = "rzp_live_SHIHekjovSvcCU"
RAZORPAY_KEY_SECRET = "tjemjJE04zEUjV1ez8hyOOIY"

# Google Gemini API Key (Loaded from environment variables in main.py)
GEMINI_API_KEY = "" # Set in Railway/Local Env
