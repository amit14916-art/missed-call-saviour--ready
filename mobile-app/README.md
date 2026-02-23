# Missed Call Saviour - Mobile App 📱

This is the mobile application (Android/iOS) for Missed Call Saviour.
It allows the app to detect missed calls automatically and sync them with the AI backend.

## Tech Stack
- **Framework:** React Native (Expo)
- **Language:** JavaScript/TypeScript
- **Features:**
  - **Manual Call Recording & Upload** (Works in Expo Go)
  - **Dashboard Webview** (Works in Expo Go)
  - **Missed Call Detection** (Requires Development Build)

## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    cd mobile-app
    npm install
    # To enable Missed Call Detection (Native Module):
    # npx expo install react-native-call-log
    ```

2.  **Run on Android Device (Simple Mode):**
    - Download **Expo Go** app from Play Store.
    - Run: `npx expo start`
    - Scan QR code.
    - *Note:* Call Log detection will NOT work in this mode. Only Manual Recording.

3.  **Run with Native Features (Advanced):**
    - Potentially requires `npx expo prebuild` and Android Studio.
    - Run: `npx expo run:android`

## Permissions
- `READ_CALL_LOG`: To detect missed calls (Restricted in Expo Go).
- `READ_PHONE_STATE`: To detect incoming call status.
- `INTERNET`: To send data to the backend.
- `RECORD_AUDIO`: To record calls manually.

## Note on Call Recording
Modern Android versions (Android 10+) **block** third-party apps from recording internal audio (the other person's voice) during native phone calls for privacy reasons. 
However, this app **CAN**:
1.  Detect when a call is missed.
2.  Automatically trigger the AI to call the person back (Server-side recording).
3.  Record calls *made through this app* (VoIP) or external audio (Speakerphone).
