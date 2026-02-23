package com.missedcallsaviour.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager
import android.util.Log

class CallReceiver : BroadcastReceiver() {
    companion object {
        private var lastState = TelephonyManager.CALL_STATE_IDLE
        private var isIncoming = false
        private var savedNumber: String? = null
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == "android.intent.action.PHONE_STATE") {
            val stateStr = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
            val number = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)
            
            val state = when (stateStr) {
                TelephonyManager.EXTRA_STATE_IDLE -> TelephonyManager.CALL_STATE_IDLE
                TelephonyManager.EXTRA_STATE_OFFHOOK -> TelephonyManager.CALL_STATE_OFFHOOK
                TelephonyManager.EXTRA_STATE_RINGING -> TelephonyManager.CALL_STATE_RINGING
                else -> TelephonyManager.CALL_STATE_IDLE
            }

            onCallStateChanged(context, state, number)
        }
    }

    private fun onCallStateChanged(context: Context, state: Int, number: String?) {
        if (lastState == state) return

        when (state) {
            TelephonyManager.CALL_STATE_RINGING -> {
                isIncoming = true
                savedNumber = number
                Log.d("CallReceiver", "Incoming call ringing: $number")
            }
            TelephonyManager.CALL_STATE_OFFHOOK -> {
                if (lastState == TelephonyManager.CALL_STATE_RINGING) {
                    isIncoming = true
                    Log.d("CallReceiver", "Call picked up")
                }
            }
            TelephonyManager.CALL_STATE_IDLE -> {
                if (lastState == TelephonyManager.CALL_STATE_RINGING) {
                    Log.d("CallReceiver", "Missed call from: $savedNumber")
                    // Trigger notification or background sync here
                    showMissedCallNotification(context, savedNumber ?: "Unknown")
                } else if (isIncoming) {
                    Log.d("CallReceiver", "Incoming call ended")
                }
                isIncoming = false
            }
        }
        lastState = state
    }

    private fun showMissedCallNotification(context: Context, number: String) {
        // This is where Alex can eventually trigger an automatic SMS or call analysis
        Log.i("Alex", "Missed call detected from $number. Alex is ready to help.")
    }
}
