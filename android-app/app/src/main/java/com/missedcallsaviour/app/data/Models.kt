package com.missedcallsaviour.app.data

data class LoginResponse(val access_token: String, val token_type: String)
data class DashboardStats(
    val missed_calls_saved: Int,
    val est_revenue: Float,
    val weekly_volume: List<Int>,
    val weekly_labels: List<String>
)
data class CallLog(
    val id: Int,
    val phone_number: String,
    val summary: String,
    val transcript: String?,
    val timestamp: String
)
data class ChatRequest(val message: String, val session_id: String)
data class ChatResponse(val reply: String)
