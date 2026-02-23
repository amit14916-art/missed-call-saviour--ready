package com.missedcallsaviour.app.network

import com.missedcallsaviour.app.data.*
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.*

interface ApiService {
    @FormUrlEncoded
    @POST("token")
    suspend fun login(
        @Field("username") email: String,
        @Field("password") pass: String
    ): Response<LoginResponse>

    @GET("api/dashboard/stats")
    suspend fun getStats(@Header("Authorization") token: String): Response<DashboardStats>

    @GET("api/calls")
    suspend fun getHistory(@Header("Authorization") token: String): Response<List<CallLog>>

    @POST("api/analyze-chat")
    suspend fun sendChat(
        @Header("Authorization") token: String,
        @Body request: ChatRequest
    ): Response<ChatResponse>

    @Multipart
    @POST("api/upload-call-recording")
    suspend fun uploadRecording(
        @Header("Authorization") token: String,
        @Part file: MultipartBody.Part
    ): Response<Map<String, String>>
}
