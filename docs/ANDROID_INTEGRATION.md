# Sereluna Android Integration

Backend production:

```text
https://larmelar-sereluna-backend.hf.space
```

Swagger:

```text
https://larmelar-sereluna-backend.hf.space/docs
```

All authenticated API calls must send a Firebase ID token:

```http
Authorization: Bearer <firebase_id_token>
```

## 1. Android Setup

In Android Studio, add Firebase Auth first. Typical Gradle setup:

```kotlin
dependencies {
    implementation(platform("com.google.firebase:firebase-bom:34.0.0"))
    implementation("com.google.firebase:firebase-auth")

    implementation("com.squareup.retrofit2:retrofit:2.11.0")
    implementation("com.squareup.retrofit2:converter-gson:2.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.8.1")
}
```

Make sure your app already has `google-services.json` and applies the Google Services plugin.

## 2. Firebase Login

Example email/password login:

```kotlin
import com.google.firebase.auth.FirebaseAuth
import kotlinx.coroutines.tasks.await

class AuthRepository(
    private val auth: FirebaseAuth = FirebaseAuth.getInstance()
) {
    suspend fun login(email: String, password: String): String {
        auth.signInWithEmailAndPassword(email, password).await()
        return getIdToken()
    }

    suspend fun register(email: String, password: String): String {
        auth.createUserWithEmailAndPassword(email, password).await()
        return getIdToken()
    }

    suspend fun getIdToken(forceRefresh: Boolean = false): String {
        val user = auth.currentUser ?: error("User is not logged in")
        return user.getIdToken(forceRefresh).await().token
            ?: error("Firebase ID token is empty")
    }

    fun logout() {
        auth.signOut()
    }
}
```

Use the returned token as:

```kotlin
"Bearer $token"
```

If a request returns `401`, refresh token once with:

```kotlin
authRepository.getIdToken(forceRefresh = true)
```

## 3. Retrofit Client

```kotlin
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object ApiClient {
    private const val BASE_URL = "https://larmelar-sereluna-backend.hf.space/"

    private val logging = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val httpClient = OkHttpClient.Builder()
        .addInterceptor(logging)
        .build()

    val api: SerelunaApi = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(httpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(SerelunaApi::class.java)
}
```

## 4. API Interface

```kotlin
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.PATCH
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.http.Query

interface SerelunaApi {
    @GET("/")
    suspend fun health(): HealthResponse

    @GET("api/v1/me/context/")
    suspend fun getContext(
        @Header("Authorization") auth: String
    ): UserContextResponse

    @GET("api/v1/me/profile/")
    suspend fun getProfile(
        @Header("Authorization") auth: String
    ): ProfileResponse

    @PUT("api/v1/me/profile/")
    suspend fun updateProfile(
        @Header("Authorization") auth: String,
        @Body body: ProfileUpdateRequest
    ): ProfileResponse

    @POST("api/v1/auth/forgot-password/")
    suspend fun forgotPassword(
        @Body body: ForgotPasswordRequest
    ): ForgotPasswordResponse

    @POST("api/v1/chat/")
    suspend fun chat(
        @Header("Authorization") auth: String,
        @Body body: ChatRequest
    ): ChatResponse

    @POST("api/v1/chat/finish/")
    suspend fun finishChat(
        @Header("Authorization") auth: String,
        @Body body: ChatFinishRequest
    ): ChatResponse

    @POST("api/v1/screening/")
    suspend fun submitScreening(
        @Header("Authorization") auth: String,
        @Body body: ScreeningRequest
    ): ScreeningResponse

    @GET("api/v1/diaries/")
    suspend fun getDiaries(
        @Header("Authorization") auth: String,
        @Query("limit") limit: Int = 30
    ): DiaryListResponse

    @GET("api/v1/diaries/{diaryId}/")
    suspend fun getDiaryDetail(
        @Header("Authorization") auth: String,
        @Path("diaryId") diaryId: String
    ): DiaryDetailResponse

    @GET("api/v1/diaries/{diaryId}/sessions/{sessionId}/messages/")
    suspend fun getDiaryMessages(
        @Header("Authorization") auth: String,
        @Path("diaryId") diaryId: String,
        @Path("sessionId") sessionId: String
    ): DiaryMessagesResponse

    @GET("api/v1/notifications/")
    suspend fun getNotifications(
        @Header("Authorization") auth: String,
        @Query("limit") limit: Int = 30
    ): NotificationListResponse

    @PATCH("api/v1/notifications/{notificationId}/read/")
    suspend fun markNotificationRead(
        @Header("Authorization") auth: String,
        @Path("notificationId") notificationId: String
    ): SuccessResponse

    @POST("api/v1/device-token/")
    suspend fun registerDeviceToken(
        @Header("Authorization") auth: String,
        @Body body: DeviceTokenRequest
    ): DeviceTokenResponse

    @POST("api/v1/sleep/daily/")
    suspend fun saveSleepDaily(
        @Header("Authorization") auth: String,
        @Body body: SleepDailyRequest
    ): SleepDailyResponse

    @GET("api/v1/sleep/daily/")
    suspend fun getSleepDaily(
        @Header("Authorization") auth: String,
        @Query("limit") limit: Int = 14
    ): SleepDailyListResponse
}
```

## 5. Data Classes

```kotlin
data class HealthResponse(
    val message: String
)

data class ProfileUpdateRequest(
    val name: String? = null,
    val photo_url: String? = null
)

data class ForgotPasswordRequest(
    val email: String,
    val continue_url: String? = null
)

data class ForgotPasswordResponse(
    val message: String,
    val reset_link: String? = null
)

data class ProfileResponse(
    val uid: String,
    val name: String = "",
    val email: String = "",
    val photo_url: String = "",
    val provider: String = "",
    val latest_screening_summary: String = "",
    val latest_diary_summary: String = "",
    val personal_context: String = "",
    val has_screening_today: Boolean = false,
    val created_at: String? = null,
    val updated_at: String? = null
)

data class UserContextResponse(
    val profile_context: String,
    val latest_screening_summary: String,
    val latest_diary_summary: String,
    val past_diaries: List<String> = emptyList(),
    val has_screening_today: Boolean
)

data class ChatRequest(
    val text: String,
    val room_id: String? = null,
    val session_id: String? = null,
    val mood_signal: String = "",
    val mode: String = "chat"
)

data class ChatFinishRequest(
    val room_id: String,
    val session_id: String
)

data class ChatResponse(
    val reply: String,
    val ui_metadata: UIMetadata,
    val clinical_insight: ClinicalInsight,
    val session_summary: String,
    val room_id: String? = null,
    val session_id: String? = null,
    val algorithm_trace: Map<String, Any>? = null
)

data class UIMetadata(
    val sentiment_score: Int,
    val suggested_action: String? = null,
    val is_risky: Boolean
)

data class ClinicalInsight(
    val detected_symptoms: List<String> = emptyList(),
    val dass_category: String = "None",
    val risk_level: String = "low"
)

data class ScreeningRequest(
    val answers: List<Int>,
    val note: String = ""
)

data class ScreeningResponse(
    val date: String,
    val scores: Map<String, Int>,
    val severity: Map<String, String>,
    val summary: String,
    val algorithm: Map<String, Any> = emptyMap(),
    val has_screening_today: Boolean = true
)

data class DiaryListResponse(
    val items: List<DiaryItem> = emptyList()
)

data class DiaryItem(
    val id: String,
    val date: String = "",
    val chat_summary: String = "",
    val created_at: String? = null,
    val updated_at: String? = null
)

data class DiaryDetailResponse(
    val id: String,
    val date: String = "",
    val chat_summary: String = "",
    val sessions: List<DiarySessionItem> = emptyList()
)

data class DiarySessionItem(
    val id: String,
    val model: String = "",
    val summary: String = "",
    val start_time: String? = null,
    val end_time: String? = null
)

data class DiaryMessagesResponse(
    val items: List<DiaryMessageItem> = emptyList()
)

data class DiaryMessageItem(
    val id: String,
    val sender_role: String,
    val text: String = "",
    val timestamp: String? = null
)

data class NotificationListResponse(
    val items: List<NotificationItem> = emptyList()
)

data class NotificationItem(
    val id: String,
    val title: String = "",
    val body: String = "",
    val type: String = "",
    val is_read: Boolean = false,
    val created_at: String? = null
)

data class SuccessResponse(
    val success: Boolean = true
)

data class DeviceTokenRequest(
    val token: String
)

data class DeviceTokenResponse(
    val success: Boolean = true
)

data class SleepDailyRequest(
    val date: String,
    val sleep_quality: String,
    val total_sleep_hours: Float
)

data class SleepDailyResponse(
    val success: Boolean = true
)

data class SleepDailyListResponse(
    val items: List<SleepDailyItem> = emptyList()
)

data class SleepDailyItem(
    val date: String,
    val sleep_quality: String = "",
    val total_sleep_hours: Float = 0f,
    val updated_at: String? = null
)
```

## 6. Common Calls

Login then call profile:

```kotlin
val token = authRepository.login(email, password)
val authHeader = "Bearer $token"
val profile = ApiClient.api.getProfile(authHeader)
```

Forgot password:

```kotlin
val result = ApiClient.api.forgotPassword(
    ForgotPasswordRequest(
        email = "etyamaaf@gmail.com",
        continue_url = "https://larmelar-sereluna-backend.hf.space/"
    )
)

// If reset_link is not null, open it in a browser or custom tab.
```

Send chat:

```kotlin
val token = authRepository.getIdToken()
val response = ApiClient.api.chat(
    auth = "Bearer $token",
    body = ChatRequest(
        text = "Aku lagi cemas banget hari ini",
        mood_signal = "anxious"
    )
)

val replyText = response.reply
val roomId = response.room_id
val sessionId = response.session_id
```

Continue the same chat session:

```kotlin
val response = ApiClient.api.chat(
    auth = "Bearer ${authRepository.getIdToken()}",
    body = ChatRequest(
        text = "Aku susah tidur juga",
        room_id = roomId,
        session_id = sessionId,
        mood_signal = "anxious"
    )
)
```

Finish chat session:

```kotlin
ApiClient.api.finishChat(
    auth = "Bearer ${authRepository.getIdToken()}",
    body = ChatFinishRequest(
        room_id = roomId ?: return,
        session_id = sessionId ?: return
    )
)
```

Submit DASS-21 screening:

```kotlin
val answers = listOf(
    0, 1, 2, 1, 0, 1, 2,
    1, 0, 1, 2, 1, 0, 1,
    2, 1, 0, 1, 2, 1, 0
)

val result = ApiClient.api.submitScreening(
    auth = "Bearer ${authRepository.getIdToken()}",
    body = ScreeningRequest(
        answers = answers,
        note = "Screening awal"
    )
)
```

Register FCM token:

```kotlin
ApiClient.api.registerDeviceToken(
    auth = "Bearer ${authRepository.getIdToken()}",
    body = DeviceTokenRequest(token = fcmToken)
)
```

Save daily sleep:

```kotlin
ApiClient.api.saveSleepDaily(
    auth = "Bearer ${authRepository.getIdToken()}",
    body = SleepDailyRequest(
        date = "2026-05-19",
        sleep_quality = "good",
        total_sleep_hours = 7.5f
    )
)
```

## 7. Endpoint Summary

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/` | No | Health check |
| GET | `/api/v1/me/context/` | Yes | User context for home/chat |
| GET | `/api/v1/me/profile/` | Yes | Get current user profile |
| PUT | `/api/v1/me/profile/` | Yes | Update profile |
| POST | `/api/v1/auth/forgot-password/` | No | Generate Firebase password reset link |
| POST | `/api/v1/chat/` | Yes | Send chat message |
| POST | `/api/v1/chat/finish/` | Yes | Finish chat session |
| POST | `/api/v1/screening/` | Yes | Save DASS-21 screening |
| GET | `/api/v1/diaries/` | Yes | List diaries |
| GET | `/api/v1/diaries/{diaryId}/` | Yes | Diary detail |
| GET | `/api/v1/diaries/{diaryId}/sessions/{sessionId}/messages/` | Yes | Chat messages in a diary session |
| GET | `/api/v1/notifications/` | Yes | List notifications |
| PATCH | `/api/v1/notifications/{notificationId}/read/` | Yes | Mark notification as read |
| POST | `/api/v1/device-token/` | Yes | Save FCM token |
| POST | `/api/v1/sleep/daily/` | Yes | Save daily sleep metric |
| GET | `/api/v1/sleep/daily/` | Yes | List sleep metrics |

## 8. Notes

- Use HTTPS only. No Android cleartext config is needed.
- `answers` in screening must contain exactly 21 integers.
- `sleep.date` must use `yyyy-MM-dd`.
- Save `room_id` and `session_id` from the first chat response if the user continues the same session.
- Do not put `GROQ_API_KEY`, Firebase service account JSON, or Hugging Face token in the APK.
- User authentication is Firebase Auth on Android; backend only verifies the Firebase ID token.
